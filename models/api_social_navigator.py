"""
Social Navigator Module for LOVON
==================================
Placeholder class for socially-aware velocity modulation.
Sits between _update_motion_control() and _control_robot() in deploy.py.

Current behavior: passthrough (no velocity modification).
TODO: Enable social cost computation and velocity modulation.
"""

from __future__ import annotations
import numpy as np
import time
import math
import logging
from typing import Dict, List, Optional, Tuple

from models.humantrajectorypredictor import HumanTrajectoryPredictor

logger = logging.getLogger("SocialNavigator")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        "[%(name)s %(levelname)s %(asctime)s] %(message)s", datefmt="%H:%M:%S"
    ))
    logger.addHandler(_handler)


class TrackedHuman:
    """State for a single tracked human."""

    def __init__(self, track_id):
        self.track_id = track_id
        self.position_image = None       # (cx, cy) in pixels
        self.bbox = None                 # (x1, y1, x2, y2) in pixels
        self.keypoints = None            # (17, 2) COCO keypoints
        self.keypoints_conf = None       # (17,) confidence per keypoint
        self.confidence = 0.0

        # --- Distance estimation ---
        self.distance_lidar = None       # meters, from lidar
        self.distance_mono = None        # meters, from monocular approx
        self.distance = None             # meters, fused best-estimate

        # --- Robot-frame metric position ---
        self.position_rf = None          # [x_lateral, depth] in meters

        # --- Tracking / prediction ---
        self.velocity = None             # [vx, vy] in robot frame, m/s
        self.predicted_path = None       # list of [x_lateral, depth] future positions
        self.orientation = None          # radians, body heading
        self.last_seen = time.time()

    def __repr__(self):
        d = "{:.2f}m".format(self.distance) if self.distance is not None else "?"
        return "<Human id={} dist={} conf={:.2f}>".format(
            self.track_id, d, self.confidence
        )

# ====================================================================== #
#  Social Navigator                                                       #
# ====================================================================== #

class SocialNavigator:
    """
    Socially-aware navigation layer for LOVON.

    Usage in deploy.py
    ------------------
    # In VisualLanguageController.__init__():
        self.social_nav = SocialNavigator()

    # In _update_motion_control(), after prediction:
        self.motion_vector = self.social_nav.step(
            motion_vector=self.motion_vector,
            pose_state=self.pose_state,
            mission_state=self.state["mission_state_in"],
            lidar_ranges=None,
        )
    """

    # ------------------------------------------------------------------ #
    #  Tunable parameters (placeholder defaults from architecture spec)
    # ------------------------------------------------------------------ #
    DEFAULT_PARAMS = {
        "sigma": 0.8,
        "h": 1.0,
        "lambda_v": 0.5,
        "beta_traj": 0.3,
        "d_safe": 0.8,
        "d_yield": 1.2,
        "d_resume": 2.0,
        "horizon_s": 2.0,
        "horizon_steps": 10,
        "k_avoid": 0.3,
        "v_min_scale": 0.1,
        "d_max": 2.5,
        "mono_k": 300.0,       # placeholder: d approx mono_k / bbox_height_px
        # --- Camera intrinsics (Go2 front camera, 120deg FoV) ---
        "image_width": 640,
        "fov_deg": 120.0,
        # --- Trajectory prediction ---
        "pred_history": 5,
        "pred_steps": 10,
        "pred_interval": 1,    # predict every frame
        # --- ByteTrack tracker ---
        "track_high_thresh": 0.5,   # confidence >= this → first association
        "track_low_thresh": 0.1,    # confidence >= this → second association
        "track_iou_thresh": 0.3,    # minimum IoU to accept a match
        "track_max_lost": 30,       # frames before a lost track is removed
    }

    def __init__(self, enabled=False, **kwargs):
        self.enabled = enabled
        self.params = {**self.DEFAULT_PARAMS, **kwargs}

        # --- Derived camera intrinsics ---
        half_fov = math.radians(self.params["fov_deg"] / 2.0)
        self._fx = (self.params["image_width"] / 2.0) / math.tan(half_fov)
        self._cx = self.params["image_width"] / 2.0

        # --- Tracked human registry ---
        self._tracked_humans = {}  # type: Dict[int, TrackedHuman]
        self._next_id = 0
        self._byte_tracks = []     # type: List[dict]  # internal ByteTrack state

        # --- Trajectory predictor ---
        self._predictor = HumanTrajectoryPredictor(
            history_length=self.params["pred_history"],
            prediction_steps=self.params["pred_steps"],
            prediction_interval=self.params["pred_interval"],
        )
        self._frame_count = 0

        # --- Yielding state ---
        self.is_yielding = False
        self._yield_enter_time = None
        self._clear_frames = 0

        # --- Motion vectors (for BEV drawing) ---
        self._motion_original = None
        self._motion_modulated = None

        # --- Diagnostics ---
        self.diag = {
            "num_humans": 0,
            "min_distance": None,
            "total_prox_cost": 0.0,
            "total_traj_cost": 0.0,
            "speed_scale": 1.0,
            "is_yielding": False,
        }

        logger.info(
            "SocialNavigator initialized  enabled=%s  fx=%.1f  cx=%.1f",
            self.enabled, self._fx, self._cx,
        )

    # ================================================================== #
    #  PUBLIC API                                                         #
    # ================================================================== #

    def step(
        self,
        motion_vector,
        pose_state,       # type: dict
        mission_state,    # type: str
        lidar_ranges=None,
    ):
        """
        Main entry point -- called once per control cycle.

        Args:
            motion_vector : list/array [v_x, v_y, omega_z] from L2MM
            pose_state    : dict from controller.pose_state
                            {"num_people", "poses", "pose_boxes"}
            mission_state : str, current state machine state
            lidar_ranges  : reserved -- raw lidar scan (not yet wired)

        Returns:
            motion_vector : [v_x, v_y, omega_z]  (unmodified for now)
        """
        if not self.enabled:
            return motion_vector

        self._frame_count += 1

        # 1. Parse detections from pose_state
        detections = self._parse_pose_state(pose_state)

        # 2. Estimate distances + compute robot-frame positions
        self._estimate_distances(detections, lidar_ranges)

        # 3. Update tracker (simple ID assignment for now)
        self._update_tracker(detections)

        # 4. Predict future trajectories
        self._predict_trajectories()

        # 5. Compute social costs (placeholder -- returns zeros)
        prox_cost, traj_cost = self._compute_social_costs()

        # 6. Evaluate yielding conditions (placeholder)
        self._evaluate_yielding(mission_state)

        # 7. Modulate velocity (DISABLED -- passthrough)
        modified_vector = self._modulate_velocity(motion_vector, prox_cost, traj_cost)

        # Store both for BEV visualisation
        self._motion_original = list(motion_vector)
        self._motion_modulated = list(modified_vector)

        # 8. Update diagnostics
        self._update_diagnostics(prox_cost, traj_cost)

        return modified_vector

    # ================================================================== #
    #  STAGE 1 -- Parse pose_state into detection dicts                   #
    # ================================================================== #

    def _parse_pose_state(self, pose_state):
        # type: (dict) -> List[dict]
        """Convert controller.pose_state into a list of detection dicts."""
        detections = []
        num = pose_state.get("num_people", 0)
        poses = pose_state.get("poses", [])
        boxes = pose_state.get("pose_boxes", [])

        for i in range(num):
            det = {}
            if i < len(boxes):
                x1, y1, x2, y2 = boxes[i]
                det["bbox"] = (x1, y1, x2, y2)
                det["center_px"] = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                det["bbox_height"] = float(y2 - y1)
            if i < len(poses):
                det["keypoints"] = np.array(poses[i]["keypoints"])      # (17, 2)
                det["keypoints_conf"] = poses[i].get("keypoints_conf")
                det["confidence"] = poses[i].get("confidence", 0.0)
            detections.append(det)

        return detections

    # ================================================================== #
    #  STAGE 2 -- Distance estimation + robot-frame projection            #
    # ================================================================== #

    def _estimate_distances(self, detections, lidar_ranges=None):
        # type: (List[dict], ...) -> None
        """Populate each detection with distance estimate + robot-frame position."""
        for det in detections:
            # --- Lidar path (placeholder) ---
            det["distance_lidar"] = self._estimate_distance_lidar(det, lidar_ranges)

            # --- Monocular path ---
            det["distance_mono"] = self._estimate_distance_mono(det)

            # --- Fuse: prefer lidar, fall back to mono ---
            if det["distance_lidar"] is not None:
                det["distance"] = det["distance_lidar"]
            elif det["distance_mono"] is not None:
                det["distance"] = det["distance_mono"]
            else:
                det["distance"] = None

            # --- Compute robot-frame 2D position ---
            det["position_rf"] = self._pixel_to_robot_frame(det)

    def _estimate_distance_lidar(self, det, lidar_ranges):
        # type: (dict, ...) -> Optional[float]
        if lidar_ranges is None:
            return None
        return None

    def _estimate_distance_mono(self, det):
        # type: (dict) -> Optional[float]
        """Monocular depth: d approx mono_k / bbox_height_px"""
        h = det.get("bbox_height", 0)
        if h < 10:
            return None
        return self.params["mono_k"] / h

    def _pixel_to_robot_frame(self, det):
        # type: (dict) -> Optional[List[float]]
        """
        Convert pixel detection + depth into robot-frame [x_lateral, depth].

        Uses pinhole camera model:
            x_lateral = depth * (u - cx) / fx

        Coordinate convention (robot frame):
            x_lateral: positive = right of robot
            depth:     positive = forward from robot

        Camera: Go2 front camera, 120deg FoV.
            fx = (image_width / 2) / tan(FoV / 2)
               = 320 / tan(60deg) ~ 184.8 px  (at 640 width)
            cx = image_width / 2 = 320

        NOTE: fx and cx should be calibrated on actual hardware.
        """
        depth = det.get("distance")
        center = det.get("center_px")

        if depth is None or center is None:
            return None

        u = center[0]  # horizontal pixel coordinate
        x_lateral = depth * (u - self._cx) / self._fx

        return [x_lateral, depth]

    # ================================================================== #
    #  STAGE 3 -- ByteTrack IoU tracker                                   #
    # ================================================================== #

    def _update_tracker(self, detections):
        # type: (List[dict]) -> None
        """
        ByteTrack IoU-based tracker for persistent multi-human tracking.

        Algorithm (adapted from ByteTrack, Zhang et al. 2022):
          1. Split detections into high-conf and low-conf groups.
          2. First association: high-conf dets vs active tracks (IoU).
          3. Second association: low-conf dets vs remaining active tracks.
          4. Third association: remaining high-conf dets vs lost tracks
             (re-identification).
          5. Unmatched high-conf dets become new tracks.
          6. Unmatched active tracks become lost; lost tracks exceeding
             track_max_lost frames are removed.
        """
        now = time.time()
        high_thresh = self.params["track_high_thresh"]
        low_thresh = self.params["track_low_thresh"]
        iou_thresh = self.params["track_iou_thresh"]
        max_lost = self.params["track_max_lost"]

        # --- Only track detections that have a bounding box ---
        valid_dets = [(i, d) for i, d in enumerate(detections)
                      if d.get("bbox") is not None]

        # --- Split by confidence ---
        high_dets = [(i, d) for i, d in valid_dets
                     if d.get("confidence", 0.0) >= high_thresh]
        low_dets = [(i, d) for i, d in valid_dets
                    if low_thresh <= d.get("confidence", 0.0) < high_thresh]

        # --- Partition existing tracks ---
        active_tracks = [t for t in self._byte_tracks if t["state"] == "active"]
        lost_tracks = [t for t in self._byte_tracks if t["state"] == "lost"]

        # === FIRST ASSOCIATION: high-conf dets vs active tracks ===
        matches_1, unmatch_det_1, unmatch_trk_1 = self._associate(
            high_dets, active_tracks, iou_thresh,
        )
        for di, ti in matches_1:
            self._apply_detection(active_tracks[ti], high_dets[di][1], now)

        remaining_active = [active_tracks[i] for i in unmatch_trk_1]

        # === SECOND ASSOCIATION: low-conf dets vs remaining active tracks ===
        matches_2, _, unmatch_trk_2 = self._associate(
            low_dets, remaining_active, iou_thresh,
        )
        for di, ti in matches_2:
            self._apply_detection(remaining_active[ti], low_dets[di][1], now)

        # Mark still-unmatched active tracks as lost
        for i in unmatch_trk_2:
            remaining_active[i]["state"] = "lost"
            remaining_active[i]["frames_lost"] += 1

        # === THIRD ASSOCIATION: remaining high-conf dets vs lost tracks ===
        remaining_high = [high_dets[i] for i in unmatch_det_1]
        matches_3, unmatch_new, unmatch_lost = self._associate(
            remaining_high, lost_tracks, iou_thresh,
        )
        for di, ti in matches_3:
            track = lost_tracks[ti]
            self._apply_detection(track, remaining_high[di][1], now)
            track["state"] = "active"
            track["frames_lost"] = 0

        # === NEW TRACKS from unmatched high-conf dets ===
        for i in unmatch_new:
            _, det_data = remaining_high[i]
            new_track = {
                "track_id": self._next_id,
                "state": "active",
                "frames_lost": 0,
            }
            self._next_id += 1
            self._apply_detection(new_track, det_data, now)
            self._byte_tracks.append(new_track)

        # === Age unmatched lost tracks, prune expired ===
        for i in unmatch_lost:
            lost_tracks[i]["frames_lost"] += 1

        self._byte_tracks = [
            t for t in self._byte_tracks
            if not (t["state"] == "lost" and t["frames_lost"] > max_lost)
        ]

        # === Build _tracked_humans from active tracks ===
        self._tracked_humans.clear()
        for track in self._byte_tracks:
            if track["state"] == "active":
                tid = track["track_id"]
                human = TrackedHuman(track_id=tid)
                human.bbox = track.get("bbox")
                human.position_image = track.get("center_px")
                human.keypoints = track.get("keypoints")
                human.keypoints_conf = track.get("keypoints_conf")
                human.confidence = track.get("confidence", 0.0)
                human.distance_lidar = track.get("distance_lidar")
                human.distance_mono = track.get("distance_mono")
                human.distance = track.get("distance")
                human.position_rf = track.get("position_rf")
                human.last_seen = track.get("last_seen", now)
                self._tracked_humans[tid] = human

        logger.debug(
            "ByteTrack: %d active, %d lost, %d total tracks",
            sum(1 for t in self._byte_tracks if t["state"] == "active"),
            sum(1 for t in self._byte_tracks if t["state"] == "lost"),
            len(self._byte_tracks),
        )

    # ---- ByteTrack helpers ------------------------------------------- #

    def _associate(self, dets, tracks, iou_thresh):
        # type: (list, list, float) -> Tuple[list, list, list]
        """
        Greedy IoU-based association between detections and tracks.

        Args:
            dets:       list of (original_index, det_dict) tuples
            tracks:     list of track dicts (must have "bbox" key)
            iou_thresh: minimum IoU to accept a match

        Returns:
            matches:          list of (det_list_idx, track_list_idx) pairs
            unmatched_dets:   list of det list indices not matched
            unmatched_tracks: list of track list indices not matched
        """
        if not dets or not tracks:
            return [], list(range(len(dets))), list(range(len(tracks)))

        det_boxes = np.array([d.get("bbox") for _, d in dets], dtype=np.float64)
        trk_boxes = np.array([t.get("bbox") for t in tracks], dtype=np.float64)

        if det_boxes.ndim != 2 or trk_boxes.ndim != 2:
            return [], list(range(len(dets))), list(range(len(tracks)))

        iou_matrix = self._compute_iou_matrix(det_boxes, trk_boxes)

        # Greedy: pick highest IoU first, mark both sides as used
        rows, cols = np.where(iou_matrix >= iou_thresh)
        if len(rows) == 0:
            return [], list(range(len(dets))), list(range(len(tracks)))

        ious = iou_matrix[rows, cols]
        order = np.argsort(-ious)

        matches = []
        used_dets = set()
        used_tracks = set()
        for idx in order:
            d, t = int(rows[idx]), int(cols[idx])
            if d not in used_dets and t not in used_tracks:
                matches.append((d, t))
                used_dets.add(d)
                used_tracks.add(t)

        unmatched_dets = [i for i in range(len(dets)) if i not in used_dets]
        unmatched_tracks = [i for i in range(len(tracks)) if i not in used_tracks]
        return matches, unmatched_dets, unmatched_tracks

    @staticmethod
    def _compute_iou_matrix(boxes_a, boxes_b):
        # type: (np.ndarray, np.ndarray) -> np.ndarray
        """Pairwise IoU between (N,4) and (M,4) boxes in [x1,y1,x2,y2] format."""
        x1 = np.maximum(boxes_a[:, 0:1], boxes_b[:, 0:1].T)
        y1 = np.maximum(boxes_a[:, 1:2], boxes_b[:, 1:2].T)
        x2 = np.minimum(boxes_a[:, 2:3], boxes_b[:, 2:3].T)
        y2 = np.minimum(boxes_a[:, 3:4], boxes_b[:, 3:4].T)

        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
        area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
        union = area_a[:, None] + area_b[None, :] - inter
        return inter / np.maximum(union, 1e-6)

    @staticmethod
    def _apply_detection(track, det, timestamp):
        """Update a track dict with data from a new detection."""
        track["bbox"] = det.get("bbox")
        track["center_px"] = det.get("center_px")
        track["keypoints"] = det.get("keypoints")
        track["keypoints_conf"] = det.get("keypoints_conf")
        track["confidence"] = det.get("confidence", 0.0)
        track["distance_lidar"] = det.get("distance_lidar")
        track["distance_mono"] = det.get("distance_mono")
        track["distance"] = det.get("distance")
        track["position_rf"] = det.get("position_rf")
        track["last_seen"] = timestamp
        track["state"] = "active"
        track["frames_lost"] = 0

    # ================================================================== #
    #  STAGE 4 -- Trajectory prediction                                   #
    # ================================================================== #

    def _predict_trajectories(self):
        # type: () -> None
        """
        Predict future positions for each tracked human using
        HumanTrajectoryPredictor (linear extrapolation in robot frame).

        Flow:
            1. For each tracked human with a valid position_rf,
               feed [x_lateral, depth] into the predictor.
            2. Run predict_all() to get extrapolated future paths.
            3. Store predicted paths back into TrackedHuman.predicted_path.
            4. Derive velocity estimate from last two history points.
            5. Prune stale agents from predictor history.

        Track IDs are persistent across frames (provided by ByteTrack),
        enabling meaningful multi-frame history and velocity estimation.
        """
        # Feed current observations into predictor
        for human in self._tracked_humans.values():
            if human.position_rf is not None:
                self._predictor.update_agent_position(
                    human.track_id,
                    human.position_rf,       # [x_lateral, depth] in meters
                    self._frame_count,
                )

        # Run prediction (throttled by pred_interval)
        predictions = self._predictor.predict_all(self._frame_count)

        # Store predictions back into tracked humans + estimate velocity
        for human in self._tracked_humans.values():
            pred = predictions.get(human.track_id)
            if pred:
                human.predicted_path = pred
                # Estimate velocity from last two history points
                traj = self._predictor.agent_trajectories.get(human.track_id)
                if traj and len(traj) >= 2:
                    p0 = traj[-2]['position']
                    p1 = traj[-1]['position']
                    dt = traj[-1]['timestep'] - traj[-2]['timestep']
                    if dt > 0:
                        human.velocity = [
                            (p1[0] - p0[0]) / dt,
                            (p1[1] - p0[1]) / dt,
                        ]
            else:
                human.predicted_path = None
                human.velocity = None

        # Remove history for agents that disappeared
        active_ids = set(self._tracked_humans.keys())
        self._predictor.prune_stale(active_ids)

        # Reset predictor entirely if no humans tracked
        if not self._tracked_humans:
            self._predictor.reset()

    # ================================================================== #
    #  STAGE 5 -- Social cost computation (placeholder)                   #
    # ================================================================== #

    def _compute_social_costs(self):
        # type: () -> Tuple[float, float]
        """
        Compute aggregate proximity and trajectory costs.

        Returns:
            (total_prox_cost, total_traj_cost) -- both 0.0 for now.

        TODO:
            - Gaussian proximity: C(d) = h * exp(-d^2 / 2*sigma^2)
            - Relative velocity scaling
            - Trajectory obstruction penalty
        """
        total_prox = 0.0
        total_traj = 0.0

        for human in self._tracked_humans.values():
            if human.distance is not None:
                rf_str = "[{:.2f}, {:.2f}]".format(*human.position_rf) if human.position_rf else "n/a"
                pred_len = len(human.predicted_path) if human.predicted_path else 0
                logger.debug(
                    "  human %d  dist=%.2fm  rf=%s  pred_len=%d",
                    human.track_id, human.distance, rf_str, pred_len,
                )

        return total_prox, total_traj

    # ================================================================== #
    #  STAGE 6 -- Yielding evaluation (placeholder)                       #
    # ================================================================== #

    def _evaluate_yielding(self, mission_state):
        # type: (str) -> None
        """
        Determine whether the robot should enter/exit the yielding state.

        TODO:
            - Check d_min < d_yield
            - Check approach angle (cos_theta > 0.5)
            - Check predicted path intersection
            - Manage _clear_frames counter for exit
            - Timeout after 15s -> exit yielding
        """
        self.is_yielding = False  # always off for now

    # ================================================================== #
    #  STAGE 7 -- Velocity modulation (DISABLED -- passthrough)           #
    # ================================================================== #

    def _modulate_velocity(self, motion_vector, prox_cost, traj_cost):
        """
        Apply social cost to scale / steer the motion vector.

        Currently: returns motion_vector unchanged.

        TODO:
            - speed_scale = clamp(1.0 - prox - traj, v_min_scale, 1.0)
            - v_x' = v_x * speed_scale
            - v_y' = v_y * speed_scale
            - omega_z' = omega_z + avoidance steering
            - If yielding: return [0, 0, 0]
        """
        return motion_vector

    # ================================================================== #
    #  STAGE 8 -- Diagnostics                                             #
    # ================================================================== #

    def _update_diagnostics(self, prox_cost, traj_cost):
        distances = [
            h.distance for h in self._tracked_humans.values()
            if h.distance is not None
        ]
        self.diag = {
            "num_humans": len(self._tracked_humans),
            "min_distance": min(distances) if distances else None,
            "total_prox_cost": prox_cost,
            "total_traj_cost": traj_cost,
            "speed_scale": 1.0,
            "is_yielding": self.is_yielding,
        }
        if self._tracked_humans:
            logger.info(
                "humans=%d  min_d=%s  prox=%.3f  traj=%.3f  yield=%s",
                self.diag["num_humans"],
                "{:.2f}m".format(self.diag["min_distance"]) if self.diag["min_distance"] else "n/a",
                prox_cost, traj_cost, self.is_yielding,
            )

    # ================================================================== #
    #  Utilities                                                          #
    # ================================================================== #

    # Draw birds-eye view minimap
    def draw_bev(self, image):
        """Draw bird's-eye-view mini-map on bottom-right of image."""
        import cv2 as _cv2

        if not self._tracked_humans:
            return image

        sz = 200
        margin = 10
        pad = 20
        d_max = self.params["d_max"]
        scale = (sz - 2 * pad) / d_max   # px per meter

        h, w = image.shape[:2]
        if w < sz + margin or h < sz + margin:
            return image

        x0 = w - sz - margin
        y0 = h - sz - margin

        # Darken background
        image[y0:y0+sz, x0:x0+sz] = (
            image[y0:y0+sz, x0:x0+sz].astype(np.float32) * 0.3
        ).astype(np.uint8)
        _cv2.rectangle(image, (x0, y0), (x0 + sz, y0 + sz), (255, 255, 255), 1)

        # Robot marker at bottom-center
        rcx = x0 + sz // 2
        rcy = y0 + sz - pad
        _cv2.drawMarker(image, (rcx, rcy), (0, 255, 0),
                        _cv2.MARKER_TRIANGLE_UP, 10, 2)

        # Label
        _cv2.putText(image, "Bird's Eye View", (x0 + 5, y0 + 15),
                     _cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Robot motion-vector curves
        horizon = self.params["horizon_s"]
        dt = 0.05
        steps = int(horizon / dt)
        for vec, color in [
            (self._motion_original,  (255, 255, 0)),   # cyan  (BGR) = original
            (self._motion_modulated, (0, 255, 255)),    # yellow (BGR) = modulated
        ]:
            if vec is None:
                continue
            v_forward, v_lateral, omega = vec[0], vec[1], vec[2]
            x, y, theta = 0.0, 0.0, 0.0
            prev = (rcx, rcy)
            for _ in range(steps):
                x += (-v_forward * math.sin(theta) - v_lateral * math.cos(theta)) * dt
                y += (v_forward * math.cos(theta) + v_lateral * math.sin(theta)) * dt
                theta += omega * dt
                px = int(rcx + x * scale)
                py = int(rcy - y * scale)
                if not (x0 <= px <= x0 + sz and y0 <= py <= y0 + sz):
                    break
                _cv2.line(image, prev, (px, py), color, 1, _cv2.LINE_AA)
                prev = (px, py)

        for human in self._tracked_humans.values():
            if human.position_rf is None:
                continue

            xl, dp = human.position_rf
            px = int(rcx + xl * scale)
            py = int(rcy - dp * scale)

            if not (x0 <= px <= x0 + sz and y0 <= py <= y0 + sz):
                continue

            # Human dot + ID label
            _cv2.circle(image, (px, py), 4, (0, 0, 255), -1)
            _cv2.putText(image, str(human.track_id), (px + 6, py - 2),
                         _cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

            # Predicted trajectory dots
            if human.predicted_path:
                for pt in human.predicted_path:
                    tx = int(rcx + pt[0] * scale)
                    ty = int(rcy - pt[1] * scale)
                    if x0 <= tx <= x0 + sz and y0 <= ty <= y0 + sz:
                        _cv2.circle(image, (tx, ty), 2, (0, 165, 255), -1)

        return image
