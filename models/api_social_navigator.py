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
        "d_max": 2.5,
        # --- Action shield ---
        "shield_thresh": 0.5,       # safety score below this → shield activates
        "shield_active_states": ["running"],  # mission states where shield is armed
        "k_repulse": 0.4,           # repulsive velocity gain (m/s per unit cost)
        "k_brake": 0.6,             # forward speed reduction gain
        "v_min_scale": 0.1,         # minimum forward speed scale when braking
        "horizon_s": 2.0,
        "horizon_steps": 10,
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
        # --- LiDAR depth estimation ---
        "use_lidar_depth": False,      # True = use LiDAR for depth, False = monocular only
        "lidar_z_min": 0.0,          # meters, min Z relative to sensor (below sensor)
        "lidar_z_max": 1.0,           # meters, max Z relative to sensor (above sensor)
        "lidar_angle_margin_deg": 2.0, # degrees, angular padding on bbox edges
        "lidar_min_points": 3,         # minimum LiDAR points for valid estimate
        # --- BEV minimap display ---
        "bev_range_m": 5.0,            # visible range in BEV (meters), independent of d_max
        "bev_z_min": 0.0,             # BEV display Z filter min (sensor-relative)
        "bev_z_max": 0.1,             # BEV display Z filter max (tight band = clean slice)
    }

    def __init__(self, enabled=False, **kwargs):
        self.enabled = enabled
        self.params = {**self.DEFAULT_PARAMS, **kwargs}

        # --- Camera parameters ---
        half_fov = math.radians(self.params["fov_deg"] / 2.0)
        self._fx = (self.params["image_width"] / 2.0) / math.tan(half_fov)
        self._cx = self.params["image_width"] / 2.0

        # --- Tracked human data for ByteTrack ---
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

        # --- Action shield info ---
        self.shield_active = False
        self.safety_score = 1.0      # 1.0 = fully safe, 0.0 = imminent collision

        # --- Motion vectors (for BEV drawing) ---
        self._motion_original = None
        self._motion_modulated = None
        self._lidar_ranges = None

        # --- Diagnostics ---
        self.diag = {
            "num_humans": 0,
            "min_distance": None,
            "safety_score": 1.0,
            "shield_active": False,
            "speed_scale": 1.0,
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
        self._lidar_ranges = lidar_ranges

        # 1. Parse detections from pose_state
        detections = self._parse_pose_state(pose_state)

        # 2. Estimate distances + compute robot-frame positions
        self._estimate_distances(detections, lidar_ranges)

        # 3. Update tracker (simple ID assignment for now)
        self._update_tracker(detections)

        # 4. Predict future trajectories
        self._predict_trajectories()

        # 5. Compute safety score
        self.safety_score = self._compute_safety_score()

        # 6. Shield gate -- decide whether to intervene
        self.shield_active = self._evaluate_shield(mission_state)

        # 7. Command correction (only when shield is active)
        modified_vector = self._correct_command(motion_vector)

        # Store both for BEV visualisation
        self._motion_original = list(motion_vector)
        self._motion_modulated = list(modified_vector)

        # 8. Update diagnostics
        self._update_diagnostics()

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
        """Estimate distance to a detected person using LiDAR point cloud.

        Projects the person's bounding box into angular space, finds LiDAR
        points within that cone, and returns the median horizontal distance.
        """
        if not self.params["use_lidar_depth"]:
            return None
        if lidar_ranges is None:
            return None

        bbox = det.get("bbox")
        if bbox is None:
            return None

        x1_px, _, x2_px, _ = bbox

        # Convert bbox left/right pixel edges to horizontal angles (radians)
        theta_left = math.atan((x1_px - self._cx) / self._fx)
        theta_right = math.atan((x2_px - self._cx) / self._fx)
        margin = math.radians(self.params["lidar_angle_margin_deg"])
        theta_left -= margin
        theta_right += margin

        # Extract LiDAR point arrays
        lx = lidar_ranges.get("x")
        ly = lidar_ranges.get("y")
        lz = lidar_ranges.get("z")
        if lx is None or ly is None or lz is None:
            return None

        lx = np.asarray(lx, dtype=np.float64)
        ly = np.asarray(ly, dtype=np.float64)
        lz = np.asarray(lz, dtype=np.float64)

        if lx.size == 0:
            return None

        # Filter: only points in front of the robot (x > 0)
        mask = lx > 0

        # Filter: human-height range (z relative to sensor)
        mask &= (lz >= self.params["lidar_z_min"]) & (lz <= self.params["lidar_z_max"])

        # Compute camera-convention angle for each point
        # Robot frame: x=forward, y=left; camera: positive angle = right
        theta = np.arctan2(-ly, lx)
        mask &= (theta >= theta_left) & (theta <= theta_right)

        if np.count_nonzero(mask) < self.params["lidar_min_points"]:
            return None

        # Horizontal distance (ignore z)
        dist = np.sqrt(lx[mask] ** 2 + ly[mask] ** 2)
        return float(np.median(dist))

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
    #  STAGE 5 -- Safety score                                             #
    #                                                                      #
    #  Produces a scalar in [0, 1].                                        #
    #    1.0 = no humans nearby / fully safe                               #
    #    0.0 = imminent collision                                          #
    #                                                                      #
    #  Two components, combined via min():                                 #
    #    s_prox  – Gaussian proximity to closest human                     #
    #    s_traj  – minimum predicted future distance over horizon          #
    # ================================================================== #

    def _compute_safety_score(self):
        # type: () -> float
        """
        Compute a single safety score in [0, 1].

        Components
        ----------
        s_prox : 1 - h * exp(-d_min^2 / 2*sigma^2)
            Penalises current proximity to nearest human.

        s_traj : min predicted distance / d_safe
            Penalises predicted future closeness.  Clamped to [0, 1].

        Returns min(s_prox, s_traj) so the most dangerous signal wins.
        """
        sigma = self.params["sigma"]
        h = self.params["h"]
        d_safe = self.params["d_safe"]

        if not self._tracked_humans:
            return 1.0

        # ---- s_prox: current proximity ----
        distances = [
            human.distance for human in self._tracked_humans.values()
            if human.distance is not None
        ]
        if distances:
            d_min = min(distances)
            s_prox = 1.0 - h * math.exp(-(d_min ** 2) / (2.0 * sigma ** 2))
        else:
            s_prox = 1.0

        # ---- s_traj: predicted future proximity ----
        # TODO: for each human, scan predicted_path to find the minimum
        #       distance to the robot's own extrapolated path (or origin
        #       as a first approximation).  Clamp to [0, 1].
        s_traj = 1.0  # placeholder -- implement trajectory threat here

        # ---- human awareness level ----
        # see if detected human is facing quadruped
        # less safe if human does not see the quadruped

        score = min(s_prox, s_traj)
        score = max(0.0, min(1.0, score))

        logger.debug("safety_score=%.3f  s_prox=%.3f  s_traj=%.3f", score, s_prox, s_traj)
        return score

    # ================================================================== #
    #  STAGE 6 -- Shield gate                                              #
    #                                                                      #
    #  The shield activates when BOTH conditions hold:                     #
    #    1. safety_score < shield_thresh                                   #
    #    2. mission_state is in shield_active_states (e.g. "running")      #
    #                                                                      #
    #  When the shield is inactive the command passes through untouched.   #
    # ================================================================== #

    def _evaluate_shield(self, mission_state):
        # type: (str) -> bool
        """
        Returns true if the shield should be activated
        """
        allowed = self.params["shield_active_states"]
        if mission_state not in allowed:
            return False

        return self.safety_score < self.params["shield_thresh"]

    # ================================================================== #
    #  STAGE 7 -- Command correction (action shield)                       #
    #                                                                      #
    #  When shield_active:                                                 #
    #    1. Brake: scale forward speed by (1 - k_brake * threat)           #
    #    2. Repulse: add lateral velocity away from nearest human          #
    #  When shield_inactive: passthrough.                                  #
    # ================================================================== #

    def _correct_command(self, motion_vector):
        # type: (list) -> list
        """
        Apply action-shield correction to the nominal L2MM command.

        The original command is preserved as closely as possible;
        corrections are the minimum needed to maintain safety.

        Returns:
            [v_x, v_y, omega_z]  -- corrected command.
        """
        if not self.shield_active:
            return motion_vector

        vx, vy, omega = motion_vector[0], motion_vector[1], motion_vector[2]

        threat = 1.0 - self.safety_score          # 0 = safe, 1 = dangerous
        k_brake = self.params["k_brake"]
        k_repulse = self.params["k_repulse"]
        v_min = self.params["v_min_scale"]

        # ---- PLACEHOLDER LOGIC ----
        vy_correction = 0
        vx_correction = 0
        vy_corrected = vy + vy_correction
        vx_corrected = vx + vx_correction

        logger.info(
            "SHIELD  threat=%.2f  brake=%.2f  vx_corr=%.3f  vy %.3f->%.3f",
            threat, vx_correction, vy, vy_corrected,
        )

        return [vx_corrected, vy_corrected, omega]

    # ================================================================== #
    #  STAGE 8 -- Diagnostics                                             #
    # ================================================================== #

    def _update_diagnostics(self):
        distances = [
            h.distance for h in self._tracked_humans.values()
            if h.distance is not None
        ]
        self.diag = {
            "num_humans": len(self._tracked_humans),
            "min_distance": min(distances) if distances else None,
            "safety_score": self.safety_score,
            "shield_active": self.shield_active,
        }
        if self._tracked_humans:
            logger.info(
                "humans=%d  min_d=%s  safety=%.3f  shield=%s",
                self.diag["num_humans"],
                "{:.2f}m".format(self.diag["min_distance"]) if self.diag["min_distance"] else "n/a",
                self.safety_score, self.shield_active,
            )

    # ================================================================== #
    #  Utilities                                                          #
    # ================================================================== #

    # Draw birds-eye view minimap
    def draw_bev(self, image):
        """Draw bird's-eye-view mini-map on bottom-right of image."""
        import cv2 as _cv2

        has_lidar = self._lidar_ranges is not None
        has_humans = bool(self._tracked_humans)
        if not has_lidar and not has_humans:
            return image

        sz = 600
        margin = 10
        pad = 60
        bev_range = self.params["bev_range_m"]
        scale = (sz - 2 * pad) / bev_range   # px per meter

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

        # Robot at bottom-center
        rcx = x0 + sz // 2
        rcy = y0 + sz - pad

        # Range-ring semicircles (forward half)
        for r_m in np.arange(1.0, bev_range + 0.01, 1.0):
            r_px = int(r_m * scale)
            _cv2.ellipse(image, (rcx, rcy), (r_px, r_px), 0, 180, 360,
                         (80, 80, 80), 1, _cv2.LINE_AA)
            _cv2.putText(image, "{}m".format(int(r_m)),
                         (rcx + 3, rcy - r_px + 5),
                         _cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

        # --- LiDAR point cloud ---
        if has_lidar:
            lx = self._lidar_ranges.get("x")
            ly = self._lidar_ranges.get("y")
            lz = self._lidar_ranges.get("z")
            if lx is not None and ly is not None and lz is not None:
                lx = np.asarray(lx, dtype=np.float64)
                ly = np.asarray(ly, dtype=np.float64)
                lz = np.asarray(lz, dtype=np.float64)
                if lx.size > 0:
                    # Filter: finite values only
                    mask = np.isfinite(lx) & np.isfinite(ly) & np.isfinite(lz)
                    # Filter: remove origin noise
                    dist_sq = lx ** 2 + ly ** 2 + lz ** 2
                    mask &= dist_sq > 0.01 ** 2
                    # Filter: forward, within BEV range, BEV-specific Z band
                    mask &= lx > 0
                    mask &= (lx ** 2 + ly ** 2) <= bev_range ** 2
                    mask &= ((lz >= self.params["bev_z_min"])
                             & (lz <= self.params["bev_z_max"]))

                    fx = lx[mask]
                    fy = ly[mask]
                    fz = lz[mask]

                    if len(fx) > 0:
                        # Robot frame → BEV pixel (x=forward, y=left)
                        px_arr = (rcx + (-fy) * scale).astype(np.int32)
                        py_arr = (rcy - fx * scale).astype(np.int32)

                        # Clip to minimap bounds
                        in_bounds = ((px_arr >= x0) & (px_arr < x0 + sz)
                                     & (py_arr >= y0) & (py_arr < y0 + sz))
                        px_arr = px_arr[in_bounds]
                        py_arr = py_arr[in_bounds]
                        fz = fz[in_bounds]

                        # Z-height colormap (JET)
                        z_min, z_max = fz.min(), fz.max()
                        z_span = z_max - z_min if (z_max - z_min) > 1e-3 else 1.0
                        z_norm = ((fz - z_min) / z_span * 255).astype(np.uint8)
                        colors = _cv2.applyColorMap(
                            z_norm.reshape(-1, 1), _cv2.COLORMAP_JET
                        ).reshape(-1, 3)

                        # Direct 1px pixel writes (fast + clean)
                        image[py_arr, px_arr] = colors

        # Robot marker (on top of lidar points)
        _cv2.drawMarker(image, (rcx, rcy), (0, 255, 0),
                        _cv2.MARKER_TRIANGLE_UP, 24, 2)

        # Label
        _cv2.putText(image, "Bird's Eye View", (x0 + 10, y0 + 25),
                     _cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

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
                _cv2.line(image, prev, (px, py), color, 2, _cv2.LINE_AA)
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
            _cv2.circle(image, (px, py), 8, (0, 0, 255), -1)
            _cv2.putText(image, str(human.track_id), (px + 10, py - 4),
                         _cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Predicted trajectory dots
            if human.predicted_path:
                for pt in human.predicted_path:
                    tx = int(rcx + pt[0] * scale)
                    ty = int(rcy - pt[1] * scale)
                    if x0 <= tx <= x0 + sz and y0 <= ty <= y0 + sz:
                        _cv2.circle(image, (tx, ty), 4, (0, 165, 255), -1)

        return image
