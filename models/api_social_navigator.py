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
from models.safety import robot_safety_score, compute_safety_grid, safety_score_at_point

logger = logging.getLogger("SocialNavigator")
logger.setLevel(logging.WARNING)
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
        self.is_ghost = False            # True = predicted from trajectory, not directly observed

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
        # --- Action shield  params ---
        "shield_thresh_on": 0.7,    # safety score below this → shield activates
        "shield_thresh_off": 0.8,   # safety score above this → shield deactivates (hysteresis)
        "shield_active_states": ["running"],  # mission states where shield is armed
        "horizon_s": 5.0,
        "horizon_steps": 25,
        "mono_k": 300.0,
        "correction_gain": 25.0,
        # --- Camera params  ---
        "image_width": 640,
        "image_height": 480,
        "fov_deg": 80.0,
        "fov_v_deg": 45.0,            # vertical FOV (set independently if lens stretch differs)
        # --- Trajectory prediction ---
        "pred_history": 25,
        "pred_steps": 60,
        "pred_interval": 1,    # predict every frame
        # --- ByteTrack tracker ---
        "track_high_thresh": 0.5,   # confidence >= this → first association
        "track_low_thresh": 0.1,    # confidence >= this → second association
        "track_iou_thresh": 0.3,    # minimum IoU to accept a match
        "track_max_lost": 30,       # frames before a lost track is removed
        # --- LiDAR depth estimation ---
        "use_lidar_depth": True,       # True = use LiDAR for depth, False = monocular only
        "lidar_z_min": -0.3,          # meters, min Z in base frame (rejects ground ~-0.5)
        "lidar_z_max": 5.0,           # meters, max Z relative to sensor (above sensor)
        "lidar_angle_margin_deg": -5.0, # degrees, angular padding on bbox edges
        "lidar_min_points": 3,         # minimum LiDAR points for valid estimate
        "lidar_ema_alpha": 0.5,        # EMA smoothing factor (0..1); lower = smoother, higher = more responsive
        "lidar_depth_percentile": 50,  # percentile to find nearest returns (seed for cluster)
        "lidar_cluster_margin": 0.5,   # meters — only keep points within this of the nearest seed; rejects wall
        "lidar_kpt_conf_thresh": 0.5,  # min keypoint confidence to use for skeleton matching
        "lidar_skeleton_dist": 0.05,   # max normalized image distance from skeleton to count as "on person"
        # --- BEV minimap display ---
        "bev_range_m": 7.0,            # visible range in BEV (meters), independent of d_max
        # --- Ego-motion compensation ---
        "time_step": 0.25,            # seconds per control cycle (for ego-motion compensation)
        # --- LiDAR-camera overlay calibration ---
        "lidar_cam_yaw_offset": -0.0,   # degrees, horizontal rotation offset
        "lidar_cam_pitch_offset": 1.0, # degrees, vertical rotation offset
        "lidar_cam_z_offset": 0.05,    # meters, camera height above lidar (positive = camera higher)
        "lidar_cam_fov_scale": 1.0,    # multiplier on fov_deg for fine-tuning projection
        # --- Ghost humans (out-of-FOV persistence) ---
        "ghost_max_frames": 120,       # max frames a ghost persists (~30s at 4 Hz)
    }

    def __init__(self, enabled=False, **kwargs):
        self.enabled = enabled
        self.params = {**self.DEFAULT_PARAMS, **kwargs}

        # --- Camera parameters ---
        half_fov_h = math.radians(self.params["fov_deg"] / 2.0)
        half_fov_v = math.radians(self.params["fov_v_deg"] / 2.0)
        self._fx = (self.params["image_width"] / 2.0) / math.tan(half_fov_h)
        self._cx = self.params["image_width"] / 2.0
        self._fy = (self.params["image_height"] / 2.0) / math.tan(half_fov_v)
        self._cy = self.params["image_height"] / 2.0

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
        self.grid = None

        self.traj_score = 0.05
        self.best_score = 0.15

        # --- Shared lidar-to-image projection (built once per frame) ---
        self._lidar_image_points = None  # (N,5) array: [u_norm, v_norm, lx, ly, lz]
        self._lidar_human_masks = []     # list of boolean masks into _lidar_image_points

        # --- Motion vectors (for BEV drawing) ---
        self._motion_original = None
        self._motion_modulated = None
        self._lidar_ranges = None
        self._robot_predicted_path = None  # list of [x, y] in robot frame
        self._ego_velocity = None          # last executed [v_fwd, v_lat, omega]
        self._goal_rf = None               # [x_lateral, depth] estimated goal position
        self._best_traj = None             # best trajectory from _get_best_traj

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
        self._frame_count += 1
        self._lidar_ranges = lidar_ranges
        self._lidar_image_points = self._project_lidar_to_image(lidar_ranges)
        self._lidar_human_masks = []

        # --- Perception (always runs so BEV can show humans) ---

        # 1. Parse detections from pose_state
        detections = self._parse_pose_state(pose_state)

        # 2. Estimate distances + compute robot-frame positions
        self._estimate_distances(detections, lidar_ranges)

        # 3. Update tracker (simple ID assignment for now)
        self._update_tracker(detections)

        # 4. Predict future trajectories
        self._predict_trajectories()

        # --- Safety / correction (only when enabled) ---

        if not self.enabled:
            self._motion_original = list(motion_vector)
            self._motion_modulated = list(motion_vector)
            self._ego_velocity = list(motion_vector)
            self._update_diagnostics()
            return motion_vector

        # 5. Compute safety score
        self.safety_score = self._compute_safety_score()

        # 6. Shield gate -- decide whether to intervene
        self.shield_active = self._evaluate_shield(mission_state)

        # 7. Command correction (only when shield is active)
        modified_vector = self._correct_command(motion_vector)

        # Store both for BEV visualisation
        self._motion_original = list(motion_vector)
        self._motion_modulated = list(modified_vector)

        # Store executed velocity for ego-motion compensation next frame
        self._ego_velocity = list(modified_vector)

        # 8. Update diagnostics
        self._update_diagnostics()


        return modified_vector

    def step_ground_truth(self, motion_vector, gt_humans, mission_state):
        """
        Entry point for ground-truth human data (e.g. from CrowdNav simulator).

        Bypasses perception stages 1-3 (detection, distance estimation, tracking)
        and directly populates _tracked_humans from pre-computed robot-frame data.
        Then runs stages 4-8 as normal.

        Args:
            motion_vector: [v_x, v_y, omega_z] from L2MM
            gt_humans:     list of dicts, each with keys:
                             track_id:     int
                             position_rf:  [x_lateral, depth] in robot frame
                             distance:     float meters
                             velocity:     [vx_lateral, v_depth] in robot frame
                             radius:       float meters
            mission_state: str, current state machine state

        Returns:
            motion_vector: [v_x, v_y, omega_z] (possibly corrected)
        """
        if not self.enabled:
            return motion_vector

        self._frame_count += 1
        self._lidar_ranges = None
        # --- Directly populate _tracked_humans (bypass stages 1-3) ---
        self._tracked_humans.clear()
        for gh in gt_humans:
            tid = gh["track_id"]
            human = TrackedHuman(track_id=tid)
            human.position_rf = gh["position_rf"]
            human.distance = gh["distance"]
            human.velocity = gh.get("velocity")
            human.confidence = 1.0
            human.last_seen = time.time()
            self._tracked_humans[tid] = human

        # --- Cache robot predicted path for safety scoring + BEV ---
        self._robot_predicted_path = self._extrapolate_robot_path(motion_vector)

        # 4. Predict future trajectories
        self._predict_trajectories()

        # 5. Compute safety score
        self.safety_score = self._compute_safety_score()

        # 6. Shield gate
        self.shield_active = self._evaluate_shield(mission_state)

        # 7. Command correction
        modified_vector = self._correct_command(motion_vector)

        # Store both for BEV visualisation
        self._motion_original = list(motion_vector)
        self._motion_modulated = list(modified_vector)

        # Store executed velocity for ego-motion compensation next frame
        self._ego_velocity = list(modified_vector)

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
    #  LiDAR-to-image projection (shared by distance est. + overlay)     #
    # ================================================================== #

    def _project_lidar_to_image(self, lidar_ranges):
        """Project all front-facing LiDAR points to normalised image coords.

        Returns an (N, 5) array: [u_norm, v_norm, lx, ly, lz]
        where u_norm/v_norm are in [0, 1] (calibrated projection) and
        lx/ly/lz are the original pointcloud coordinates.
        Returns None if no valid points.
        """
        if lidar_ranges is None:
            return None

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

        # --- Filter: finite, non-origin, front-only, z-range ---
        mask = (np.isfinite(lx) & np.isfinite(ly) & np.isfinite(lz))
        mask &= (lx ** 2 + ly ** 2 + lz ** 2) > 0.01 ** 2
        mask &= lx > 0
        mask &= (lz >= self.params["lidar_z_min"]) & (lz <= self.params["lidar_z_max"])

        lx, ly, lz = lx[mask], ly[mask], lz[mask]
        if lx.size == 0:
            return None

        # --- Apply translation + rotation calibration offsets ---
        lz = lz - self.params["lidar_cam_z_offset"]  # shift to camera height

        yaw = math.radians(self.params["lidar_cam_yaw_offset"])
        pitch = math.radians(self.params["lidar_cam_pitch_offset"])

        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        rx = lx * cos_y - ly * sin_y
        ry = lx * sin_y + ly * cos_y

        cos_p, sin_p = math.cos(pitch), math.sin(pitch)
        rx2 = rx * cos_p + lz * sin_p
        rz = -rx * sin_p + lz * cos_p

        # Keep only points in front after rotation
        front = rx2 > 0
        rx2, ry, rz = rx2[front], ry[front], rz[front]
        lx, ly, lz = lx[front], ly[front], lz[front]
        if rx2.size == 0:
            return None

        # --- Pinhole projection to normalised coords ---
        fov_h = math.radians(self.params["fov_deg"] * self.params["lidar_cam_fov_scale"])
        fov_v = math.radians(self.params["fov_v_deg"] * self.params["lidar_cam_fov_scale"])
        half_tan_h = math.tan(fov_h / 2.0)
        half_tan_v = math.tan(fov_v / 2.0)

        u_norm = 0.5 + (-ry / rx2) / (2.0 * half_tan_h)
        v_norm = 0.5 + (-rz / rx2) / (2.0 * half_tan_v)

        # Keep only in-frame points
        in_frame = (u_norm >= 0) & (u_norm <= 1) & (v_norm >= 0) & (v_norm <= 1)
        u_norm, v_norm = u_norm[in_frame], v_norm[in_frame]
        lx, ly, lz = lx[in_frame], ly[in_frame], lz[in_frame]
        if lx.size == 0:
            return None

        return np.column_stack([u_norm, v_norm, lx, ly, lz])

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

    # COCO skeleton connections
    _SKELETON = [
        [0, 1], [0, 2], [1, 3], [2, 4],              # Head
        [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],     # Arms
        [5, 11], [6, 12], [11, 12],                   # Torso
        [11, 13], [13, 15], [12, 14], [14, 16],       # Legs
    ]

    @staticmethod
    def _point_to_segment_dist(px, py, ax, ay, bx, by):
        """Vectorised min distance from points (px,py) to segment (a→b)."""
        abx, aby = bx - ax, by - ay
        ab_sq = abx * abx + aby * aby
        if ab_sq < 1e-12:
            return np.hypot(px - ax, py - ay)
        t = np.clip(((px - ax) * abx + (py - ay) * aby) / ab_sq, 0.0, 1.0)
        return np.hypot(px - (ax + t * abx), py - (ay + t * aby))

    def _estimate_distance_lidar(self, det, lidar_ranges):
        # type: (dict, ...) -> Optional[float]
        """Estimate distance using the shared lidar-to-image projection table.

        Selects lidar points whose normalised image position is close to the
        person's skeleton lines (preferred) or inside the bounding box
        (fallback).  Saves a boolean mask into ``self._lidar_human_masks``
        so the overlay can colour those points black.
        """
        if not self.params["use_lidar_depth"]:
            return None
        if self._lidar_image_points is None or len(self._lidar_image_points) == 0:
            return None

        bbox = det.get("bbox")
        if bbox is None:
            return None

        pts = self._lidar_image_points  # (N, 5): u_n, v_n, lx, ly, lz
        u_n = pts[:, 0]
        v_n = pts[:, 1]
        lx  = pts[:, 2]

        img_w = float(self.params["image_width"])
        img_h = float(self.params["image_height"])

        # --- Try skeleton-based selection ---
        kpts = det.get("keypoints")
        kpts_conf = det.get("keypoints_conf")
        selected = None
        used_skeleton = False

        if kpts is not None and kpts_conf is not None:
            kpts = np.asarray(kpts, dtype=np.float64)
            kpts_conf = np.asarray(kpts_conf, dtype=np.float64)
            min_conf = self.params["lidar_kpt_conf_thresh"]

            # Normalise keypoints to [0, 1]
            kpts_n = kpts / np.array([img_w, img_h])

            segments = [(kpts_n[i], kpts_n[j]) for i, j in self._SKELETON
                        if kpts_conf[i] >= min_conf and kpts_conf[j] >= min_conf]

            if len(segments) >= 3:
                min_dists = np.full(len(u_n), np.inf)
                for seg_a, seg_b in segments:
                    d = self._point_to_segment_dist(
                        u_n, v_n, seg_a[0], seg_a[1], seg_b[0], seg_b[1])
                    np.minimum(min_dists, d, out=min_dists)

                thresh = self.params["lidar_skeleton_dist"]
                selected = min_dists <= thresh
                if np.count_nonzero(selected) >= self.params["lidar_min_points"]:
                    used_skeleton = True

        # --- Fallback: normalised bbox ---
        if not used_skeleton:
            x1_px, y1_px, x2_px, y2_px = bbox
            u_min, u_max = x1_px / img_w, x2_px / img_w
            v_min, v_max = y1_px / img_h, y2_px / img_h
            selected = ((u_n >= u_min) & (u_n <= u_max)
                        & (v_n >= v_min) & (v_n <= v_max))

        if np.count_nonzero(selected) < self.params["lidar_min_points"]:
            return None

        # --- Nearest-cluster ---
        depths = lx[selected]
        near_ref = float(np.percentile(depths, self.params["lidar_depth_percentile"]))
        cluster_margin = self.params["lidar_cluster_margin"]
        cluster_mask = depths <= near_ref + cluster_margin
        if np.count_nonzero(cluster_mask) < self.params["lidar_min_points"]:
            return near_ref

        # Build full mask (into _lidar_image_points) for overlay
        full_mask = np.zeros(len(pts), dtype=bool)
        sel_indices = np.where(selected)[0]
        full_mask[sel_indices[cluster_mask]] = True
        self._lidar_human_masks.append(full_mask)

        return float(np.median(depths[cluster_mask]))

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

    def _apply_detection(self, track, det, timestamp):
        """Update a track dict with data from a new detection."""
        track["bbox"] = det.get("bbox")
        track["center_px"] = det.get("center_px")
        track["keypoints"] = det.get("keypoints")
        track["keypoints_conf"] = det.get("keypoints_conf")
        track["confidence"] = det.get("confidence", 0.0)
        track["distance_lidar"] = det.get("distance_lidar")
        track["distance_mono"] = det.get("distance_mono")

        # EMA smoothing on lidar distance
        raw_dist = det.get("distance")
        prev_dist = track.get("distance")
        if raw_dist is not None and prev_dist is not None:
            alpha = self.params["lidar_ema_alpha"]
            track["distance"] = alpha * raw_dist + (1 - alpha) * prev_dist
        else:
            track["distance"] = raw_dist

        # Recompute position_rf from smoothed distance
        if track["distance"] is not None and det.get("center_px") is not None:
            u = det["center_px"][0]
            x_lateral = track["distance"] * (u - self._cx) / self._fx
            track["position_rf"] = [x_lateral, track["distance"]]
        else:
            track["position_rf"] = det.get("position_rf")

        track["last_seen"] = timestamp
        track["state"] = "active"
        track["frames_lost"] = 0

    # ================================================================== #
    #  STAGE 4 -- Trajectory prediction                                 #
    # ================================================================== #

    def _compensate_ego_motion(self):
        """Transform predictor history from previous robot frame to current.

        Between frames the robot executed self._ego_velocity for time_step
        seconds.  A world-fixed point at [x_lat, depth] in the OLD robot
        frame maps to the NEW robot frame as:

            1. Translate:  x' = x_lat + v_lat*dt,  d' = depth - v_fwd*dt
            2. Rotate by -omega*dt (robot turned left, world rotates right):
               x'' =  x'*cos + d'*sin
               d'' = -x'*sin + d'*cos
        """
        if self._ego_velocity is None:
            return
        v_fwd, v_lat, omega = self._ego_velocity
        dt = self.params["time_step"]
        dtheta = omega * dt
        cos_dt = math.cos(dtheta)
        sin_dt = math.sin(dtheta)

        for traj in self._predictor.agent_trajectories.values():
            for entry in traj:
                x_lat, depth = entry['position']
                # translate
                x_lat += v_lat * dt
                depth -= v_fwd * dt
                # rotate
                entry['position'] = [
                    x_lat * cos_dt + depth * sin_dt,
                   -x_lat * sin_dt + depth * cos_dt,
                ]

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
        # Transform stored history from previous robot frame to current
        self._compensate_ego_motion()

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

        # --- Ghost humans: persist out-of-FOV agents via trajectory prediction ---
        active_ids = set(self._tracked_humans.keys())
        ghost_max = self.params["ghost_max_frames"]

        for agent_id in list(self._predictor.agent_trajectories.keys()):
            if agent_id in active_ids:
                continue  # still directly observed, not a ghost

            traj = self._predictor.agent_trajectories[agent_id]
            if not traj:
                continue

            last_timestep = traj[-1]['timestep']
            frames_since_seen = self._frame_count - last_timestep

            # Expired ghost — prune entirely
            if frames_since_seen > ghost_max:
                del self._predictor.agent_trajectories[agent_id]
                self._predictor.predicted_trajectories.pop(agent_id, None)
                continue

            # Get predicted trajectory for this agent
            pred = predictions.get(agent_id)
            if not pred or len(pred) == 0:
                continue

            # Index into prediction to get ghost's "current" position
            # predictions[0] = 1 step after last observation, so index = frames_since_seen - 1
            pred_idx = frames_since_seen - 1
            if pred_idx >= len(pred):
                # Beyond prediction horizon — prune
                del self._predictor.agent_trajectories[agent_id]
                self._predictor.predicted_trajectories.pop(agent_id, None)
                continue

            ghost_pos = pred[pred_idx]  # [x_lateral, depth]

            # Skip if predicted behind robot (depth <= 0)
            if ghost_pos[1] <= 0:
                del self._predictor.agent_trajectories[agent_id]
                self._predictor.predicted_trajectories.pop(agent_id, None)
                continue

            # De-duplicate: if an observed human is near the ghost's predicted
            # position, they are the same person re-entering FOV with a new
            # tracker ID.  Discard the stale predictor entry.
            merged = False
            for obs in self._tracked_humans.values():
                if obs.position_rf is None:
                    continue
                dist = math.hypot(ghost_pos[0] - obs.position_rf[0],
                                  ghost_pos[1] - obs.position_rf[1])
                if dist < 1.0:  # meters
                    del self._predictor.agent_trajectories[agent_id]
                    self._predictor.predicted_trajectories.pop(agent_id, None)
                    merged = True
                    break
            if merged:
                continue

            # Create ghost TrackedHuman
            ghost = TrackedHuman(agent_id)
            ghost.is_ghost = True
            ghost.position_rf = ghost_pos
            ghost.distance = ghost_pos[1]  # depth as distance estimate
            ghost.predicted_path = pred[pred_idx:]  # remaining predicted path
            # Estimate velocity from predictor history
            if len(traj) >= 2:
                p0 = traj[-2]['position']
                p1 = traj[-1]['position']
                dt = traj[-1]['timestep'] - traj[-2]['timestep']
                if dt > 0:
                    ghost.velocity = [
                        (p1[0] - p0[0]) / dt,
                        (p1[1] - p0[1]) / dt,
                    ]
            self._tracked_humans[agent_id] = ghost

        # Prune predictor entries not in updated tracked humans set
        updated_ids = set(self._tracked_humans.keys())
        self._predictor.prune_stale(updated_ids)

        # Reset predictor entirely if no humans tracked
        if not self._tracked_humans:
            self._predictor.reset()

    # ================================================================== #
    #  STAGE 5 -- Safety score                                             #
    # ================================================================== #

    def _compute_safety_score(self):
        # type: () -> float
        """
        Compute safety score at robot's location using the same function
        that the heatmap uses. Robot is at origin in robot frame; humans
        are at their position_rf coordinates.
        """
        if not self._tracked_humans:
            return 1.0

        # Human positions in robot frame (robot is at origin)
        human_positions = [
            tuple(human.position_rf)
            for human in self._tracked_humans.values()
            if human.position_rf is not None
        ]

        human_predicted_paths = {
            human.track_id: human.predicted_path
            for human in self._tracked_humans.values()
            if human.predicted_path
        }

        score = robot_safety_score(0.0, 0.0, human_positions,
                                   human_predicted_paths=human_predicted_paths)

        logger.debug("safety_score=%.3f", score)
        return score

    # ================================================================== #
    #  STAGE 6 -- Shield gate                                              #
    # ================================================================== #

    def _evaluate_shield(self, mission_state):
        # type: (str) -> bool
        """
        Returns true if the shield should be activated
        """

        # logic to handle ensuring the shield only activates when mission state is running
        allowed = self.params["shield_active_states"]
        if mission_state not in allowed:
            return False
        # Hysteresis: lower threshold to activate, higher to deactivate
        if self.shield_active:
            return self.safety_score < self.params["shield_thresh_off"]
        else:
            return self.safety_score < self.params["shield_thresh_on"]

    # ================================================================== #
    #  STAGE 7 -- Command correction (action shield)                       #
    # ================================================================== #

    def _correct_command(self, motion_vector):
        # type: (list) -> list
        """
        when shield active:
            v* = argmin_{v'}(|v' - v|)
                    st. still in safe region + buffer/padding/safety_factor
        else:
            passthrough
        Returns:
            [v_x', v_y', omega_z']  -- corrected motion vector

        
        """
        if not self.shield_active:
            return motion_vector

        vx, vy, omega = motion_vector[0], motion_vector[1], motion_vector[2]

        threat = 1.0 - self.safety_score          # 0 = safe, 1 = dangerous

        # Ensure safety grid is computed for potential field correction
        bev_range = self.params["bev_range_m"]
        self.grid, _ = self.get_safety_heatmap(
            xlim=(-bev_range / 2, bev_range / 2),
            ylim=(0, bev_range),
            resolution=bev_range / 50,
        )

        # Scale correction proportionally to threat level so it ramps
        # up smoothly rather than snapping to full strength at the threshold.
        omega_correction = self._potential_field_correction() * threat
        vx_corrected = vx
        vy_corrected = vy
        omega_corrected = omega + omega_correction
        logger.info(
            "SHIELD  threat=%.2f  omega_corr=%.3f  omega %.3f->%.3f",
            threat, omega_correction, omega, omega_corrected,
        )
        return [vx_corrected, vy_corrected, omega_corrected]
    
    def _potential_field_correction(self):
        """
        Compute angular correction to steer toward safer areas.
        Grid: robot at bottom-middle, y-axis points forward, x-axis points right
        """
        robot_i = 0  # bottom row
        robot_j = self.grid.shape[1] // 2  # middle column
        
        # Compute gradient (points toward higher safety)
        grad_y, grad_x = np.gradient(self.grid)
        
        # Safety gradient at robot position
        safety_grad_x = grad_x[robot_i, robot_j]  # right is positive
        safety_grad_y = grad_y[robot_i, robot_j]  # forward is positive
        
        # Convert to angular correction
        # If danger on right (grad_x < 0), turn left (omega > 0)
        # If danger on left (grad_x > 0), turn right (omega < 0)
        omega_correction = -self.DEFAULT_PARAMS["correction_gain"] * safety_grad_x
        
        return omega_correction

    def _bezier_curve_correction(self):
        best_curve, best_score = self._get_best_traj()
        # then make it correct velocity to closely match the best curve
        correction = 0.0
        return correction

    def _get_best_traj(self, motion_vector):
        """
        Compute angular correction to steer away from a specific human using a Bezier curve approach.
        This is a placeholder for a more advanced correction method that considers the predicted path of the human.
        """
        if self._goal_rf is None:
            return [], 0.0

        t_start = time.time()

        traj_check_range = 3.0 # m
        tangent_range = 3.0 # m
        tangent_min = 2.0 # m
        step_size = 1.0 # m (coarse grid for now)

        steps = 20 # number of arc segments
        minimum_allowed_safety = 0.1 #
        traj_min_similarity = 1.0 #

        [x_lat, depth] = self._goal_rf
        heading = np.array([0.0, 1.0])

        # P0: robot at origin
        p0 = np.array([0.0, 0.0])
        # P3: goal in BEV coords [x_lateral, depth]
        p3 = np.array([self._goal_rf[0], self._goal_rf[1]])

        robot_path = self._extrapolate_robot_path_full(motion_vector, steps=steps)
        if not robot_path:
            logger.info("_get_best_traj: no robot_path, elapsed=%.3fs", time.time() - t_start)
            return [], 0.0

        x_lats = np.arange(-traj_check_range, traj_check_range, step_size)
        depths = np.arange(-traj_check_range, traj_check_range, step_size)
        tangent_lengths = np.arange(tangent_min, tangent_range+step_size, step_size)
        best_curve = []
        best_score = -1.0
        best_lowest_safety = 0.0
        n_evaluated = 0
        best_similarity = 0.0
        for x in x_lats:
            for d in depths:
                for l in tangent_lengths:
                    p1 = p0 + heading * l
                    p2 = np.array([x, d])
                    curve = self._bezier(p0, p1, p2, p3, steps=steps)
                    score, lowest_safety_val = self._trajectory_eval(curve)

                    traj_similarity = self._trajectory_similarity(curve, robot_path)
                    n_evaluated += 1
                    if score > best_score:
                        best_curve = curve
                        best_score = score
                        best_lowest_safety = lowest_safety_val
                        best_similarity = traj_similarity

        elapsed = time.time() - t_start
        logger.warning(
            "_get_best_traj: %d curves in %.3fs  best_score=%.3f  lowest_safety=%.3f best_similarity=%.3f",
            n_evaluated, elapsed, best_score, best_lowest_safety, best_similarity
        )
        return best_curve, best_score

    def _trajectory_eval(self, curve):
        trajectory_score = 0.0
        lowest_safety_val = 1.0

        human_positions = [
        tuple(human.position_rf)
        for human in self._tracked_humans.values()
        if human.position_rf is not None]

        human_predicted_paths = {
        h.track_id: h.predicted_path
        for h in self._tracked_humans.values()
        if h.predicted_path}

        
        total_length = 0.0
        for i in range(len(curve)-1):
            x, y = curve[i]
            x2, y2 = curve[i+1]
            segment_length = np.linalg.norm([x2 - x, y2 - y])
            safety_at_point = safety_score_at_point(x, y, human_positions, human_predicted_paths)
            trajectory_score += safety_at_point * segment_length
            total_length += segment_length
            lowest_safety_val = min(lowest_safety_val, safety_at_point)

        # Normalize by total arc length so score is average safety [0, 1]
        if total_length > 0:
            trajectory_score /= total_length

        return trajectory_score, lowest_safety_val

    def _trajectory_similarity(self, traj1, traj2):
        distances = 0.0
        assert len(traj1) == len(traj2), "Trajectories must have the same number of points for similarity evaluation."
        for p1, p2 in zip(traj1, traj2):
            dist = np.linalg.norm(np.array(p1) - np.array(p2))
            distances -= dist
        return 1 / distances if distances != 0 else float('inf')

    # ================================================================== #
    #  STAGE 8 -- Diagnostics                                             #
    # ================================================================== #

    def update_goal(self, object_xyn, bbox_height_px):
        """Estimate goal position in robot frame from camera detection."""
        if bbox_height_px is None or bbox_height_px < 10:
            return  # keep last valid goal
        depth = self.params["mono_k"] / bbox_height_px
        u_px = object_xyn[0] * self.params["image_width"]
        self._goal_rf = [depth * (u_px - self._cx) / self._fx, depth]

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
            "traj_score": 0.0,
            "best_traj_score": 0.0,
        }
        if self._tracked_humans:
            logger.info(
                "humans=%d  min_d=%s  safety=%.3f  shield=%s",
                self.diag["num_humans"],
                "{:.2f}m".format(self.diag["min_distance"]) if self.diag["min_distance"] else "n/a",
                self.safety_score, self.shield_active,
            )
        motion = self._motion_original or [0, 0, 0]
        try:
            traj = self._extrapolate_robot_path_full(motion)
            traj_score, lowest_safety = self._trajectory_eval(traj)
            self.diag["traj_score"] = traj_score

            if self._goal_rf is not None:
                best_traj, best_score = self._get_best_traj(motion)
                self.diag["best_traj_score"] = best_score
                self._best_traj = best_traj if best_traj else (traj if traj else None)
            else:
                print("no valid goal")
                self.diag["best_traj_score"] = traj_score
                self._best_traj = traj if traj else None
        except Exception as e:
            logger.error("_update_diagnostics traj eval FAILED: %s", e, exc_info=True)


    # ================================================================== #
    #  Utilities                                                          #
    # ================================================================== #

    # Render bird's-eye view as standalone image
    def render_bev(self, show_heatmap=False):
        """Render a standalone  bird's-eye-view image and return it."""
        import cv2 as _cv2

        sz = 400
        pad = 60
        bev_range = self.params["bev_range_m"]
        scale = (sz - 2 * pad) / bev_range

        bev = np.zeros((sz, sz, 3), dtype=np.uint8)

        has_lidar = self._lidar_ranges is not None
        has_humans = bool(self._tracked_humans)
        _lidar_z_min = _lidar_z_max = None

        # Safety heatmap underlay
        if show_heatmap:
            self.grid, extent = self.get_safety_heatmap(
                xlim=(-bev_range / 2, bev_range / 2),
                ylim=(0, bev_range),
                resolution=bev_range / 50,
            )
            if self.grid is not None:
                if not hasattr(self, '_bev_cmap'):
                    import matplotlib
                    matplotlib.use('Agg')
                    self._bev_cmap = matplotlib.cm.get_cmap('RdYlGn')
                rgba = self._bev_cmap(self.grid)[:, :, :3]
                hm_bgr = (rgba[:, :, ::-1] * 255).astype(np.uint8)
                inner = sz - 2 * pad
                hm_bgr = _cv2.resize(hm_bgr, (inner, inner), interpolation=_cv2.INTER_NEAREST)
                hm_bgr = _cv2.flip(hm_bgr, 0)
                bev[pad:pad+inner, pad:pad+inner] = hm_bgr

                # Shield-threshold contour
                thresh = self.params["shield_thresh_on"]
                binary = (self.grid < thresh).astype(np.uint8) * 255
                binary = _cv2.resize(binary, (inner, inner),
                                     interpolation=_cv2.INTER_NEAREST)
                binary = _cv2.flip(binary, 0)
                contours, _ = _cv2.findContours(
                    binary, _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    cnt += np.array([[[pad, pad]]])
                _cv2.drawContours(bev, contours, -1, (0, 0, 0), 1)

        _cv2.rectangle(bev, (0, 0), (sz - 1, sz - 1), (255, 255, 255), 1)

        # Robot at bottom-center
        rcx = sz // 2
        rcy = sz - pad

        # Camera FOV lines
        half_fov = np.radians(self.params["fov_deg"] / 2.0)
        fov_len = int(bev_range * scale)
        for sign in (-1, 1):
            ex = int(rcx + sign * fov_len * np.sin(half_fov))
            ey = int(rcy - fov_len * np.cos(half_fov))
            _cv2.line(bev, (rcx, rcy), (ex, ey), (100, 100, 100), 1, _cv2.LINE_AA)

        # Range-ring semicircles
        for r_m in np.arange(1.0, bev_range + 0.01, 1.0):
            r_px = int(r_m * scale)
            _cv2.ellipse(bev, (rcx, rcy), (r_px, r_px), 0, 180, 360,
                         (80, 80, 80), 1, _cv2.LINE_AA)
            _cv2.putText(bev, "{}m".format(int(r_m)),
                         (rcx + 3, rcy - r_px + 5),
                         _cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

        # LiDAR point cloud
        if has_lidar:
            lx = self._lidar_ranges.get("x")
            ly = self._lidar_ranges.get("y")
            lz = self._lidar_ranges.get("z")
            if lx is not None and ly is not None and lz is not None:
                lx = np.asarray(lx, dtype=np.float64)
                ly = np.asarray(ly, dtype=np.float64)
                lz = np.asarray(lz, dtype=np.float64)
                if lx.size > 0:
                    mask = np.isfinite(lx) & np.isfinite(ly) & np.isfinite(lz)
                    dist_sq = lx ** 2 + ly ** 2 + lz ** 2
                    mask &= dist_sq > 0.01 ** 2
                    mask &= lx > 0
                    mask &= (lx ** 2 + ly ** 2) <= bev_range ** 2
                    mask &= ((lz >= self.params["lidar_z_min"])
                             & (lz <= self.params["lidar_z_max"]))

                    fx, fy, fz = lx[mask], ly[mask], lz[mask]

                    if len(fx) > 0:
                        px_arr = (rcx + (-fy) * scale).astype(np.int32)
                        py_arr = (rcy - fx * scale).astype(np.int32)

                        in_bounds = ((px_arr >= 0) & (px_arr < sz)
                                     & (py_arr >= 0) & (py_arr < sz))
                        px_arr = px_arr[in_bounds]
                        py_arr = py_arr[in_bounds]
                        fz = fz[in_bounds]

                        z_min, z_max = fz.min(), fz.max()
                        z_span = z_max - z_min if (z_max - z_min) > 1e-3 else 1.0
                        z_norm = ((fz - z_min) / z_span * 255).astype(np.uint8)
                        colors = _cv2.applyColorMap(
                            z_norm.reshape(-1, 1), _cv2.COLORMAP_JET
                        ).reshape(-1, 3)
                        for _px, _py, _col in zip(px_arr, py_arr, colors):
                            _cv2.circle(bev, (int(_px), int(_py)), 1,
                                        tuple(int(c) for c in _col), -1)
                        _lidar_z_min, _lidar_z_max = z_min, z_max

        # LiDAR Z-height colorbar legend
        if has_lidar and _lidar_z_min is not None:
            cb_x = sz - 30          # right edge
            cb_y0, cb_y1 = pad, sz - pad
            cb_h = cb_y1 - cb_y0
            cb_w = 12
            gradient = np.linspace(255, 0, cb_h, dtype=np.uint8).reshape(-1, 1)
            cb_color = _cv2.applyColorMap(gradient, _cv2.COLORMAP_JET)
            bev[cb_y0:cb_y1, cb_x:cb_x + cb_w] = cb_color
            _cv2.rectangle(bev, (cb_x, cb_y0), (cb_x + cb_w, cb_y1), (200, 200, 200), 1)
            _cv2.putText(bev, f"{_lidar_z_max:.1f}m", (cb_x - 40, cb_y0 + 10),
                         _cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
            _cv2.putText(bev, f"{_lidar_z_min:.1f}m", (cb_x - 40, cb_y1),
                         _cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
            _cv2.putText(bev, "Z", (cb_x + 2, cb_y0 - 5),
                         _cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

        # Robot marker
        _cv2.drawMarker(bev, (rcx, rcy), (0, 255, 0),
                        _cv2.MARKER_TRIANGLE_UP, 24, 2)

        humans_with_pos = [h for h in self._tracked_humans.values()
                           if h.position_rf is not None]
        n_observed = len([h for h in humans_with_pos if not h.is_ghost])
        n_ghosts = len([h for h in humans_with_pos if h.is_ghost])
        count_str = f"{n_observed}"
        if n_ghosts > 0:
            count_str += f"+{n_ghosts}g"
        _cv2.putText(bev, f"Bird's Eye View  [{count_str} human{'s' if (n_observed + n_ghosts) != 1 else ''}]",
                     (10, 25), _cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # Robot motion-vector curves
        path_tips = {}  # key: "original" or "corrected" -> (px, py)
        for vec, color, key in [
            (self._motion_original,  (255, 255, 0),  "original"),
            (self._motion_modulated, (0, 255, 255),  "corrected"),
        ]:
            if vec is None:
                continue
            path = self._extrapolate_robot_path(vec)
            prev = (rcx, rcy)
            tip = prev
            for pt in path:
                px = int(rcx + pt[0] * scale)
                py = int(rcy - pt[1] * scale)
                if not (0 <= px < sz and 0 <= py < sz):
                    break
                _cv2.line(bev, prev, (px, py), color, 2, _cv2.LINE_AA)
                prev = (px, py)
                tip = prev
            path_tips[key] = tip

        # Full Robot trajectory curves
        traj_tips = {}  # key: "original" or "corrected" -> (px, py)
        for vec, color, key in [
            (self._motion_original,  (255, 255, 0),  "original")
            # (self._motion_modulated, (0, 255, 255),  "corrected"),
        ]:
            if vec is None:
                continue
            path = self._extrapolate_robot_path_full(vec)
            prev = (rcx, rcy)
            tip = prev
            for pt in path:
                px = int(rcx + pt[0] * scale)
                py = int(rcy - pt[1] * scale)
                if not (0 <= px < sz and 0 <= py < sz):
                    break
                _cv2.line(bev, prev, (px, py), color, 2, _cv2.LINE_AA)
                prev = (px, py)
                tip = prev
            traj_tips[key] = tip

        # Best trajectory curve (green)
        if self._best_traj:
            prev = (rcx, rcy)
            for pt in self._best_traj:
                px = int(rcx + pt[0] * scale)
                py = int(rcy - pt[1] * scale)
                if not (0 <= px < sz and 0 <= py < sz):
                    break
                _cv2.line(bev, prev, (px, py), (0, 255, 0), 2, _cv2.LINE_AA)
                prev = (px, py)

        # Correction arrow: original tip -> corrected tip (magenta)
        if self.shield_active and "original" in path_tips and "corrected" in path_tips:
            o_tip = path_tips["original"]
            c_tip = path_tips["corrected"]
            if o_tip != c_tip:
                _cv2.arrowedLine(bev, o_tip, c_tip,
                                 (255, 0, 255), 2, _cv2.LINE_AA, tipLength=0.3)

        # Legend
        lx, ly = 10, sz - 75
        for label, color in [("Original", (255, 255, 0)),
                              ("Corrected", (0, 255, 255)),
                              ("Best Traj", (0, 255, 0)),
                              ("Correction", (255, 0, 255))]:
            _cv2.line(bev, (lx, ly), (lx + 20, ly), color, 2, _cv2.LINE_AA)
            _cv2.putText(bev, label, (lx + 25, ly + 4),
                         _cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
            ly += 16

        # Tracked humans
        for human in self._tracked_humans.values():
            if human.position_rf is None:
                continue

            xl, dp = human.position_rf
            px = int(rcx + xl * scale)
            py = int(rcy - dp * scale)

            if not (0 <= px < sz and 0 <= py < sz):
                continue

            if human.is_ghost:
                # Hollow circle, lighter color for ghost humans
                _cv2.circle(bev, (px, py), 10, (100, 100, 255), 2)
                dist_str = f"{human.distance:.1f}m" if human.distance is not None else "?"
                _cv2.putText(bev, f"#{human.track_id} {dist_str} (ghost)",
                             (px + 13, py + 4),
                             _cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 255), 1)
            else:
                _cv2.circle(bev, (px, py), 10, (0, 0, 255), -1)
                _cv2.circle(bev, (px, py), 10, (255, 255, 255), 1)
                dist_str = f"{human.distance:.1f}m" if human.distance is not None else "?"
                _cv2.putText(bev, f"#{human.track_id} {dist_str}", (px + 13, py + 4),
                             _cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            if human.predicted_path:
                for pt in human.predicted_path:
                    tx = int(rcx + pt[0] * scale)
                    ty = int(rcy - pt[1] * scale)
                    if 0 <= tx < sz and 0 <= ty < sz:
                        _cv2.circle(bev, (tx, ty), 4, (0, 165, 255), -1)

        # Goal marker
        if self._goal_rf is not None:
            gpx = int(rcx + self._goal_rf[0] * scale)
            gpy = int(rcy - self._goal_rf[1] * scale)
            if 0 <= gpx < sz and 0 <= gpy < sz:
                _cv2.drawMarker(bev, (gpx, gpy), (0, 255, 0),
                                _cv2.MARKER_STAR, 16, 2)

        return bev

    #  LiDAR-camera overlay                                              #
    def overlay_lidar(self, image):
        """Draw LiDAR points on a BGR camera image using the shared projection.

        Reads ``self._lidar_image_points`` (built in step()) and colours
        points by forward distance.  Points used for human distance
        estimation (``self._lidar_human_masks``) are drawn black.
        """
        import cv2 as _cv2

        if self._lidar_image_points is None or len(self._lidar_image_points) == 0:
            return image

        pts = self._lidar_image_points  # (N, 5): u_n, v_n, lx, ly, lz
        h_img, w_img = image.shape[:2]

        u = (pts[:, 0] * w_img).astype(np.int32)
        v = (pts[:, 1] * h_img).astype(np.int32)
        lx = pts[:, 2]  # forward distance

        # --- Distance colourmap (close=red, far=blue) ---
        d_min, d_max = lx.min(), lx.max()
        d_span = d_max - d_min if (d_max - d_min) > 1e-3 else 1.0
        z_norm = ((lx - d_min) / d_span * 255).astype(np.uint8)
        colors = _cv2.applyColorMap(z_norm.reshape(-1, 1), _cv2.COLORMAP_JET).reshape(-1, 3)

        # --- Build human-point mask ---
        is_human = np.zeros(len(u), dtype=bool)
        for mask in self._lidar_human_masks:
            is_human |= mask

        # Draw non-human points as coloured circles
        for _u, _v, _col, _h in zip(u, v, colors, is_human):
            if not _h:
                _cv2.circle(image, (int(_u), int(_v)), 5,
                            tuple(int(c) for c in _col), -1)

        # Draw human points as small white stars
        for _u, _v, _h in zip(u, v, is_human):
            if _h:
                _cv2.drawMarker(image, (int(_u), int(_v)), (255, 255, 255),
                                _cv2.MARKER_STAR, 8, 1)

        return image

    #  Safety heatmap (shared by deploy + simulation) 
    def get_safety_heatmap(self, xlim=(-5, 5), ylim=(-5, 5), resolution=0.2):
        """
        Compute a 2D safety heatmap from the currently tracked humans.

        Works identically for both paths:
          - deploy:  _tracked_humans filled by step()  (perception)
          - sim:     _tracked_humans filled by step_ground_truth()

        All coordinates are in robot frame (robot at origin).

        Args:
            xlim: (xmin, xmax) lateral bounds in meters
            ylim: (ymin, ymax) depth bounds in meters
            resolution: grid cell size in meters

        Returns:
            safety_grid: 2D numpy array, values in [0, 1]
            extent:      [xmin, xmax, ymin, ymax] for imshow
            Returns (None, None) if no humans are tracked.
        """

        human_positions = [
            tuple(h.position_rf)
            for h in self._tracked_humans.values()
            if h.position_rf is not None
        ]
        if not human_positions:
            # No humans — return uniform safe grid
            x = np.arange(xlim[0], xlim[1], resolution)
            y = np.arange(ylim[0], ylim[1], resolution)
            return np.ones((len(y), len(x))), [xlim[0], xlim[1], ylim[0], ylim[1]]

        human_predicted_paths = {
            h.track_id: h.predicted_path
            for h in self._tracked_humans.values()
            if h.predicted_path
        }

        self.grid, extent = compute_safety_grid(
            human_positions, xlim, ylim,
            resolution=resolution,
            human_predicted_paths=human_predicted_paths or None,
        )
        return self.grid, extent

    # Extrapolate robot path
    def _extrapolate_robot_path(self, motion_vector):
        """
        Extrapolate the robot's future path from its current motion vector.

        Uses the same unicycle integration as draw_bev lines 885-906.
        Returns a list of [x, y] positions in robot frame (robot starts at origin).

        Args:
            motion_vector: [v_forward, v_lateral, omega_z]

        Returns:
            list of [x, y] points in robot frame
        """
        horizon = self.params["horizon_s"]
        steps = self.params["horizon_steps"]
        if steps <= 0:
            return []
        dt = horizon / steps

        v_forward, v_lateral, omega = motion_vector[0], motion_vector[1], motion_vector[2]
        x, y, theta = 0.0, 0.0, 0.0
        path = []
        for _ in range(steps):
            x += (-v_forward * math.sin(theta) - v_lateral * math.cos(theta)) * dt
            y += (v_forward * math.cos(theta) + v_lateral * math.sin(theta)) * dt
            theta += omega * dt
            path.append([x, y])
        return path

    def _extrapolate_robot_path_full(self, motion_vector, steps=50):
        """
        Cubic Bezier curve from robot to goal.

        The curve starts tangent to the current motion_vector direction and
        arrives at the goal with decreasing curvature (tight turn early,
        straightening out toward the end).

        Tunable via self.params["path_curvature"]:
            0.0 → straight line to goal
            1.0 → default (tangent handle = 1/3 of goal distance)
            >1  → exaggerated initial arc

        Args:
            motion_vector: [v_forward, v_lateral, omega_z]
            steps:         number of samples along the curve

        Returns:
            list of [x, y] points in BEV robot frame
        """
        if self._goal_rf is None:
            return []

        curvature = self.params.get("path_curvature", 0.5)

        # P0: robot at origin
        p0 = np.array([0.0, 0.0])

        # P3: goal in BEV coords [x_lateral, depth]
        p3 = np.array([self._goal_rf[0], self._goal_rf[1]])

        goal_dist = np.linalg.norm(p3)
        if goal_dist < 0.05:
            return []

        # Robot heading is always forward in robot frame
        heading = np.array([0.0, 1.0])

        # P1: extend along initial heading (controls departure curvature)
        tangent_len = curvature * goal_dist / 3.0
        p1 = p0 + heading * tangent_len

        # P2: pull back from goal along goal direction (smooth straight arrival)
        goal_dir = p3 / goal_dist
        p2 = p3 - goal_dir * (goal_dist / 3.0)

        # Evaluate cubic Bezier
        return self._bezier(p0, p1, p2, p3, steps=steps)
    
    def _construct_bezier(self, p0, tangent_len, p2, steps=50):
        if self._goal_rf is None:
            return []

        # P3: goal in BEV coords [x_lateral, depth]
        p3 = np.array([self._goal_rf[0], self._goal_rf[1]])

        goal_dist = np.linalg.norm(p3)
        if goal_dist < 0.05:
            return []

        # Robot heading is always forward in robot frame
        heading = np.array([0.0, 1.0])
        p1 = p0 + heading * tangent_len

        # Evaluate cubic Bezier
        return self._bezier(p0, p1, p2, p3, steps=steps)

    def _bezier(self, p0, p1, p2, p3, steps=50):
        """
        Evaluate cubic Bezier curve defined by control points p0, p1, p2, p3.

        Returns a list of [x, y] points along the curve.
        """
        path = []
        for i in range(steps + 1):
            t = i / steps
            s = 1.0 - t
            pt = s**3 * p0 + 3 * s**2 * t * p1 + 3 * s * t**2 * p2 + t**3 * p3
            path.append([float(pt[0]), float(pt[1])])
        return path