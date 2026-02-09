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
import logging
from typing import Dict, List, Optional, Tuple

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

        # --- Distance estimation (not yet populated) ---
        self.distance_lidar = None       # meters, from lidar
        self.distance_mono = None        # meters, from monocular approx
        self.distance = None             # meters, fused best-estimate

        # --- Tracking / prediction (not yet populated) ---
        self.velocity = None             # (vx, vy) in robot frame, m/s
        self.predicted_path = None       # list of (x, y) future positions
        self.orientation = None          # radians, body heading
        self.last_seen = time.time()

    def __repr__(self):
        d = f"{self.distance:.2f}m" if self.distance is not None else "?"
        return f"<Human id={self.track_id} dist={d} conf={self.confidence:.2f}>"


class SocialNavigator:
    """
    Socially-aware navigation layer for LOVON.

    Parameters (all tunable, placeholder defaults)
    ------------------------------------------------
    sigma           : float  – Gaussian proximity std dev (meters)
    h               : float  – Gaussian proximity peak cost
    lambda_v        : float  – relative-velocity weighting
    beta_traj       : float  – trajectory obstruction weight
    d_safe          : float  – safe distance threshold (meters)
    d_yield         : float  – distance to trigger yielding (meters)
    d_resume        : float  – distance to exit yielding (meters)
    horizon_s       : float  – prediction horizon (seconds)
    horizon_steps   : int    – prediction time-steps
    k_avoid         : float  – avoidance steering gain (rad/s)
    v_min_scale     : float  – minimum speed scale (0–1)
    d_max           : float  – ignore humans beyond this (meters)
    mono_k          : float  – monocular depth constant (pixels·meters)
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
        "mono_k": 300.0,       # placeholder: d ≈ mono_k / bbox_height_px
    }

    def __init__(self, enabled=False, **kwargs):
        """
        Args:
            enabled: If False, step() is a pure passthrough (no computation).
                     Set True to activate social cost logging (still no
                     velocity modification until modulation is implemented).
        """
        self.enabled = enabled
        self.params = {**self.DEFAULT_PARAMS, **kwargs}

        # --- Tracked human registry ---
        self._tracked_humans: Dict[int, TrackedHuman] = {}
        self._next_id = 0

        # --- Yielding state ---
        self.is_yielding = False
        self._yield_enter_time = None
        self._clear_frames = 0           # consecutive frames with no yield trigger

        # --- Diagnostics (read by UI / logging) ---
        self.diag = {
            "num_humans": 0,
            "min_distance": None,
            "total_prox_cost": 0.0,
            "total_traj_cost": 0.0,
            "speed_scale": 1.0,
            "is_yielding": False,
        }

        logger.info(
            "SocialNavigator initialized  enabled=%s  params=%s",
            self.enabled, self.params,
        )

    # ================================================================== #
    #  PUBLIC API                                                         #
    # ================================================================== #

    def step(
        self,
        motion_vector,
        pose_state: dict,
        mission_state: str,
        lidar_ranges=None,
    ):
        """
        Main entry point — called once per control cycle.

        Args:
            motion_vector : list/array [v_x, v_y, omega_z] from L2MM
            pose_state    : dict from controller.pose_state
                            {"num_people", "poses", "pose_boxes"}
            mission_state : str, current state machine state
            lidar_ranges  : reserved — raw lidar scan (not yet wired)

        Returns:
            motion_vector : [v_x, v_y, omega_z]  (unmodified for now)
        """
        if not self.enabled:
            return motion_vector

        # 1. Parse detections from pose_state
        detections = self._parse_pose_state(pose_state)

        # 2. Estimate distances
        self._estimate_distances(detections, lidar_ranges)

        # 3. Update tracker (simple ID assignment for now)
        self._update_tracker(detections)

        # 4. Predict future trajectories (placeholder)
        self._predict_trajectories()

        # 5. Compute social costs (placeholder — returns zeros)
        prox_cost, traj_cost = self._compute_social_costs()

        # 6. Evaluate yielding conditions (placeholder)
        self._evaluate_yielding(mission_state)

        # 7. Modulate velocity (DISABLED — passthrough)
        modified_vector = self._modulate_velocity(motion_vector, prox_cost, traj_cost)

        # 8. Update diagnostics
        self._update_diagnostics(prox_cost, traj_cost)

        return modified_vector

    # ================================================================== #
    #  STAGE 1 — Parse pose_state into detection dicts                    #
    # ================================================================== #

    def _parse_pose_state(self, pose_state: dict) -> List[dict]:
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
    #  STAGE 2 — Distance estimation                                      #
    # ================================================================== #

    def _estimate_distances(self, detections: List[dict], lidar_ranges=None):
        """
        Populate each detection with a distance estimate.

        Priority:
            1. Lidar (when wired)
            2. Monocular approximation  d ≈ mono_k / bbox_height

        Both paths are skeletons — lidar is not yet connected,
        monocular constant needs calibration.
        """
        for det in detections:
            # --- Lidar path (placeholder) ---
            det["distance_lidar"] = self._estimate_distance_lidar(det, lidar_ranges)

            # --- Monocular path (placeholder) ---
            det["distance_mono"] = self._estimate_distance_mono(det)

            # --- Fuse (placeholder: prefer lidar, fall back to mono) ---
            if det["distance_lidar"] is not None:
                det["distance"] = det["distance_lidar"]
            elif det["distance_mono"] is not None:
                det["distance"] = det["distance_mono"]
            else:
                det["distance"] = None

    def _estimate_distance_lidar(self, det: dict, lidar_ranges) -> Optional[float]:
        """
        Estimate distance to a detected human using lidar scan.

        TODO:
            - Accept lidar_ranges (e.g. LaserScan or point cloud slice)
            - Project bbox center into lidar frame
            - Extract range at corresponding angle
            - Handle occlusion / multi-return
        """
        if lidar_ranges is None:
            return None

        # --- Skeleton: not yet implemented ---
        # angle = self._pixel_to_lidar_angle(det["center_px"])
        # distance = lidar_ranges[angle_index]
        # return distance
        return None

    def _estimate_distance_mono(self, det: dict) -> Optional[float]:
        """
        Monocular depth approximation:  d ≈ mono_k / bbox_height_px

        TODO: Calibrate mono_k per camera (Go2 front camera focal length).
        """
        h = det.get("bbox_height", 0)
        if h < 10:  # too small to be reliable
            return None
        return self.params["mono_k"] / h

    # ================================================================== #
    #  STAGE 3 — Tracker (simple nearest-neighbor, placeholder)           #
    # ================================================================== #

    def _update_tracker(self, detections: List[dict]):
        """
        Minimal ID-assignment tracker (not ByteTrack yet).

        TODO: Replace with ByteTrack IoU-based tracker for robust
              multi-human tracking with persistent IDs.
        """
        now = time.time()

        # For now: assign sequential IDs, no re-identification
        self._tracked_humans.clear()
        for i, det in enumerate(detections):
            tid = i  # placeholder — no persistent IDs yet
            human = TrackedHuman(track_id=tid)
            human.bbox = det.get("bbox")
            human.position_image = det.get("center_px")
            human.keypoints = det.get("keypoints")
            human.keypoints_conf = det.get("keypoints_conf")
            human.confidence = det.get("confidence", 0.0)
            human.distance_lidar = det.get("distance_lidar")
            human.distance_mono = det.get("distance_mono")
            human.distance = det.get("distance")
            human.last_seen = now
            self._tracked_humans[tid] = human

    # ================================================================== #
    #  STAGE 4 — Trajectory prediction (placeholder)                      #
    # ================================================================== #

    def _predict_trajectories(self):
        """
        Predict future positions for each tracked human.

        TODO:
            - Maintain velocity history per track ID
            - Constant-velocity extrapolation over horizon
            - Requires robot-frame positions (needs distance + angle)
        """
        for human in self._tracked_humans.values():
            human.velocity = None
            human.predicted_path = None

    # ================================================================== #
    #  STAGE 5 — Social cost computation (placeholder)                    #
    # ================================================================== #

    def _compute_social_costs(self) -> Tuple[float, float]:
        """
        Compute aggregate proximity and trajectory costs.

        Returns:
            (total_prox_cost, total_traj_cost) — both 0.0 for now.

        TODO:
            - Gaussian proximity: C(d) = h * exp(-d^2 / 2σ^2)
            - Relative velocity scaling
            - Trajectory obstruction penalty
        """
        total_prox = 0.0
        total_traj = 0.0

        for human in self._tracked_humans.values():
            if human.distance is not None:
                # Placeholder: log distance, compute nothing yet
                logger.debug("  human %d  dist=%.2fm", human.track_id, human.distance)

        return total_prox, total_traj

    # ================================================================== #
    #  STAGE 6 — Yielding evaluation (placeholder)                        #
    # ================================================================== #

    def _evaluate_yielding(self, mission_state: str):
        """
        Determine whether the robot should enter/exit the yielding state.

        TODO:
            - Check d_min < d_yield
            - Check approach angle (cos_θ > 0.5)
            - Check predicted path intersection
            - Manage _clear_frames counter for exit
            - Timeout after 15s → exit yielding
        """
        self.is_yielding = False  # always off for now

    # ================================================================== #
    #  STAGE 7 — Velocity modulation (DISABLED — passthrough)             #
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
    #  STAGE 8 — Diagnostics                                              #
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
                f"{self.diag['min_distance']:.2f}m" if self.diag["min_distance"] else "n/a",
                prox_cost, traj_cost, self.is_yielding,
            )