"""
ArUco marker-based goal detection for physical deployment.

Two ArUco markers placed ~8ft apart define the goal as their midpoint.
Replaces YOLO-based goal detection while keeping pose (human) detection intact.
"""

import numpy as np
import cv2


class ArucoGoalDetector:
    """Detects two ArUco markers and returns goal state as their midpoint."""

    # Approximate separation between the two markers in meters (~8 ft)
    MARKER_SEPARATION_M = 2.0

    def __init__(self, marker_ids=(0, 1), marker_size_m=0.15,
                 camera_matrix=None, dist_coeffs=None, ema_smooth=True,
                 hold_frames=10):
        self.marker_ids = tuple(marker_ids)
        self.marker_size_m = marker_size_m
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs if dist_coeffs is not None else np.zeros(5)

        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)

        # Smoothing / memory state
        self._ema_smooth = ema_smooth
        self._ema_alpha = 0.3
        self._last_centers = {}   # marker_id → last-known pixel center
        self._last_depths = {}    # marker_id → last-known solvePnP depth
        self._smoothed_depth = None
        self._smoothed_xyn = None

        # Detection hold — return last valid result for N frames after losing sight
        self._hold_frames = hold_frames
        self._frames_since_detect = 0
        self._last_valid_result = None

    def detect(self, frame):
        """Detect ArUco markers and return object_state dict.

        Returns dict with keys: predicted_object, confidence, object_xyn,
        object_whn, bounding_box.  Returns NULL detection if no markers found.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners_list, ids, _ = self.detector.detectMarkers(gray)

        h_img, w_img = frame.shape[:2]

        if ids is None:
            return self._null_detection()

        ids_flat = ids.flatten()

        # Find our target markers
        found = {}
        for target_id in self.marker_ids:
            matches = np.where(ids_flat == target_id)[0]
            if len(matches) > 0:
                idx = matches[0]
                corners = corners_list[idx][0]  # (4, 2)
                center = corners.mean(axis=0)    # (2,) pixel coords
                found[target_id] = {"corners": corners, "center": center, "idx": idx}

        if len(found) == 0:
            return self._null_detection()

        # Estimate pose for depth if we have camera intrinsics
        depths = {}
        if self.camera_matrix is not None:
            half = self.marker_size_m / 2.0
            obj_pts = np.array([
                [-half,  half, 0],
                [ half,  half, 0],
                [ half, -half, 0],
                [-half, -half, 0],
            ], dtype=np.float32)
            for tid, info in found.items():
                _, rvec, tvec = cv2.solvePnP(
                    obj_pts,
                    info["corners"].astype(np.float32),
                    self.camera_matrix,
                    self.dist_coeffs,
                )
                depths[tid] = float(tvec[2][0])  # z = depth
        # Update memory for each visible marker
        for tid, info in found.items():
            self._last_centers[tid] = info["center"].copy()
            if tid in depths:
                self._last_depths[tid] = depths[tid]

        if len(found) == 2:
            # Both markers visible — midpoint
            centers = [found[tid]["center"] for tid in self.marker_ids]
            midpoint_px = (centers[0] + centers[1]) / 2.0

            # Bounding box: union of both marker corners
            all_corners = np.vstack([found[tid]["corners"] for tid in self.marker_ids])
            x_min, y_min = all_corners.min(axis=0)
            x_max, y_max = all_corners.max(axis=0)

            if depths:
                depth_vals = [depths[tid] for tid in self.marker_ids]
                avg_depth = sum(depth_vals) / 2.0
            else:
                avg_depth = None

        else:
            # Only 1 marker visible — estimate midpoint from separation
            tid = list(found.keys())[0]
            other_tid = [m for m in self.marker_ids if m != tid][0]
            center = found[tid]["center"]
            corners = found[tid]["corners"]

            if depths and self.camera_matrix is not None:
                # Fallback: estimate from separation
                d = depths[tid]
                fx = self.camera_matrix[0, 0]
                sep_px = fx * self.MARKER_SEPARATION_M / d
                if center[0] < w_img / 2:
                    midpoint_px = center + np.array([sep_px / 2, 0])
                else:
                    midpoint_px = center - np.array([sep_px / 2, 0])
                avg_depth = d
            else:
                midpoint_px = center
                avg_depth = None

            x_min, y_min = corners.min(axis=0)
            x_max, y_max = corners.max(axis=0)

        # Normalize to [0, 1]
        xn = float(np.clip(midpoint_px[0] / w_img, 0, 1))
        yn = float(np.clip(midpoint_px[1] / h_img, 0, 1))

        if self._ema_smooth:
            a = self._ema_alpha
            if self._smoothed_xyn is None:
                self._smoothed_xyn = [xn, yn]
            else:
                self._smoothed_xyn[0] = a * xn + (1 - a) * self._smoothed_xyn[0]
                self._smoothed_xyn[1] = a * yn + (1 - a) * self._smoothed_xyn[1]
            if avg_depth is not None:
                if self._smoothed_depth is None:
                    self._smoothed_depth = avg_depth
                else:
                    self._smoothed_depth = a * avg_depth + (1 - a) * self._smoothed_depth
            xn = self._smoothed_xyn[0]
            yn = self._smoothed_xyn[1]
            avg_depth = self._smoothed_depth

        box_w = (x_max - x_min) / w_img
        box_h = (y_max - y_min) / h_img

        result = {
            "predicted_object": "aruco_goal",
            "confidence": [0.99],
            "object_xyn": [xn, yn],
            "object_whn": [float(box_w), float(box_h)],
            "bounding_box": [int(x_min), int(y_min), int(x_max), int(y_max)],
            "goal_depth": avg_depth,
        }
        self._last_valid_result = result
        self._frames_since_detect = 0
        return result

    def _null_detection(self):
        self._frames_since_detect += 1
        # Hold last valid detection for N frames before giving up
        if (self._last_valid_result is not None
                and self._frames_since_detect <= self._hold_frames):
            held = dict(self._last_valid_result)
            held["confidence"] = [0.5]  # signal it's a held detection
            return held
        # Beyond hold window — let ego-motion compensation maintain the goal
        return {
            "predicted_object": "NULL",
            "confidence": [0.0],
            "object_xyn": [0.0, 0.0],
            "object_whn": [0.0, 0.0],
            "bounding_box": None,
            "goal_depth": None,
        }
