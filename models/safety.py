"""
Shared safety computation for Social-LOVON.
============================================
"""

import math
import numpy as np


# ------------------------------------------------------------------
#  Term 1: Gaussian proximity
# ------------------------------------------------------------------

def gaussian_proximity_score(distance, sigma=0.8, h=1.0):
    """
    s = 1 - h * exp(-d^2 / (2 * sigma^2))

    Args:
        distance: scalar or numpy array of distances (meters)
        sigma:    Gaussian width parameter (meters)
        h:        peak danger at distance=0
    """
    d = np.asarray(distance, dtype=np.float64)
    score = 1.0 - h * np.exp(-(d ** 2) / (2.0 * sigma ** 2))
    return np.clip(score, 0.0, 1.0)


# ------------------------------------------------------------------
#  Term 2: Trajectory safety
# ------------------------------------------------------------------

def trajectory_safety_score(robot_path, human_paths, sigma=0.8, h=1.0):
    """
    Worst-case safety score across all future timesteps and humans.

    For each timestep, computes distance between robot and each human's
    predicted position. Returns the worst (lowest) Gaussian safety score.

    Args:
        robot_path:  list of [x, y] robot predicted positions (robot frame)
        human_paths: dict {track_id: list of [x, y]} human predicted paths
        sigma:       Gaussian width parameter
        h:           peak danger at distance=0

    Returns:
        float in [0, 1]. 1.0 if no human paths.
    """
    if not human_paths or not robot_path:
        return 1.0

    worst_score = 1.0
    robot_arr = np.asarray(robot_path, dtype=np.float64)

    for _tid, hpath in human_paths.items():
        h_arr = np.asarray(hpath, dtype=np.float64)
        n = min(len(robot_arr), len(h_arr))
        if n == 0:
            continue
        diffs = robot_arr[:n] - h_arr[:n]
        dists = np.sqrt(np.sum(diffs ** 2, axis=1))
        min_dist = float(np.min(dists))
        s = float(gaussian_proximity_score(min_dist, sigma=sigma, h=h))
        worst_score = min(worst_score, s)

    return worst_score


# ------------------------------------------------------------------
#  Combiner: skeleton that merges all terms
# ------------------------------------------------------------------

def combined_safety_score(distances, robot_path=None, human_paths=None,
                          sigma=0.8, h=1.0):
    """
    Combine all safety terms into a single score.

    Current terms:
      - s_prox: Gaussian proximity (closest human distance)
      - s_traj: trajectory prediction threat

    Add new terms here as they are developed.

    Args:
        distances:   list of float distances to each human (meters), may be empty
        robot_path:  list of [x, y] robot predicted positions (for s_traj)
        human_paths: dict {track_id: list of [x, y]} human predicted paths
        sigma:       Gaussian width
        h:           peak danger

    Returns:
        float in [0, 1]. 1.0 = safe, 0.0 = dangerous.
    """
    # Term 1: proximity
    if distances:
        d_min = min(distances)
        s_prox = float(gaussian_proximity_score(d_min, sigma=sigma, h=h))
    else:
        s_prox = 1.0

    # Term 2: trajectory
    s_traj = trajectory_safety_score(
        robot_path or [], human_paths or {}, sigma=sigma, h=h
    )

    # --- Combine (add new terms to this min) ---
    score = min(s_prox, s_traj)
    return max(0.0, min(1.0, score))


# ------------------------------------------------------------------
#  2D safety grid (for heatmap visualization)
# ------------------------------------------------------------------

def compute_safety_grid(human_positions, xlim, ylim, resolution=0.1,
                        sigma=0.8, h=1.0):
    """
    Compute a 2D safety field over a grid in world frame.

    For each grid cell, computes distance to every human and calls
    combined_safety_score. Currently only the proximity term contributes
    (trajectory data is not available at the grid level).

    Args:
        human_positions: list of (px, py) tuples in world frame
        xlim:           (xmin, xmax)
        ylim:           (ymin, ymax)
        resolution:     grid cell size in meters
        sigma:          Gaussian width
        h:              peak danger

    Returns:
        safety_grid: 2D numpy array, shape (ny, nx), values in [0, 1]
        extent:      [xmin, xmax, ymin, ymax] for matplotlib imshow
    """
    x = np.arange(xlim[0], xlim[1], resolution)
    y = np.arange(ylim[0], ylim[1], resolution)
    xx, yy = np.meshgrid(x, y)

    safety = np.ones_like(xx)
    for (hx, hy) in human_positions:
        dist = np.sqrt((xx - hx) ** 2 + (yy - hy) ** 2)
        # Each human contributes a proximity term; combined_safety_score
        # applied per-human (grid has no trajectory data, so s_traj=1.0)
        per_human = gaussian_proximity_score(dist, sigma=sigma, h=h)
        safety = np.minimum(safety, per_human)

    safety = np.clip(safety, 0.0, 1.0)
    extent = [xlim[0], xlim[1], ylim[0], ylim[1]]
    return safety, extent


# ------------------------------------------------------------------
#  Coordinate transform: world frame -> robot frame
# ------------------------------------------------------------------

def world_to_robot_frame(human_px, human_py, robot_px, robot_py, robot_theta):
    """
    Transform a world-frame position into robot-frame [x_lateral, depth].

    Robot frame convention (matching SocialNavigator):
        x_lateral: positive = right of robot
        depth:     positive = forward from robot
    """
    dx = human_px - robot_px
    dy = human_py - robot_py
    cos_t = math.cos(-robot_theta)
    sin_t = math.sin(-robot_theta)
    x_rot = dx * cos_t - dy * sin_t
    y_rot = dx * sin_t + dy * cos_t
    depth = x_rot
    x_lateral = -y_rot
    return x_lateral, depth


def world_to_robot_frame_velocity(human_vx, human_vy, robot_theta):
    """
    Transform a world-frame velocity into robot-frame [vx_lateral, v_depth].
    Same rotation as position but without translation.
    """
    cos_t = math.cos(-robot_theta)
    sin_t = math.sin(-robot_theta)
    x_rot = human_vx * cos_t - human_vy * sin_t
    y_rot = human_vx * sin_t + human_vy * cos_t
    v_depth = x_rot
    vx_lateral = -y_rot
    return vx_lateral, v_depth
