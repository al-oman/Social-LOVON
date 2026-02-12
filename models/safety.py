"""
Shared safety computation for Social-LOVON.
============================================
"""

import math
import numpy as np


# ------------------------------------------------------------------
#  Core Gaussian safety function
# ------------------------------------------------------------------

def gaussian_proximity_score(distance, sigma=0.8, h=1.0):
    """
    Compute safety score for a single distance value.

    s = 1 - h * exp(-d^2 / (2 * sigma^2))

    Args:
        distance: scalar or numpy array of distances (meters)
        sigma:    Gaussian width parameter (meters)
        h:        peak danger at distance=0 (should be 1.0 for full [0,1] range)

    Returns:
        Safety score(s) in [0, 1].  1.0 = safe, 0.0 = dangerous.
    """
    d = np.asarray(distance, dtype=np.float64)
    score = 1.0 - h * np.exp(-(d ** 2) / (2.0 * sigma ** 2))
    return np.clip(score, 0.0, 1.0)


def aggregate_safety_score(distances, sigma=0.8, h=1.0):
    """
    Compute aggregate safety score from a list of human distances.

    Uses minimum distance (most dangerous human dominates).

    Args:
        distances: list of float distances (meters), may be empty
        sigma:     Gaussian width
        h:         peak danger

    Returns:
        float safety score in [0, 1]. Returns 1.0 if distances is empty.
    """
    if not distances:
        return 1.0
    d_min = min(distances)
    return float(gaussian_proximity_score(d_min, sigma=sigma, h=h))


# ------------------------------------------------------------------
#  2D safety field (for heatmap visualization)
# ------------------------------------------------------------------

def compute_safety_grid(human_positions, xlim, ylim, resolution=0.1,
                        sigma=0.8, h=1.0):
    """
    Compute a 2D safety field over a grid in world frame.

    For each grid cell, computes the distance to every human and
    returns the minimum safety score (most-dangerous-human dominates).

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
        per_human = gaussian_proximity_score(dist, sigma=sigma, h=h)
        safety = np.minimum(safety, per_human)

    safety = np.clip(safety, 0.0, 1.0)
    extent = [xlim[0], xlim[1], ylim[0], ylim[1]]
    return safety, extent


def trajectory_safety_score(robot_path, human_paths, sigma=0.8, h=1.0):
    """
    Compute worst-case safety score across all future timesteps and humans.

    For each future timestep, finds the minimum distance between the robot's
    predicted position and each human's predicted position. Returns the worst
    (lowest) safety score across all humans and timesteps.

    Args:
        robot_path:  list of [x, y] robot predicted positions (robot frame)
        human_paths: dict {track_id: list of [x, y]} human predicted paths
                     (robot frame). Each path should have the same length
                     as robot_path (or shorter — extra robot steps are ignored).
        sigma:       Gaussian width parameter
        h:           peak danger at distance=0

    Returns:
        float safety score in [0, 1]. Returns 1.0 if no human paths.
    """
    if not human_paths or not robot_path:
        return 1.0

    worst_score = 1.0
    robot_arr = np.asarray(robot_path, dtype=np.float64)

    for _tid, hpath in human_paths.items():
        h_arr = np.asarray(hpath, dtype=np.float64)
        # Compare timestep-by-timestep up to the shorter path length
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
#  Coordinate transform: world frame -> robot frame
# ------------------------------------------------------------------

def world_to_robot_frame(human_px, human_py, robot_px, robot_py, robot_theta):
    """
    Transform a world-frame position into robot-frame [x_lateral, depth].

    Robot frame convention (matching SocialNavigator):
        x_lateral: positive = right of robot
        depth:     positive = forward from robot

    Args:
        human_px, human_py: human world-frame position
        robot_px, robot_py: robot world-frame position
        robot_theta:        robot heading in world frame (radians)

    Returns:
        (x_lateral, depth) in robot frame
    """
    dx = human_px - robot_px
    dy = human_py - robot_py
    cos_t = math.cos(-robot_theta)
    sin_t = math.sin(-robot_theta)
    # Rotate by -theta to align with robot heading
    # x_rot aligns with robot forward, y_rot aligns with robot left
    x_rot = dx * cos_t - dy * sin_t
    y_rot = dx * sin_t + dy * cos_t
    depth = x_rot
    x_lateral = -y_rot  # right = negative left
    return x_lateral, depth


def world_to_robot_frame_velocity(human_vx, human_vy, robot_theta):
    """
    Transform a world-frame velocity into robot-frame [vx_lateral, v_depth].

    Same rotation as position but without translation.

    Args:
        human_vx, human_vy: human world-frame velocity
        robot_theta:        robot heading in world frame (radians)

    Returns:
        (vx_lateral, v_depth) in robot frame
    """
    cos_t = math.cos(-robot_theta)
    sin_t = math.sin(-robot_theta)
    x_rot = human_vx * cos_t - human_vy * sin_t
    y_rot = human_vx * sin_t + human_vy * cos_t
    v_depth = x_rot
    vx_lateral = -y_rot
    return vx_lateral, v_depth
