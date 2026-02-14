"""
Shared safety computation for Social-LOVON.
============================================

Core function: safety_score_at_point(point, human_positions, ...)
  - Heatmap calls it over a meshgrid via compute_safety_grid
  - Robot calls it at its own location via robot_safety_score
  - Combines all terms: gaussian proximity + trajectory prediction
"""

import math
import numpy as np

# ------------------------------------------------------------------
#  Safety parameters
# ------------------------------------------------------------------
SIGMA = 0.5    # Gaussian width (meters)
H = 1.0        # peak danger at distance=0
GAMMA = 0.99 # amount that trajectory penalty decreases in the future (ie: 0.9**0=1, 0.9**1=0.9, ...)
#             equivalent to discount factor


# ------------------------------------------------------------------
#  Term 1: Gaussian proximity
# ------------------------------------------------------------------

def gaussian_term(point_x, point_y, human_positions, sigma=SIGMA, h=H):
    """
    Gaussian proximity safety for a single point against all humans.

    Returns:
        float in [0, 1]. Worst (min) across all humans.
    """
    safety = 1.0
    for (hx, hy) in human_positions:
        dist = math.sqrt((point_x - hx) ** 2 + (point_y - hy) ** 2)
        s = 1.0 - h * math.exp(-(dist ** 2) / (2.0 * sigma ** 2))
        safety = min(safety, s)
    return safety


# ------------------------------------------------------------------
#  Term 2: Trajectory prediction threat
# ------------------------------------------------------------------

def trajectory_term(point_x, point_y, human_predicted_paths, sigma=SIGMA, h=H):
    """
    Trajectory-based safety: how close will predicted human paths
    come to this point in the future?

    Args:
        point_x, point_y: the query point
        human_predicted_paths: dict {track_id: [(x,y), (x,y), ...]}
                               predicted future positions per human.
                               None or empty means no trajectory data.
        sigma, h: Gaussian parameters

    Returns:
        float in [0, 1]. Worst (min) across all humans and timesteps.
    
        
    ***WILL I NEED THE ROBOT TRAJECTORY? ( no i dont think so)
    """
    if not human_predicted_paths:
        return 1.0

    safety = 1.0

    for track_id, path in human_predicted_paths.items():
        for i, (fx,fy) in enumerate(path):
            dist = math.sqrt((point_x - fx)**2 + (point_y - fy)**2)
            threat = ( GAMMA**i )* h * math.exp(-(dist**2) / (2.0 * sigma **2))
            s = 1.0 - threat
            safety = min(safety, s)

    return safety

    # for _tid, path in human_predicted_paths.items():
    #     for (fx, fy) in path:
    #         dist = math.sqrt((point_x - fx) ** 2 + (point_y - fy) ** 2)
    #         s = 1.0 - h * math.exp(-(dist ** 2) / (2.0 * sigma ** 2))
    #         safety = min(safety, s)

    # return safety

def socialforce_term(point_x, point_y, human_predicted_paths, sigma=SIGMA, h=H):
    return 1.0


# ------------------------------------------------------------------
#  THE safety function: score at a single point
# ------------------------------------------------------------------

def safety_score_at_point(point_x, point_y, human_positions,
                          human_predicted_paths=None, sigma=SIGMA, h=H):
    """
    Compute the safety score at a single (x, y) location.

    This is THE single source of truth for safety — both the heatmap
    and the robot score use this same function.

    Combines:
      - Gaussian proximity (current human positions)
      - Trajectory threat (predicted future human positions)

    Args:
        point_x: float, x coordinate
        point_y: float, y coordinate
        human_positions: list of (hx, hy) tuples, current positions
        human_predicted_paths: dict {track_id: [(x,y),...]} or None
        sigma:   Gaussian width parameter (meters)
        h:       peak danger at distance=0

    Returns:
        float in [0, 1]. 1.0 = safe, 0.0 = dangerous.
    """
    s_gauss = gaussian_term(point_x, point_y, human_positions, sigma, h)
    s_traj = trajectory_term(point_x, point_y, human_predicted_paths, sigma, h)
    s_sfm = socialforce_term(point_x, point_y, human_predicted_paths, sigma, h)

    score = min(s_gauss, s_traj, s_sfm) # output lowest safety score
    score = max(0.0, min(1.0, score)) # clip > 0
    return score


# ------------------------------------------------------------------
#  2D safety grid (heatmap) — just runs safety_score_at_point on grid
# ------------------------------------------------------------------

def compute_safety_grid(human_positions, xlim, ylim, resolution=0.1,
                        human_predicted_paths=None, sigma=SIGMA, h=H):
    """
    Compute a 2D safety field by evaluating safety_score_at_point
    over a meshgrid.

    Args:
        human_positions: list of (px, py) tuples in world frame
        xlim:           (xmin, xmax)
        ylim:           (ymin, ymax)
        resolution:     grid cell size in meters
        human_predicted_paths: dict {track_id: [(x,y),...]} or None
        sigma:          Gaussian width
        h:              peak danger

    Returns:
        safety_grid: 2D numpy array, shape (ny, nx), values in [0, 1]
        extent:      [xmin, xmax, ymin, ymax] for matplotlib imshow
    """
    x = np.arange(xlim[0], xlim[1], resolution)
    y = np.arange(ylim[0], ylim[1], resolution)
    safety = np.empty((len(y), len(x)), dtype=np.float64)

    for i, yi in enumerate(y):
        for j, xj in enumerate(x):
            safety[i, j] = safety_score_at_point(
                xj, yi, human_positions,
                human_predicted_paths=human_predicted_paths,
                sigma=sigma, h=h,
            )

    extent = [xlim[0], xlim[1], ylim[0], ylim[1]]
    return safety, extent


# ------------------------------------------------------------------
#  Robot safety score — safety_score_at_point at robot's location
# ------------------------------------------------------------------

def robot_safety_score(robot_x, robot_y, human_positions,
                       human_predicted_paths=None, sigma=SIGMA, h=H):
    """
    Compute the safety score at the robot's current position.

    Same function as the heatmap uses, evaluated at one point.

    Args:
        robot_x, robot_y: robot position
        human_positions:   list of (hx, hy) tuples
        human_predicted_paths: dict {track_id: [(x,y),...]} or None
        sigma, h:          Gaussian parameters

    Returns:
        float in [0, 1].
    """
    return safety_score_at_point(robot_x, robot_y, human_positions,
                                human_predicted_paths=human_predicted_paths,
                                sigma=sigma, h=h)



