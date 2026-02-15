"""
Shared safety computation for Social-LOVON.
============================================

All grid/point evaluations are vectorized with numpy.
Single source of truth: _safety_scores() handles both
scalar points and 2D meshgrids.
"""

import numpy as np

# ------------------------------------------------------------------
#  Safety parameters
# ------------------------------------------------------------------
SIGMA = 0.5    # Gaussian width (meters)
H = 1.0        # peak danger at distance=0
GAMMA = 0.99   # trajectory discount factor


# ------------------------------------------------------------------
#  Vectorized core (works on scalars, 1D arrays, or 2D meshgrids)
# ------------------------------------------------------------------

def _gaussian_grid(X, Y, human_positions, sigma=SIGMA, h=H):
    """Min-over-humans Gaussian proximity. Returns array same shape as X."""
    if len(human_positions) == 0:
        return np.ones_like(X, dtype=np.float64)

    hp = np.asarray(human_positions, dtype=np.float64)  # (N, 2)
    inv_2s2 = 1.0 / (2.0 * sigma * sigma)

    safety = np.ones_like(X, dtype=np.float64)
    for i in range(hp.shape[0]):
        dist_sq = (X - hp[i, 0]) ** 2 + (Y - hp[i, 1]) ** 2
        s = 1.0 - h * np.exp(-dist_sq * inv_2s2)
        np.minimum(safety, s, out=safety)
    return safety


def _trajectory_grid(X, Y, human_predicted_paths, sigma=SIGMA, h=H):
    """Min-over-humans-and-timesteps trajectory threat. Returns array same shape as X."""
    if not human_predicted_paths:
        return np.ones_like(X, dtype=np.float64)

    inv_2s2 = 1.0 / (2.0 * sigma * sigma)
    safety = np.ones_like(X, dtype=np.float64)

    for _tid, path in human_predicted_paths.items():
        if not path:
            continue
        pts = np.asarray(path, dtype=np.float64)  # (T, 2)
        gammas = GAMMA ** np.arange(pts.shape[0])  # (T,)
        for t in range(pts.shape[0]):
            dist_sq = (X - pts[t, 0]) ** 2 + (Y - pts[t, 1]) ** 2
            s = 1.0 - gammas[t] * h * np.exp(-dist_sq * inv_2s2)
            np.minimum(safety, s, out=safety)
    return safety


def _safety_scores(X, Y, human_positions,
                   human_predicted_paths=None, sigma=SIGMA, h=H):
    """Combined safety. X, Y can be any broadcastable shape."""
    s_gauss = _gaussian_grid(X, Y, human_positions, sigma, h)
    s_traj = _trajectory_grid(X, Y, human_predicted_paths, sigma, h)
    np.minimum(s_gauss, s_traj, out=s_gauss)
    np.clip(s_gauss, 0.0, 1.0, out=s_gauss)
    return s_gauss


# ------------------------------------------------------------------
#  Public API (unchanged signatures)
# ------------------------------------------------------------------

def safety_score_at_point(point_x, point_y, human_positions,
                          human_predicted_paths=None, sigma=SIGMA, h=H):
    """Safety score at a single (x, y). Returns float in [0, 1]."""
    result = _safety_scores(
        np.float64(point_x), np.float64(point_y),
        human_positions, human_predicted_paths, sigma, h)
    return float(result)


def compute_safety_grid(human_positions, xlim, ylim, resolution=0.1,
                        human_predicted_paths=None, sigma=SIGMA, h=H):
    """Vectorized 2D safety field. Returns (grid, extent)."""
    x = np.arange(xlim[0], xlim[1], resolution)
    y = np.arange(ylim[0], ylim[1], resolution)
    X, Y = np.meshgrid(x, y)

    safety = _safety_scores(X, Y, human_positions,
                            human_predicted_paths, sigma, h)

    extent = [xlim[0], xlim[1], ylim[0], ylim[1]]
    return safety, extent


def robot_safety_score(robot_x, robot_y, human_positions,
                       human_predicted_paths=None, sigma=SIGMA, h=H):
    """Safety score at the robot's current position. Returns float in [0, 1]."""
    return safety_score_at_point(robot_x, robot_y, human_positions,
                                human_predicted_paths=human_predicted_paths,
                                sigma=sigma, h=h)



