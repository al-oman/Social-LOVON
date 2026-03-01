"""
Shared safety computation for Social-LOVON.
============================================

All grid/point evaluations are vectorized with numpy.
Single source of truth: _safety_scores() handles both
scalar points and 2D meshgrids.

Gaussian shape parameters
-------------------------
- sigma         : width of the Gaussian at the human's current position (meters)
- h             : peak danger amplitude at distance=0 (0..1)
- gamma         : per-step multiplier on H along the predicted trajectory.
                  gamma < 1 → danger fades into the future (conservative)
                  gamma = 1 → constant danger along trajectory
                  gamma > 1 → danger grows into the future (aggressive)
- sigma_spread  : sigma grows by this amount per prediction step (meters/step).
                  Larger → future positions have wider, softer Gaussians.
- h_traj_scale  : per-step multiplier on H along the trajectory (applied on
                  top of gamma).  Lets you scale the peak independently of
                  the gamma scaling.
                  h_traj_scale < 1 → peak shrinks along trajectory
                  h_traj_scale = 1 → peak unchanged (default)
                  h_traj_scale > 1 → peak grows along trajectory
"""

import numpy as np

# ------------------------------------------------------------------
#  Safety parameter defaults
# ------------------------------------------------------------------
SIGMA = 0.75          # Gaussian width at current position (meters)
H = 1.0              # peak danger at distance=0
GAMMA = 1.00         # trajectory H multiplier per step
SIGMA_SPREAD = 0.05   # sigma growth per prediction step (meters/step)
H_TRAJ_SCALE = 1.00        # per-step H multiplier along trajectory (<1 shrinks, >1 grows)


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


def _trajectory_grid(X, Y, human_predicted_paths,
                     sigma=SIGMA, h=H, gamma=GAMMA,
                     sigma_spread=SIGMA_SPREAD, h_traj_scale=H_TRAJ_SCALE):
    """Min-over-humans-and-timesteps trajectory threat with anisotropic Gaussians.

    Each waypoint's Gaussian uses the base *sigma* along the trajectory tangent
    and ``sigma + sigma_spread * (t+1)`` perpendicular to it.  This prevents
    future-step spread from inflating the field behind (or ahead along) the
    trajectory — uncertainty only grows laterally.

    For the first waypoint (or when the tangent is degenerate), falls back to
    an isotropic Gaussian at base sigma.
    """
    if not human_predicted_paths:
        return np.ones_like(X, dtype=np.float64)

    safety = np.ones_like(X, dtype=np.float64)

    for _tid, path in human_predicted_paths.items():
        if not path:
            continue
        pts = np.asarray(path, dtype=np.float64)  # (T, 2)
        T = pts.shape[0]
        for t in range(T):
            h_t = h * (gamma ** t) * (h_traj_scale ** t)

            # Compute tangent direction from adjacent waypoints
            if T >= 2:
                if t == 0:
                    tangent = pts[1] - pts[0]
                elif t == T - 1:
                    tangent = pts[T - 1] - pts[T - 2]
                else:
                    tangent = pts[t + 1] - pts[t - 1]
                tang_len = np.sqrt(tangent[0] ** 2 + tangent[1] ** 2)
            else:
                tang_len = 0.0

            # Displacement from this waypoint
            dx = X - pts[t, 0]
            dy = Y - pts[t, 1]

            if tang_len > 1e-9:
                # Unit tangent and normal
                tx, ty = tangent[0] / tang_len, tangent[1] / tang_len
                # Project displacement onto tangent (along) and normal (perp)
                d_along = dx * tx + dy * ty
                d_perp = dx * (-ty) + dy * tx

                sigma_along = sigma
                sigma_perp = sigma + sigma_spread * (t + 1)

                inv_along = 1.0 / (2.0 * sigma_along * sigma_along)
                inv_perp = 1.0 / (2.0 * sigma_perp * sigma_perp)

                exponent = d_along ** 2 * inv_along + d_perp ** 2 * inv_perp
            else:
                # Degenerate tangent (stationary) — isotropic at base sigma
                dist_sq = dx ** 2 + dy ** 2
                exponent = dist_sq / (2.0 * sigma * sigma)

            s = 1.0 - h_t * np.exp(-exponent)
            np.minimum(safety, s, out=safety)
    return safety


def _safety_scores(X, Y, human_positions,
                   human_predicted_paths=None, sigma=SIGMA, h=H,
                   human_traj_pred=True, gamma=GAMMA,
                   sigma_spread=SIGMA_SPREAD, h_traj_scale=H_TRAJ_SCALE):
    """Combined safety. X, Y can be any broadcastable shape."""
    s_gauss = _gaussian_grid(X, Y, human_positions, sigma, h)
    if human_traj_pred:
        s_traj = _trajectory_grid(X, Y, human_predicted_paths,
                                  sigma, h, gamma, sigma_spread, h_traj_scale)
    else:
        s_traj = np.ones_like(s_gauss)
    np.minimum(s_gauss, s_traj, out=s_gauss)
    np.clip(s_gauss, 0.0, 1.0, out=s_gauss)
    return s_gauss


# ------------------------------------------------------------------
#  Public API
# ------------------------------------------------------------------

def safety_score_at_point(point_x, point_y, human_positions,
                          human_predicted_paths=None, sigma=SIGMA, h=H,
                          human_traj_pred=True, gamma=GAMMA,
                          sigma_spread=SIGMA_SPREAD, h_traj_scale=H_TRAJ_SCALE):
    """Safety score at a single (x, y). Returns float in [0, 1]."""
    result = _safety_scores(
        np.float64(point_x), np.float64(point_y),
        human_positions, human_predicted_paths, sigma, h, human_traj_pred,
        gamma, sigma_spread, h_traj_scale)
    return float(result)


def compute_safety_grid(human_positions, xlim, ylim, resolution=0.1,
                        human_predicted_paths=None, sigma=SIGMA, h=H,
                        num_cells=None, human_traj_pred=True, gamma=GAMMA,
                        sigma_spread=SIGMA_SPREAD, h_traj_scale=H_TRAJ_SCALE):
    """Vectorized 2D safety field. Returns (grid, extent).

    If *num_cells* is given, both axes use exactly that many cells
    (producing a square grid).  Otherwise falls back to *resolution*.
    """
    if num_cells is not None:
        x = np.linspace(xlim[0], xlim[1], num_cells)
        y = np.linspace(ylim[0], ylim[1], num_cells)
    else:
        x = np.arange(xlim[0], xlim[1], resolution)
        y = np.arange(ylim[0], ylim[1], resolution)
    X, Y = np.meshgrid(x, y)

    safety = _safety_scores(X, Y, human_positions,
                            human_predicted_paths, sigma, h, human_traj_pred,
                            gamma, sigma_spread, h_traj_scale)

    extent = [xlim[0], xlim[1], ylim[0], ylim[1]]
    return safety, extent


def robot_safety_score(robot_x, robot_y, human_positions,
                       human_predicted_paths=None, sigma=SIGMA, h=H,
                       human_traj_pred=True, gamma=GAMMA,
                       sigma_spread=SIGMA_SPREAD, h_traj_scale=H_TRAJ_SCALE):
    """Safety score at the robot's current position. Returns float in [0, 1]."""
    return safety_score_at_point(robot_x, robot_y, human_positions,
                                human_predicted_paths=human_predicted_paths,
                                sigma=sigma, h=h, human_traj_pred=human_traj_pred,
                                gamma=gamma, sigma_spread=sigma_spread,
                                h_traj_scale=h_traj_scale)



