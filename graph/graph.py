import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from models.safety import compute_safety_grid

# ── Arc (fake human trajectory) ────────────────────────────────────────────────
ARC_RADIUS       = 3.0    # radius of the circular arc (metres)
ARC_SPAN_DEG     = 90.0   # total angular span of the arc (degrees)
N_POINTS         = 10     # number of waypoints along the arc
DOT_SIZE         = 30     # scatter dot size
PAD              = 1.5    # padding around arc extents (metres)

# ── Safety field parameters (passed directly to safety.py) ─────────────────────
SIGMA            = 0.75   # Gaussian width at current position (m)
H                = 1.0    # peak danger amplitude (0..1)
GAMMA            = 1.01    # per-step H multiplier along trajectory
SIGMA_SPREAD     = 0.1   # sigma growth per prediction step (m/step)
H_TRAJ_SCALE     = 1.0   # per-step H scale along trajectory

# ── Heatmap parameters ─────────────────────────────────────────────────────────
GRID_RESOLUTION  = 250    # grid cells per axis
COLORMAP         = "RdYlGn"  # try "hot", "plasma", "YlOrRd", "RdYlGn", "Blues"
HEATMAP_ALPHA    = 0.85   # opacity of the heatmap underlay
SHOW_CONTOURS    = True   # draw contour lines over the heatmap
N_CONTOURS       = 6      # number of contour levels
CONTOUR_COLOR    = "black"
CONTOUR_ALPHA    = 0.5
CONTOUR_LINEWIDTH = 0.8

# ── Display ────────────────────────────────────────────────────────────────────
FIGSIZE          = (7, 7)
SHOW_COLORBAR    = True
SHOW_DOTS        = True
DOT_COLOR        = "white"
DOT_EDGECOLOR    = "gray"


def make_arc(radius, span_deg, n):
    """Circular arc starting vertical at bottom, curving right."""
    angles = np.linspace(np.pi, np.pi - np.radians(span_deg), n)
    x = radius + radius * np.cos(angles)
    y = radius * np.sin(angles)
    return x, y


def main():
    arc_x, arc_y = make_arc(ARC_RADIUS, ARC_SPAN_DEG, N_POINTS)

    human_positions = [[arc_x[0], arc_y[0]]]
    human_predicted_paths = {0: [[arc_x[i], arc_y[i]] for i in range(1, N_POINTS)]}

    xmin, xmax = arc_x.min() - PAD, arc_x.max() + PAD
    ymin, ymax = arc_y.min() - PAD, arc_y.max() + PAD

    Z, extent = compute_safety_grid(
        human_positions=human_positions,
        xlim=(xmin, xmax),
        ylim=(ymin, ymax),
        num_cells=GRID_RESOLUTION,
        human_predicted_paths=human_predicted_paths,
        sigma=SIGMA,
        h=H,
        gamma=GAMMA,
        sigma_spread=SIGMA_SPREAD,
        h_traj_scale=H_TRAJ_SCALE,
    )

    gx = np.linspace(xmin, xmax, GRID_RESOLUTION)
    gy = np.linspace(ymin, ymax, GRID_RESOLUTION)
    X, Y = np.meshgrid(gx, gy)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    im = ax.imshow(Z, origin="lower", extent=extent,
                   cmap=COLORMAP, alpha=HEATMAP_ALPHA, aspect="equal")

    if SHOW_CONTOURS:
        ax.contour(X, Y, Z, levels=N_CONTOURS,
                   colors=CONTOUR_COLOR, alpha=CONTOUR_ALPHA,
                   linewidths=CONTOUR_LINEWIDTH)

    if SHOW_COLORBAR:
        fig.colorbar(im, ax=ax, label="Safety score")

    if SHOW_DOTS:
        ax.scatter(arc_x, arc_y, s=DOT_SIZE, c=DOT_COLOR,
                   edgecolors=DOT_EDGECOLOR, zorder=3)

    for idx in [0, N_POINTS // 2, N_POINTS - 1]:
        ax.scatter(arc_x[idx], arc_y[idx], s=DOT_SIZE * 3,
                   c="cyan", edgecolors="black", zorder=4)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Trajectory Gaussian Safety Field")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
