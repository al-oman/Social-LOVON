import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from models.safety import compute_safety_grid

# ── Arc (fake human trajectory) ────────────────────────────────────────────────
ARC_RADIUS       = 3.0    # radius of the circular arc (metres)
ARC_SPAN_DEG     = 60.0   # total angular span of the arc (degrees)
N_POINTS         = 10     # number of waypoints along the arc
DOT_SIZE         = 30     # scatter dot size
PAD_LEFT         = 4    # padding around arc extents (metres)
PAD_RIGHT        = 3.5
PAD_BOTTOM       = 3
PAD_TOP          = 4

# ── Safety field parameters (passed directly to safety.py) ─────────────────────
SIGMA            = 0.75   # Gaussian width at current position (m)
H                = 1.0    # peak danger amplitude (0..1)
GAMMA            = 1.05    # per-step H multiplier along trajectory
SIGMA_SPREAD     = 0.05   # sigma growth per prediction step (m/step)
H_TRAJ_SCALE     = 1.0   # per-step H scale along trajectory

# ── Heatmap parameters ─────────────────────────────────────────────────────────
GRID_RESOLUTION  = 100    # grid cells per axis
COLORMAP         = "viridis"
N_FILL_LEVELS    = 30     # contourf fill levels
N_CONTOURS       = 12     # contour line levels
CONTOUR_COLOR    = "black"
CONTOUR_LINEWIDTH = 0.6

# ── Display ────────────────────────────────────────────────────────────────────
FIGSIZE          = (5, 4)
DPI              = 100   # screen display DPI; PDF saves are vector so this doesn't affect print quality
SAVE_DPI         = 300   # used only for raster saves
SHOW_COLORBAR    = True
SHOW_DOTS        = True
DOT_COLOR        = "white"
DOT_EDGECOLOR    = "gray"
SAVE_PDF         = False
PDF_PATH         = os.path.join(os.path.dirname(__file__), "contour_plot.pdf")

# ── Output ─────────────────────────────────────────────────────────────────────
SAVE_DAT         = False       # write gnuplot-compatible field.dat
DAT_PATH         = os.path.join(os.path.dirname(__file__), "field.dat")
TRAJ_PATH        = os.path.join(os.path.dirname(__file__), "traj.dat")


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

    xmin, xmax = arc_x.min() - PAD_LEFT, arc_x.max() + PAD_RIGHT
    ymin, ymax = arc_y.min() - PAD_BOTTOM, arc_y.max() + PAD_TOP

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

    if SAVE_DAT:
        N = GRID_RESOLUTION
        with open(DAT_PATH, "w") as f:
            f.write("x y z\n")
            for i in range(N):
                for j in range(N):
                    f.write(f"{X[i,j]} {Y[i,j]} {Z[i,j]}\n")
                f.write("\n")
        print(f"Saved {DAT_PATH}")

        with open(TRAJ_PATH, "w") as f:
            f.write("x y\n")
            for x, y in zip(arc_x, arc_y):
                f.write(f"{x} {y}\n")
        print(f"Saved {TRAJ_PATH}")

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    im = ax.contourf(X, Y, Z, levels=N_FILL_LEVELS, cmap=COLORMAP)
    ax.contour(X, Y, Z, levels=N_CONTOURS, colors=CONTOUR_COLOR,
               linewidths=CONTOUR_LINEWIDTH)

    if SHOW_COLORBAR:
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Safety")

    if SHOW_DOTS:
        ax.scatter(arc_x, arc_y, s=DOT_SIZE, c=DOT_COLOR,
                   edgecolors=DOT_EDGECOLOR, zorder=3)

    for idx in [0, N_POINTS // 2, N_POINTS - 1]:
        ax.scatter(arc_x[idx], arc_y[idx], s=DOT_SIZE * 3,
                   c="cyan", edgecolors="black", zorder=4)

    ax.set_xlabel(r"$x$ (m)")
    ax.set_ylabel(r"$y$ (m)")
    ax.set_aspect("equal", "box")
    plt.tight_layout()

    if SAVE_PDF:
        plt.savefig(PDF_PATH, bbox_inches="tight", dpi=SAVE_DPI)
        print(f"Saved {PDF_PATH}")

    plt.show()


if __name__ == "__main__":
    main()
