import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from models.safety import compute_safety_grid

# region: Parameters
# ── Arc (fake human trajectory) ────────────────────────────────────────────────
ARC_RADIUS       = 3.0    # radius of the circular arc (metres)
HUMAN_SPEED      = 1.0    # m/s — scales how far along the arc predictions reach
HORIZON_S        = 5.0    # prediction horizon in seconds
POINT_SPACING    = 0.4    # metres between consecutive prediction points
# n_points is derived: arc_length / POINT_SPACING → fewer points at lower speed
# at HUMAN_SPEED=0 only the current position is shown (no predictions)
DOT_SIZE         = 10     # scatter dot size
X_MIN            = -3.0   # absolute axis extents (metres)
X_MAX            =  6.0
Y_MIN            = -2.5
Y_MAX            =  7.0

# ── Safety field parameters (passed directly to safety.py) ─────────────────────
SIGMA            = 0.75   # Gaussian width at current position (m)
H                = 1.0    # peak danger amplitude (0..1)
GAMMA            = 1.05    # per-step H multiplier along trajectory
SIGMA_SPREAD     = 0.05   # sigma growth per prediction step (m/step)
H_TRAJ_SCALE     = 1.0   # per-step H scale along trajectory

# ── Heatmap parameters ─────────────────────────────────────────────────────────
GRID_RESOLUTION  = 500    # grid cells per axis
COLORMAP         = "viridis"#"plasma"
N_FILL_LEVELS    = 50     # contourf fill levels
N_CONTOURS       = 10    # contour line levels
CONTOUR_COLOR    = "black"
CONTOUR_LINEWIDTH = 1.0

# ── Display ────────────────────────────────────────────────────────────────────
FONT_SIZE        = 24      # IEEE body text is 9-10pt
FONT_FAMILY      = "serif"
FONT_SERIF       = "Times New Roman"
FIGSIZE          = (9, 6)
DPI              = 100   # screen display DPI; PDF saves are vector so this doesn't affect print quality
SAVE_DPI         = 300   # used only for raster saves
SHOW_COLORBAR    = True
SHOW_DOTS        = True
DOT_COLOR        = "white"
DOT_EDGECOLOR    = "gray"
SAVE_PDF         = True
PDF_PATH         = os.path.join(os.path.dirname(__file__), "3D_fast.pdf")
PLOT_3D          = True  # True for 3D surface plot, False for 2D contourf
VIEW_ELEV        = 30     # 3D camera elevation angle (degrees)
VIEW_AZIM        = -45    # 3D camera azimuth angle (degrees)
SURFACE_ALPHA    = 1.0    # surface opacity (0.0 = transparent, 1.0 = opaque)
MARGIN_3D        = 0.0   # figure margin for 3D plots (fraction); increase if labels clip
LABEL_PAD        = 15    # distance between axis label and tick marks (points)
SAVE_PAD         = 0.5   # extra padding (inches) around saved PDF; increase if labels clip

# ── Output ─────────────────────────────────────────────────────────────────────
SAVE_DAT         = False       # write gnuplot-compatible field.dat
DAT_PATH         = os.path.join(os.path.dirname(__file__), "field.dat")
TRAJ_PATH        = os.path.join(os.path.dirname(__file__), "traj.dat")
# endregion

def make_arc(radius, span_deg, n):
    """Circular arc starting vertical at bottom, curving right."""
    angles = np.linspace(np.pi, np.pi - np.radians(span_deg), n)
    x = radius + radius * np.cos(angles)
    y = radius * np.sin(angles)
    return x, y


def main(show=True):
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": FONT_FAMILY,
        "font.serif": [FONT_SERIF],
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE,
    })

    arc_length = HUMAN_SPEED * HORIZON_S
    if arc_length <= 0:
        arc_x, arc_y = make_arc(ARC_RADIUS, 0.0, 1)
        n_pts = 1
    else:
        arc_span_deg = np.degrees(arc_length / ARC_RADIUS)
        n_pts = max(2, round(arc_length / POINT_SPACING) + 1)  # +1 includes current pos
        arc_x, arc_y = make_arc(ARC_RADIUS, arc_span_deg, n_pts)

    human_positions = [[arc_x[0], arc_y[0]]]
    human_predicted_paths = {0: [[arc_x[i], arc_y[i]] for i in range(1, n_pts)]}

    xmin, xmax = X_MIN, X_MAX
    ymin, ymax = Y_MIN, Y_MAX

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

    # Z = 1-Z

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

    if PLOT_3D:
        fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
        ax = fig.add_subplot(projection="3d")
        ax.plot_surface(X, Y, Z, cmap=COLORMAP, linewidth=0.0, antialiased=True, alpha=SURFACE_ALPHA)
        # ax.contour(X, Y, Z, N_CONTOURS, zdir='z', offset=Z.min(),
                #    colors=CONTOUR_COLOR, linewidths=CONTOUR_LINEWIDTH)
        if SHOW_COLORBAR:
            mappable = plt.cm.ScalarMappable(cmap=COLORMAP)
            mappable.set_array(Z)
            # fig.colorbar(mappable, ax=ax, shrink=0.5).set_label(r"$\Psi$")
        if SHOW_DOTS:
            dot_z = [Z[int(np.clip(round((py - gy[0]) / (gy[-1] - gy[0]) * (GRID_RESOLUTION - 1)), 0, GRID_RESOLUTION - 1)),
                       int(np.clip(round((px - gx[0]) / (gx[-1] - gx[0]) * (GRID_RESOLUTION - 1)), 0, GRID_RESOLUTION - 1))]
                     for px, py in zip(arc_x, arc_y)]
            ax.scatter(arc_x, arc_y, dot_z, s=DOT_SIZE, c=DOT_COLOR,
                       edgecolors=DOT_EDGECOLOR, zorder=3)
        ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
        ax.set_xlabel(r"$y$ (m)", labelpad=LABEL_PAD)
        ax.set_ylabel(r"$x$ (m)", labelpad=LABEL_PAD)
        ax.set_zlabel(r"$\Psi$", labelpad=LABEL_PAD)
    else:
        fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
        im = ax.contourf(X, Y, Z, levels=N_FILL_LEVELS, cmap=COLORMAP)
        ax.contour(X, Y, Z, levels=N_CONTOURS, colors=CONTOUR_COLOR,
                   linewidths=CONTOUR_LINEWIDTH)
        if SHOW_COLORBAR:
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label(r"$\psi$")
        if SHOW_DOTS:
            ax.scatter(arc_x, arc_y, s=DOT_SIZE, c=DOT_COLOR,
                       edgecolors=DOT_EDGECOLOR, zorder=3)
        ax.set_xlabel(r"$y$ (m)", labelpad=LABEL_PAD)
        ax.set_ylabel(r"$x$ (m)", labelpad=LABEL_PAD)
        ax.set_aspect("equal", "box")

    if PLOT_3D:
        plt.subplots_adjust(left=MARGIN_3D, right=1 - MARGIN_3D,
                            bottom=MARGIN_3D, top=1 - MARGIN_3D)
    else:
        plt.tight_layout()

    if SAVE_PDF:
        fig.canvas.draw()
        plt.savefig(PDF_PATH, bbox_inches="tight", pad_inches=SAVE_PAD, dpi=SAVE_DPI)
        print(f"Saved {PDF_PATH}")

    if show:
        plt.show()
        if PLOT_3D:
            print(f"VIEW_ELEV = {ax.elev:.1f}")
            print(f"VIEW_AZIM = {ax.azim:.1f}")


def batch():
    """Save all four variants: 2D/3D × still/fast."""
    import sys
    m = sys.modules[__name__]
    script_dir = os.path.dirname(os.path.abspath(__file__))

    CONFIGS = [
        {
            "name": "2D_still",
            "PLOT_3D": False, "HUMAN_SPEED": 0.0,
            "X_MIN": -3.0, "X_MAX": 3.0, "Y_MIN": -3.0, "Y_MAX": 3.0,
        },
        {
            "name": "2D_fast",
            "PLOT_3D": False, "HUMAN_SPEED": 1.0,
            "X_MIN": -3.0, "X_MAX": 6.0, "Y_MIN": -2.5, "Y_MAX": 7.0,
        },
        {
            "name": "3D_still",
            "PLOT_3D": True, "HUMAN_SPEED": 0.0,
            "X_MIN": -3.0, "X_MAX": 3.0, "Y_MIN": -3.0, "Y_MAX": 3.0,
        },
        {
            "name": "3D_fast",
            "PLOT_3D": True, "HUMAN_SPEED": 1.0,
            "X_MIN": -3.0, "X_MAX": 6.0, "Y_MIN": -2.5, "Y_MAX": 7.0,
        },
    ]

    for cfg in CONFIGS:
        name = cfg["name"]
        overrides = {k: v for k, v in cfg.items() if k != "name"}
        overrides["SAVE_PDF"] = True
        overrides["PDF_PATH"] = os.path.join(script_dir, f"{name}.pdf")

        originals = {k: getattr(m, k) for k in overrides}
        for k, v in overrides.items():
            setattr(m, k, v)

        print(f"Rendering {name} ...")
        main(show=False)
        plt.close("all")

        for k, v in originals.items():
            setattr(m, k, v)

        print(f"  → {name}.pdf")


def graph():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Load data
    df = pd.read_csv("graph/field.dat", sep=r"\s+")

    # Pivot to grid
    pivot = df.pivot(index="y", columns="x", values="z")

    X = pivot.columns.values
    Y = pivot.index.values
    X, Y = np.meshgrid(X, Y)
    Z = pivot.values

    # Plot
    # Z_SCALE = 0.5  # parameter for scaling the z-axis

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_zlim(0, Z.max())
    ax.view_init(elev=20, azim=45)

    surf = ax.plot_surface(X, Y, Z, cmap="viridis")

    fig.colorbar(surf, ax=ax, shrink=0.6, label="z")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("3D Surface Plot")

    plt.show()

if __name__ == "__main__":
    # main()
    batch()
    # graph()
