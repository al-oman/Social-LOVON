import numpy as np
import matplotlib.pyplot as plt

# ── Arc parameters ─────────────────────────────────────────────────────────────
ARC_RADIUS       = 1.0       # radius of the circular arc (metres)
ARC_SPAN_DEG     = 90.0     # total angular span of the arc (degrees)
N_POINTS         = 10        # number of dots along the arc
DOT_SIZE         = 30        # scatter dot size

# ── Gaussian parameters ────────────────────────────────────────────────────────
SIGMA_ALONG      = 0.3       # sigma parallel to curve (same for all three)
SIGMA_PERP_FIRST = 0.3       # sigma perpendicular at first point
SIGMA_PERP_MID   = 0.4       # sigma perpendicular at middle point
SIGMA_PERP_LAST  = 0.5       # sigma perpendicular at last point
H_FIRST          = 1.0       # peak amplitude at first point
H_MID            = 1.0       # peak amplitude at middle point
H_LAST           = 1.0       # peak amplitude at last point

# ── Heatmap parameters ─────────────────────────────────────────────────────────
GRID_RESOLUTION  = 100       # grid cells per axis
COLORMAP         = "Blues"     # try "hot", "plasma", "YlOrRd", "RdYlGn", "Blues"
HEATMAP_ALPHA    = 0.85      # opacity of the heatmap underlay
SHOW_CONTOURS    = True      # draw contour lines over the heatmap
N_CONTOURS       = 6        # number of contour levels
CONTOUR_COLOR    = "black"   # contour line colour
CONTOUR_ALPHA    = 0.5       # contour line opacity
CONTOUR_LINEWIDTH = 0.8      # contour line width

# ── Display ────────────────────────────────────────────────────────────────────
FIGSIZE          = (7, 7)
SHOW_COLORBAR    = True
SHOW_DOTS        = True
DOT_COLOR        = "white"
DOT_EDGECOLOR    = "gray"


def make_arc(radius, span_deg, n):
    """Circular arc starting vertical at bottom, curving right."""
    # Circle centred at (radius, 0); start angle=pi (point at origin, tangent up)
    # decreasing angle = clockwise = curving right
    angles = np.linspace(np.pi, np.pi - np.radians(span_deg), n)
    x = radius + radius * np.cos(angles)
    y = radius * np.sin(angles)
    return x, y, angles


def tangent_at(angles, idx):
    """Unit tangent vector in direction of travel (decreasing angle)."""
    a = angles[idx]
    t = np.array([np.sin(a), -np.cos(a)])
    return t / np.linalg.norm(t)


def gaussian_2d(X, Y, cx, cy, tangent, sigma_along, sigma_perp, h):
    """Anisotropic 2D Gaussian aligned to tangent/normal axes."""
    nx = np.array([-tangent[1], tangent[0]])
    dx = X - cx
    dy = Y - cy
    d_along = dx * tangent[0] + dy * tangent[1]
    d_perp  = dx * nx[0]      + dy * nx[1]
    exponent = (d_along ** 2 / (2 * sigma_along ** 2) +
                d_perp  ** 2 / (2 * sigma_perp  ** 2))
    return h * np.exp(-exponent)


def main():
    arc_x, arc_y, angles = make_arc(ARC_RADIUS, ARC_SPAN_DEG, N_POINTS)

    key_points = [
        (0,              SIGMA_PERP_FIRST, H_FIRST),
        (N_POINTS // 2,  SIGMA_PERP_MID,   H_MID),
        (N_POINTS - 1,   SIGMA_PERP_LAST,  H_LAST),
    ]

    pad = 1.5
    xmin, xmax = arc_x.min() - pad, arc_x.max() + pad
    ymin, ymax = arc_y.min() - pad, arc_y.max() + pad
    gx = np.linspace(xmin, xmax, GRID_RESOLUTION)
    gy = np.linspace(ymin, ymax, GRID_RESOLUTION)
    X, Y = np.meshgrid(gx, gy)

    Z = np.ones_like(X)
    for idx, sigma_perp, h in key_points:
        tang = tangent_at(angles, idx)
        g = gaussian_2d(X, Y, arc_x[idx], arc_y[idx],
                        tang, SIGMA_ALONG, sigma_perp, h)
        np.minimum(Z, 1.0 - g, out=Z)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    im = ax.imshow(Z, origin="lower", extent=[xmin, xmax, ymin, ymax],
                   cmap=COLORMAP, alpha=HEATMAP_ALPHA, aspect="equal")

    if SHOW_CONTOURS:
        ax.contour(X, Y, Z, levels=N_CONTOURS,
                   colors=CONTOUR_COLOR, alpha=CONTOUR_ALPHA,
                   linewidths=CONTOUR_LINEWIDTH)

    if SHOW_COLORBAR:
        fig.colorbar(im, ax=ax, label="Gaussian intensity")

    if SHOW_DOTS:
        ax.scatter(arc_x, arc_y, s=DOT_SIZE, c=DOT_COLOR,
                   edgecolors=DOT_EDGECOLOR, zorder=3)

    for idx, *_ in key_points:
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
