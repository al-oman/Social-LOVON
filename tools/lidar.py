import struct
import time
import threading
import argparse
from collections import deque
import numpy as np
import cv2
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_

DTYPE_TO_STRUCT = {
    1: 'b',
    2: 'B',
    3: 'h',
    4: 'H',
    5: 'i',
    6: 'I',
    7: 'f',
    8: 'd',
}

def pointcloud2_to_array(msg: PointCloud2_):
    data = bytearray(msg.data)
    n_points = msg.width * msg.height
    result = {}
    for field in msg.fields:
        fmt = DTYPE_TO_STRUCT[field.datatype]
        size = struct.calcsize(fmt)
        values = []
        for i in range(n_points):
            offset = i * msg.point_step + field.offset
            values.append(struct.unpack_from(fmt, data, offset)[0])
        result[field.name] = np.array(values)
    return result


# ---------------------------------------------------------------------------
#  Shared state between subscriber callback and render loop
# ---------------------------------------------------------------------------
cloud_lock = threading.Lock()
latest_cloud = None
cloud_count = 0
cloud_freq = 0.0
scan_total = 0        # total points in latest single scan
scan_filtered = 0     # points passing filter in latest single scan
_freq_start = time.time()
_freq_count = 0

ACCUMULATE_N = 10  # number of recent scans to merge for persistence
_cloud_buffer = deque(maxlen=ACCUMULATE_N)


def _count_filtered(cloud):
    """Count how many points in a single scan pass the standard filters."""
    x = cloud.get('x', np.array([]))
    y = cloud.get('y', np.array([]))
    z = cloud.get('z', np.zeros_like(x))
    if len(x) == 0:
        return 0
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    dist_sq = x**2 + y**2 + z**2
    not_origin = dist_sq > (ORIGIN_THRESH ** 2)
    not_far = dist_sq < (DIST_MAX ** 2)
    z_ok = (z >= Z_MIN) & (z <= Z_MAX)
    return int((finite & not_origin & not_far & z_ok).sum())


def on_pointcloud(msg: PointCloud2_):
    global latest_cloud, cloud_count, cloud_freq, _freq_start, _freq_count
    global scan_total, scan_filtered
    try:
        cloud = pointcloud2_to_array(msg)

        # per-scan counters
        n_total = len(next(iter(cloud.values()))) if cloud else 0
        n_filt = _count_filtered(cloud)

        with cloud_lock:
            scan_total = n_total
            scan_filtered = n_filt
            _cloud_buffer.append(cloud)
            # merge all buffered scans into one cloud
            keys = cloud.keys()
            latest_cloud = {k: np.concatenate([c[k] for c in _cloud_buffer])
                            for k in keys}
            cloud_count = len(next(iter(latest_cloud.values())))

        _freq_count += 1
        now = time.time()
        if now - _freq_start >= 1.0:
            cloud_freq = _freq_count / (now - _freq_start)
            _freq_start = now
            _freq_count = 0

    except Exception:
        import traceback
        traceback.print_exc()


# ---------------------------------------------------------------------------
#  Rendering helpers
# ---------------------------------------------------------------------------
WINDOW_NAME = "LiDAR Top-Down View"
CANVAS_SIZE = 800
RANGE_M = 10.0          # initial visible range in meters (half-width)
RANGE_STEP = 2.0        # zoom step in meters
RANGE_MIN = 0.1
RANGE_MAX = 50.0
POINT_RADIUS = 1

# --- Tunable filtering parameters ---
Z_MIN = 0         # meters — drop points below this (ground plane)
Z_MAX = 0.5             # meters — drop points above this (ceiling / noise)
DIST_MAX = 100.0   
      # meters — drop points farther than this
ORIGIN_THRESH = 0.3  # meters — drop points within this radius of (0,0,0)
_first_frame_printed = False

AXES_LIM = 2.0


def world_to_px(x, y, range_m):
    """Convert world (x forward, y left) to pixel coords (center = robot)."""
    scale = (CANVAS_SIZE / 2.0) / range_m
    px = int(CANVAS_SIZE / 2.0 - y * scale)   # y-left  -> px-right
    py = int(CANVAS_SIZE / 2.0 - x * scale)   # x-fwd   -> px-up
    return px, py


def z_to_color(z_vals):
    """Map z-height values to a BGR colormap."""
    if len(z_vals) == 0:
        return np.empty((0, 3), dtype=np.uint8)
    z_min, z_max = z_vals.min(), z_vals.max()
    span = z_max - z_min
    if span < 1e-3:
        span = 1.0
    norm = ((z_vals - z_min) / span * 255).astype(np.uint8)
    colored = cv2.applyColorMap(norm.reshape(-1, 1), cv2.COLORMAP_JET)
    return colored.reshape(-1, 3)


def render(cloud, n_points, freq, range_m, scan_tot=0, scan_filt=0):
    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE, 3), dtype=np.uint8)

    # Grid rings
    scale = (CANVAS_SIZE / 2.0) / range_m
    cx, cy = CANVAS_SIZE // 2, CANVAS_SIZE // 2
    for r_m in np.arange(2.0, range_m + 0.1, 2.0):
        r_px = int(r_m * scale)
        cv2.circle(canvas, (cx, cy), r_px, (40, 40, 40), 1)
        cv2.putText(canvas, f"{r_m:.0f}m", (cx + r_px + 2, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80, 80, 80), 1)

    # Axis lines
    cv2.line(canvas, (cx, 0), (cx, CANVAS_SIZE), (50, 50, 50), 1)
    cv2.line(canvas, (0, cy), (CANVAS_SIZE, cy), (50, 50, 50), 1)

    # Draw points
    n_drawn = 0
    if cloud is not None and 'x' in cloud and 'y' in cloud:
        global _first_frame_printed
        x = cloud['x']
        y = cloud['y']
        z = cloud.get('z', np.zeros_like(x))

        # Print raw data stats once so you can tune the filter params
        if not _first_frame_printed:
            _first_frame_printed = True
            print(f"[DEBUG] raw points: {len(x)}")
            print(f"[DEBUG] x range: [{np.nanmin(x):.2f}, {np.nanmax(x):.2f}]")
            print(f"[DEBUG] y range: [{np.nanmin(y):.2f}, {np.nanmax(y):.2f}]")
            print(f"[DEBUG] z range: [{np.nanmin(z):.2f}, {np.nanmax(z):.2f}]")
            dist = np.sqrt(x**2 + y**2 + z**2)
            print(f"[DEBUG] dist range: [{np.nanmin(dist):.2f}, {np.nanmax(dist):.2f}]")
            print(f"[DEBUG] fields: {list(cloud.keys())}")

        # --- Filter invalid points ---
        finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        dist_sq = x**2 + y**2 + z**2
        not_origin = dist_sq > (ORIGIN_THRESH ** 2)
        not_far = dist_sq < (DIST_MAX ** 2)
        z_ok = (z >= Z_MIN) & (z <= Z_MAX)
        in_view = (np.abs(x) < range_m) & (np.abs(y) < range_m)

        mask = finite & not_origin & not_far & z_ok & in_view
        x, y, z = x[mask], y[mask], z[mask]

        if len(x) > 0:
            colors = z_to_color(z)
            pts_px = np.column_stack([
                (CANVAS_SIZE / 2.0 - y * scale).astype(int),
                (CANVAS_SIZE / 2.0 - x * scale).astype(int),
            ])
            on_canvas = (
                (pts_px[:, 0] >= 0) & (pts_px[:, 0] < CANVAS_SIZE) &
                (pts_px[:, 1] >= 0) & (pts_px[:, 1] < CANVAS_SIZE)
            )
            for px, py, col in zip(pts_px[on_canvas, 0], pts_px[on_canvas, 1], colors[on_canvas]):
                cv2.circle(canvas, (int(px), int(py)), 3,
                           tuple(int(c) for c in col), -1)
            n_drawn = int(on_canvas.sum())

    # Robot marker (green triangle)
    tri = np.array([
        [cx, cy - 8],
        [cx - 5, cy + 4],
        [cx + 5, cy + 4],
    ], dtype=np.int32)
    cv2.fillPoly(canvas, [tri], (0, 255, 0))

    # HUD
    hud_lines = [
        f"Points: {n_drawn} / {n_points} raw",
        f"Last scan: {scan_filt} / {scan_tot} pts",
        f"Freq: {freq:.1f} Hz",
        f"Range: {range_m:.0f} m",
        f"Z filter: [{Z_MIN}, {Z_MAX}]  Dist max: {DIST_MAX}",
        "Controls: +/- zoom, q quit",
    ]
    for i, line in enumerate(hud_lines):
        cv2.putText(canvas, line, (10, 20 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return canvas


def filter_cloud(cloud):
    """Apply standard filters to a point cloud dict. Returns filtered x, y, z."""
    if cloud is None or 'x' not in cloud or 'y' not in cloud:
        return np.array([]), np.array([]), np.array([])
    x = cloud['x']
    y = cloud['y']
    z = cloud.get('z', np.zeros_like(x))

    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    dist_sq = x**2 + y**2 + z**2
    not_origin = dist_sq > (ORIGIN_THRESH ** 2)
    not_far = dist_sq < (DIST_MAX ** 2)
    z_ok = (z >= Z_MIN) & (z <= Z_MAX)
    mask = finite & not_origin & not_far & z_ok

    return x[mask], y[mask], z[mask]


# ---------------------------------------------------------------------------
#  Embeddable LiDAR window (for use from other modules)
# ---------------------------------------------------------------------------
def render_side(cloud, n_points, freq, range_m, z_range=3.0):
    """Render a side view: X (forward) horizontal, Z (up) vertical."""
    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE, 3), dtype=np.uint8)

    x_scale = CANVAS_SIZE / range_m          # full width = range_m forward
    z_scale = CANVAS_SIZE / (2.0 * z_range)  # full height = -z_range..+z_range
    z_center = CANVAS_SIZE // 2              # z=0 at vertical center

    # Grid lines
    for x_m in np.arange(1.0, range_m + 0.1, 1.0):
        px = int(x_m * x_scale)
        cv2.line(canvas, (px, 0), (px, CANVAS_SIZE), (40, 40, 40), 1)
        cv2.putText(canvas, f"{x_m:.0f}m", (px + 2, CANVAS_SIZE - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80, 80, 80), 1)
    for z_m in np.arange(-z_range, z_range + 0.1, 0.5):
        py = int(z_center - z_m * z_scale)
        cv2.line(canvas, (0, py), (CANVAS_SIZE, py), (30, 30, 30), 1)
        if abs(z_m) < 0.01:
            cv2.line(canvas, (0, py), (CANVAS_SIZE, py), (60, 60, 60), 1)
        cv2.putText(canvas, f"{z_m:.1f}m", (3, py - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80, 80, 80), 1)

    n_drawn = 0
    if cloud is not None and 'x' in cloud:
        x = cloud['x']
        y = cloud.get('y', np.zeros_like(x))
        z = cloud.get('z', np.zeros_like(x))

        # Filter
        finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        dist_sq = x**2 + y**2 + z**2
        not_origin = dist_sq > (ORIGIN_THRESH ** 2)
        not_far = dist_sq < (DIST_MAX ** 2)
        z_ok = (z >= Z_MIN) & (z <= Z_MAX)
        mask = finite & not_origin & not_far & z_ok & (x > 0)
        x, y, z = x[mask], y[mask], z[mask]

        if len(x) > 0:
            colors = z_to_color(z)
            px_arr = (x * x_scale).astype(int)
            py_arr = (z_center - z * z_scale).astype(int)

            on_canvas = ((px_arr >= 0) & (px_arr < CANVAS_SIZE) &
                         (py_arr >= 0) & (py_arr < CANVAS_SIZE))
            for px, py, col in zip(px_arr[on_canvas], py_arr[on_canvas], colors[on_canvas]):
                cv2.circle(canvas, (int(px), int(py)), 2,
                           tuple(int(c) for c in col), -1)
            n_drawn = int(on_canvas.sum())

    # Robot marker at origin
    cv2.circle(canvas, (0, z_center), 5, (0, 255, 0), -1)

    # Z_MIN / Z_MAX lines
    zmin_py = int(z_center - Z_MIN * z_scale)
    zmax_py = int(z_center - Z_MAX * z_scale)
    if 0 <= zmin_py < CANVAS_SIZE:
        cv2.line(canvas, (0, zmin_py), (CANVAS_SIZE, zmin_py), (0, 0, 200), 1)
        cv2.putText(canvas, f"Z_MIN={Z_MIN:.2f}", (CANVAS_SIZE - 180, zmin_py - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200), 1)
    if 0 <= zmax_py < CANVAS_SIZE:
        cv2.line(canvas, (0, zmax_py), (CANVAS_SIZE, zmax_py), (200, 0, 0), 1)
        cv2.putText(canvas, f"Z_MAX={Z_MAX:.2f}", (CANVAS_SIZE - 180, zmax_py - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 0, 0), 1)

    # HUD
    hud_lines = [
        f"Side View (X fwd, Z up)  |  Points: {n_drawn} / {n_points}",
        f"Freq: {freq:.1f} Hz  |  Range: {range_m:.0f}m  |  Z: [{Z_MIN}, {Z_MAX}]",
    ]
    for i, line in enumerate(hud_lines):
        cv2.putText(canvas, line, (10, 20 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    return canvas


class LidarWindowSide:
    """OpenCV window showing a side view (X forward, Z up)."""

    def __init__(self, window_name="LiDAR Side View", range_m=RANGE_M, z_range=3.0):
        self._range_m = range_m
        self._z_range = z_range
        self._window_name = window_name
        self._created = False

    def update(self, cloud, n_raw=None):
        if cloud is None:
            cloud = {}
        if n_raw is None:
            n_raw = len(next(iter(cloud.values()))) if cloud else 0

        frame = render_side(cloud, n_raw, 0.0, self._range_m, self._z_range)

        if not self._created:
            cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self._window_name, CANVAS_SIZE, CANVAS_SIZE // 2)
            self._created = True

        cv2.imshow(self._window_name, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('+') or key == ord('='):
            self._range_m = max(RANGE_MIN, self._range_m - RANGE_STEP)
        elif key == ord('-'):
            self._range_m = min(RANGE_MAX, self._range_m + RANGE_STEP)
        return key

    def close(self):
        if self._created:
            cv2.destroyWindow(self._window_name)
            self._created = False


class LidarWindow:
    """OpenCV window that renders a top-down lidar view.

    Usage from another module::

        from tools.lidar import LidarWindow
        lw = LidarWindow()
        # in your loop:
        lw.update(cloud_dict)   # cloud_dict = {"x": ..., "y": ..., "z": ...}
    """

    def __init__(self, window_name="LiDAR Top-Down", range_m=RANGE_M):
        self._range_m = range_m
        self._window_name = window_name
        self._created = False

    def update(self, cloud, n_raw=None):
        """Render one frame.  Returns the key pressed (or -1)."""
        if cloud is None:
            cloud = {}
        if n_raw is None:
            n_raw = len(next(iter(cloud.values()))) if cloud else 0

        frame = render(cloud, n_raw, 0.0, self._range_m)

        if not self._created:
            cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self._window_name, CANVAS_SIZE, CANVAS_SIZE)
            self._created = True

        cv2.imshow(self._window_name, frame)
        key = cv2.waitKey(1) & 0xFF

        # zoom controls
        if key == ord('+') or key == ord('='):
            self._range_m = max(RANGE_MIN, self._range_m - RANGE_STEP)
        elif key == ord('-'):
            self._range_m = min(RANGE_MAX, self._range_m + RANGE_STEP)

        return key

    def close(self):
        if self._created:
            cv2.destroyWindow(self._window_name)
            self._created = False


class LidarWindow3D:
    """Non-blocking matplotlib 3D scatter window fed externally via .update().

    Usage::

        from tools.lidar import LidarWindow3D
        lw = LidarWindow3D()
        # in your loop:
        lw.update(cloud_dict)
    """

    def __init__(self, axes_lim=AXES_LIM):
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider

        self._plt = plt
        self._axes_lim = axes_lim
        self._z_min = Z_MIN
        self._z_max = Z_MAX

        self._fig = plt.figure("LiDAR 3D View", figsize=(8, 6))
        self._fig.subplots_adjust(bottom=0.20)
        self._ax = self._fig.add_subplot(111, projection='3d')

        # Z_MIN slider
        ax_zmin = self._fig.add_axes([0.15, 0.08, 0.65, 0.03])
        self._slider_zmin = Slider(ax_zmin, 'Z Min', -10.0, 10.0,
                                   valinit=self._z_min, valstep=0.05)
        self._slider_zmin.on_changed(self._on_zmin)

        # Z_MAX slider
        ax_zmax = self._fig.add_axes([0.15, 0.03, 0.65, 0.03])
        self._slider_zmax = Slider(ax_zmax, 'Z Max', -10.0, 10.0,
                                   valinit=self._z_max, valstep=0.05)
        self._slider_zmax.on_changed(self._on_zmax)

        self._fig.show()

    def _on_zmin(self, val):
        self._z_min = val

    def _on_zmax(self, val):
        self._z_max = val

    def update(self, cloud):
        """Render one frame with the given cloud dict. Call from main thread."""
        if cloud is None:
            cloud = {}

        x = np.asarray(cloud.get('x', []), dtype=np.float64)
        y = np.asarray(cloud.get('y', []), dtype=np.float64)
        z = np.asarray(cloud.get('z', []), dtype=np.float64)
        n_raw = len(x)

        # filter
        if n_raw > 0:
            finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
            dist_sq = x**2 + y**2 + z**2
            not_origin = dist_sq > (ORIGIN_THRESH ** 2)
            not_far = dist_sq < (DIST_MAX ** 2)
            z_ok = (z >= self._z_min) & (z <= self._z_max)
            mask = finite & not_origin & not_far & z_ok
            x, y, z = x[mask], y[mask], z[mask]

        ax = self._ax
        ax.cla()
        ax.set_xlabel('X (forward)')
        ax.set_ylabel('Y (left)')
        ax.set_zlabel('Z (up)')
        ax.set_title(f"Points: {len(x)} / {n_raw} raw  |  Z:[{self._z_min:.2f}, {self._z_max:.2f}]")

        lim = self._axes_lim
        if len(x) > 0:
            ax.scatter(x, y, z, c=z, cmap='jet', s=1, depthshade=True)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(0, lim)

        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    def close(self):
        self._plt.close(self._fig)


# ---------------------------------------------------------------------------
#  3D matplotlib rendering
# ---------------------------------------------------------------------------
def run_3d_viewer():
    """Live-updating 3D scatter plot of the LiDAR point cloud."""
    global Z_MIN, Z_MAX
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure("LiDAR 3D View", figsize=(10, 8))
    # Make room for sliders at the bottom
    fig.subplots_adjust(bottom=0.20)
    ax = fig.add_subplot(111, projection='3d')

    # Z_MIN slider
    ax_zmin = fig.add_axes([0.15, 0.08, 0.65, 0.03])
    slider_zmin = Slider(ax_zmin, 'Z Min', -10.0, 10.0, valinit=Z_MIN, valstep=0.05)

    # Z_MAX slider
    ax_zmax = fig.add_axes([0.15, 0.03, 0.65, 0.03])
    slider_zmax = Slider(ax_zmax, 'Z Max', -10.0, 10.0, valinit=Z_MAX, valstep=0.05)

    def on_zmin_changed(val):
        global Z_MIN
        Z_MIN = val

    def on_zmax_changed(val):
        global Z_MAX
        Z_MAX = val

    slider_zmin.on_changed(on_zmin_changed)
    slider_zmax.on_changed(on_zmax_changed)

    scatter = [None]  # mutable container so the timer callback can update it

    def update(_frame=None):
        with cloud_lock:
            cloud_snapshot = (
                {k: v.copy() for k, v in latest_cloud.items()}
                if latest_cloud else None
            )
            n = cloud_count
            st, sf = scan_total, scan_filtered
        freq = cloud_freq

        xf, yf, zf = filter_cloud(cloud_snapshot)

        ax.cla()
        ax.set_xlabel('X (forward)')
        ax.set_ylabel('Y (left)')
        ax.set_zlabel('Z (up)')
        ax.set_title(f"Points: {len(xf)} / {n} raw  |  Scan: {sf}/{st}  |  {freq:.1f} Hz  |  Z:[{Z_MIN:.2f}, {Z_MAX:.2f}]")

        if len(xf) > 0:
            ax.scatter(xf, yf, zf, c=zf, cmap='jet', s=1, depthshade=True)
            ax.set_xlim(-AXES_LIM, AXES_LIM)
            ax.set_ylim(-AXES_LIM, AXES_LIM)
            ax.set_zlim(0, AXES_LIM)
        else:
            ax.set_xlim(-AXES_LIM, AXES_LIM)
            ax.set_ylim(-AXES_LIM, AXES_LIM)
            ax.set_zlim(0, AXES_LIM)

        fig.canvas.draw_idle()

    from matplotlib.animation import FuncAnimation
    _anim = FuncAnimation(fig, update, interval=200, cache_frame_data=False)
    plt.show()


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LiDAR point-cloud visualizer")
    parser.add_argument('iface', nargs='?', default='enx00e06c79d1cb',
                        help='Network interface')
    parser.add_argument('--3d', dest='view_3d', action='store_true',
                        help='Show an interactive 3D matplotlib plot instead of the 2D top-down view')
    parser.add_argument('--topic', type=str, default='rt/utlidar/cloud_base',
                    help='Network interface')
    args = parser.parse_args()

    ChannelFactoryInitialize(0, args.iface)

    sub = ChannelSubscriber(args.topic, PointCloud2_)
    sub.Init(handler=on_pointcloud, queueLen=10)

    if args.view_3d:
        print(f"LiDAR 3D visualizer running (interface: {args.iface}). Close the window to quit.")
        run_3d_viewer()
    else:
        range_m = RANGE_M
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, CANVAS_SIZE, CANVAS_SIZE)

        print(f"LiDAR visualizer running (interface: {args.iface}). Press q to quit.")

        while True:
            with cloud_lock:
                cloud_snapshot = {k: v.copy() for k, v in latest_cloud.items()} if latest_cloud else None
                n = cloud_count
                st, sf = scan_total, scan_filtered
            freq = cloud_freq

            frame = render(cloud_snapshot, n, freq, range_m, st, sf)
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(33) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('+') or key == ord('='):
                range_m = max(RANGE_MIN, range_m - RANGE_STEP)
            elif key == ord('-'):
                range_m = min(RANGE_MAX, range_m + RANGE_STEP)

        cv2.destroyAllWindows()
