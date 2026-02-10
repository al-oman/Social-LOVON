import struct
import time
import threading
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
_freq_start = time.time()
_freq_count = 0


def on_pointcloud(msg: PointCloud2_):
    global latest_cloud, cloud_count, cloud_freq, _freq_start, _freq_count
    try:
        cloud = pointcloud2_to_array(msg)
        with cloud_lock:
            latest_cloud = cloud
            cloud_count = msg.width * msg.height

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
RANGE_MIN = 2.0
RANGE_MAX = 50.0
POINT_RADIUS = 1


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


def render(cloud, n_points, freq, range_m):
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
    if cloud is not None and 'x' in cloud and 'y' in cloud:
        x = cloud['x']
        y = cloud['y']
        z = cloud.get('z', np.zeros_like(x))

        # Filter to visible range
        mask = (np.abs(x) < range_m) & (np.abs(y) < range_m)
        x, y, z = x[mask], y[mask], z[mask]

        if len(x) > 0:
            colors = z_to_color(z)
            pts_px = np.column_stack([
                (CANVAS_SIZE / 2.0 - y * scale).astype(int),
                (CANVAS_SIZE / 2.0 - x * scale).astype(int),
            ])
            # Clip to canvas
            valid = (
                (pts_px[:, 0] >= 0) & (pts_px[:, 0] < CANVAS_SIZE) &
                (pts_px[:, 1] >= 0) & (pts_px[:, 1] < CANVAS_SIZE)
            )
            for px, py, col in zip(pts_px[valid, 0], pts_px[valid, 1], colors[valid]):
                canvas[py, px] = col

    # Robot marker (green triangle)
    tri = np.array([
        [cx, cy - 8],
        [cx - 5, cy + 4],
        [cx + 5, cy + 4],
    ], dtype=np.int32)
    cv2.fillPoly(canvas, [tri], (0, 255, 0))

    # HUD
    hud_lines = [
        f"Points: {n_points}",
        f"Freq: {freq:.1f} Hz",
        f"Range: {range_m:.0f} m",
        "Controls: +/- zoom, q quit",
    ]
    for i, line in enumerate(hud_lines):
        cv2.putText(canvas, line, (10, 20 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return canvas


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    iface = sys.argv[1] if len(sys.argv) > 1 else 'enp8s0'
    ChannelFactoryInitialize(0, iface)

    sub = ChannelSubscriber('rt/utlidar/cloud', PointCloud2_)
    sub.Init(handler=on_pointcloud, queueLen=10)

    range_m = RANGE_M
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, CANVAS_SIZE, CANVAS_SIZE)

    print(f"LiDAR visualizer running (interface: {iface}). Press q to quit.")

    while True:
        with cloud_lock:
            cloud_snapshot = {k: v.copy() for k, v in latest_cloud.items()} if latest_cloud else None
            n = cloud_count
        freq = cloud_freq

        frame = render(cloud_snapshot, n, freq, range_m)
        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(33) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('+') or key == ord('='):
            range_m = max(RANGE_MIN, range_m - RANGE_STEP)
        elif key == ord('-'):
            range_m = min(RANGE_MAX, range_m + RANGE_STEP)

    cv2.destroyAllWindows()
