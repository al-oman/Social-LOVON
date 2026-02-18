#!/usr/bin/env python3
"""DDS connection quality benchmark for Unitree Go2 LiDAR.

Run once per network interface, then compare the reports.

Usage:
    python test/dds_quality.py enp8s0          # 10-second test (default)
    python test/dds_quality.py wlan0 --duration 30
    python test/dds_quality.py enp8s0 --topic rt/utlidar/cloud
"""

import struct
import time
import argparse
import numpy as np
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_

DTYPE_TO_STRUCT = {
    1: 'b', 2: 'B', 3: 'h', 4: 'H',
    5: 'i', 6: 'I', 7: 'f', 8: 'd',
}


def pointcloud2_to_array(msg: PointCloud2_):
    data = bytearray(msg.data)
    n_points = msg.width * msg.height
    result = {}
    for field in msg.fields:
        fmt = DTYPE_TO_STRUCT[field.datatype]
        for i in range(n_points):
            offset = i * msg.point_step + field.offset
            result.setdefault(field.name, []).append(
                struct.unpack_from(fmt, data, offset)[0])
    return {k: np.array(v) for k, v in result.items()}


# ---------------------------------------------------------------------------
#  Collector — records raw timing + size data for every message
# ---------------------------------------------------------------------------
class MessageCollector:
    def __init__(self):
        self.timestamps = []      # wall-clock arrival time
        self.point_counts = []    # points per message
        self.parse_times = []     # seconds to parse each message
        self.byte_sizes = []      # raw payload bytes

    def on_msg(self, msg: PointCloud2_):
        t_arrive = time.perf_counter()
        try:
            t0 = time.perf_counter()
            cloud = pointcloud2_to_array(msg)
            parse_dt = time.perf_counter() - t0

            n_pts = msg.width * msg.height
            raw_bytes = len(msg.data)

            self.timestamps.append(t_arrive)
            self.point_counts.append(n_pts)
            self.parse_times.append(parse_dt)
            self.byte_sizes.append(raw_bytes)
        except Exception as e:
            print(f"  parse error: {e}")


# ---------------------------------------------------------------------------
#  Report
# ---------------------------------------------------------------------------
def print_report(col: MessageCollector, iface: str, topic: str, duration: float):
    n = len(col.timestamps)
    print()
    print("=" * 62)
    print(f"  DDS Quality Report — {iface}")
    print(f"  topic: {topic}")
    print("=" * 62)

    if n < 2:
        print(f"  Only received {n} message(s). Check interface / topic.")
        print("=" * 62)
        return

    ts = np.array(col.timestamps)
    gaps = np.diff(ts) * 1000  # ms

    actual_duration = ts[-1] - ts[0]
    avg_hz = (n - 1) / actual_duration if actual_duration > 0 else 0

    pts = np.array(col.point_counts)
    parse = np.array(col.parse_times) * 1000  # ms
    bsizes = np.array(col.byte_sizes)

    # Detect likely drops: gaps > 2x the median inter-message interval
    median_gap = np.median(gaps)
    big_gaps = gaps[gaps > median_gap * 2]
    estimated_drops = int(np.sum(np.round(gaps[gaps > median_gap * 2] / median_gap) - 1)) if len(big_gaps) > 0 else 0

    print()
    print("  MESSAGE RATE")
    print(f"    messages received : {n}")
    print(f"    test duration     : {actual_duration:.1f}s")
    print(f"    average rate      : {avg_hz:.2f} Hz")
    print()
    print("  INTER-MESSAGE GAP (jitter)")
    print(f"    min    : {gaps.min():.1f} ms")
    print(f"    median : {np.median(gaps):.1f} ms")
    print(f"    mean   : {gaps.mean():.1f} ms")
    print(f"    p95    : {np.percentile(gaps, 95):.1f} ms")
    print(f"    p99    : {np.percentile(gaps, 99):.1f} ms")
    print(f"    max    : {gaps.max():.1f} ms")
    print(f"    stddev : {gaps.std():.1f} ms")
    print()
    print("  ESTIMATED DROPS")
    print(f"    large gaps (>2x median) : {len(big_gaps)}")
    print(f"    estimated missed msgs   : ~{estimated_drops}")
    print(f"    effective loss rate      : ~{estimated_drops / (n + estimated_drops) * 100:.1f}%")
    print()
    print("  POINT CLOUD SIZE")
    print(f"    min    : {pts.min()} pts")
    print(f"    median : {int(np.median(pts))} pts")
    print(f"    mean   : {pts.mean():.0f} pts")
    print(f"    max    : {pts.max()} pts")
    print(f"    stddev : {pts.std():.0f} pts")
    print()
    print("  PAYLOAD")
    print(f"    avg bytes/msg     : {bsizes.mean():.0f}")
    print(f"    avg throughput    : {bsizes.mean() * avg_hz / 1024:.1f} KB/s")
    print()
    print("  PARSE PERFORMANCE")
    print(f"    min    : {parse.min():.1f} ms")
    print(f"    mean   : {parse.mean():.1f} ms")
    print(f"    max    : {parse.max():.1f} ms")
    can_keep_up = parse.mean() < (1000 / avg_hz) if avg_hz > 0 else True
    print(f"    keeps up with sensor : {'YES' if can_keep_up else 'NO — parsing is a bottleneck'}")
    print()
    print("=" * 62)
    print()


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="DDS connection quality benchmark for Unitree LiDAR")
    parser.add_argument('iface', help='Network interface (e.g. enp8s0, wlan0)', default="enx00e06c79d1cb")
    parser.add_argument('--topic', default='rt/utlidar/cloud_base',
                        help='DDS topic to subscribe to (default: rt/utlidar/cloud_base)')
    parser.add_argument('--duration', type=float, default=10.0,
                        help='Test duration in seconds (default: 10)')
    args = parser.parse_args()

    print(f"Initialising DDS on interface '{args.iface}' ...")
    ChannelFactoryInitialize(0, args.iface)

    collector = MessageCollector()

    sub = ChannelSubscriber(args.topic, PointCloud2_)
    sub.Init(handler=collector.on_msg, queueLen=50)

    print(f"Listening on '{args.topic}' for {args.duration:.0f}s ...")
    t_start = time.time()
    try:
        while time.time() - t_start < args.duration:
            elapsed = time.time() - t_start
            n = len(collector.timestamps)
            print(f"\r  {elapsed:.0f}s / {args.duration:.0f}s  |  {n} msgs received", end="", flush=True)
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n  interrupted.")

    print()
    print_report(collector, args.iface, args.topic, args.duration)


if __name__ == "__main__":
    main()
