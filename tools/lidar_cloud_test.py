"""Compare point cloud size across lidar topics."""
import struct
import time
import threading
import argparse
import numpy as np
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_

DTYPE_TO_STRUCT = {
    1: 'b', 2: 'B', 3: 'h', 4: 'H',
    5: 'i', 6: 'I', 7: 'f', 8: 'd',
}

TOPICS = [
    'rt/utlidar/cloud',
    'rt/utlidar/cloud_base',
    'rt/utlidar/cloud_deskewed',
]


class TopicMonitor:
    def __init__(self, topic):
        self.topic = topic
        self.lock = threading.Lock()
        self.count = 0
        self.total_pts = 0
        self.last_pts = 0
        self.last_fields = []
        self.last_ring_range = None
        self.start_time = None
        self.active = False

    def callback(self, msg: PointCloud2_):
        n_points = msg.width * msg.height
        fields = [f.name for f in msg.fields]

        # Parse ring field if present
        ring_range = None
        for field in msg.fields:
            if field.name == 'ring':
                fmt = DTYPE_TO_STRUCT[field.datatype]
                dt = np.dtype(fmt)
                data = bytes(msg.data)
                try:
                    values = []
                    for i in range(n_points):
                        offset = i * msg.point_step + field.offset
                        values.append(struct.unpack_from(fmt, data, offset)[0])
                    ring_range = (min(values), max(values))
                except Exception:
                    pass
                break

        with self.lock:
            if self.start_time is None:
                self.start_time = time.time()
            self.count += 1
            self.total_pts += n_points
            self.last_pts = n_points
            self.last_fields = fields
            self.last_ring_range = ring_range
            self.active = True

    def report(self):
        with self.lock:
            if not self.active:
                return f"  {self.topic:40s} -- no messages"
            elapsed = time.time() - self.start_time if self.start_time else 0
            hz = self.count / elapsed if elapsed > 0 else 0
            avg_pts = self.total_pts / self.count if self.count > 0 else 0
            pts_per_sec = self.total_pts / elapsed if elapsed > 0 else 0
            ring_str = f"rings:[{self.last_ring_range[0]}, {self.last_ring_range[1]}]" if self.last_ring_range else "no ring field"
            return (f"  {self.topic:40s} {hz:5.1f} Hz | "
                    f"{self.last_pts:5d} pts/scan | "
                    f"avg={avg_pts:7.0f} | "
                    f"{pts_per_sec:8.0f} pts/sec | "
                    f"{ring_str} | "
                    f"fields={self.last_fields}")


def main():
    parser = argparse.ArgumentParser(description="Compare lidar topic point cloud sizes")
    parser.add_argument('iface', nargs='?', default='enx00e06c79d1cb',
                        help='Network interface')
    args = parser.parse_args()

    ChannelFactoryInitialize(0, args.iface)

    monitors = []
    for topic in TOPICS:
        mon = TopicMonitor(topic)
        sub = ChannelSubscriber(topic, PointCloud2_)
        sub.Init(handler=mon.callback, queueLen=10)
        monitors.append(mon)
        print(f"Subscribed to {topic}")

    print("\nListening... (Ctrl+C to stop)\n")

    try:
        while True:
            time.sleep(2.0)
            print(f"--- {time.strftime('%H:%M:%S')} ---")
            for mon in monitors:
                print(mon.report())
            print()
    except KeyboardInterrupt:
        print("\nDone.")


if __name__ == '__main__':
    main()
