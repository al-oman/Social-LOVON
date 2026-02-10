import struct
import time
import numpy as np
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
    # print(f"fields: {msg.fields}")
    for field in msg.fields:
        fmt = DTYPE_TO_STRUCT[field.datatype]
        size = struct.calcsize(fmt)
        values = []
        for i in range(n_points):
            offset = i * msg.point_step + field.offset
            values.append(struct.unpack_from(fmt, data, offset)[0])
        result[field.name] = np.array(values)
    return result

def on_pointcloud(msg: PointCloud2_):
    try:
        cloud = pointcloud2_to_array(msg)
        # print(f"cloud is: {cloud}")
        print(f"got {msg.width * msg.height} points, fields: {list(cloud.keys())}")
        if 'x' in cloud:
            print(f"x_range: [{cloud['x'].min():.2f}, {cloud['x'].max():.2f}]")
    except Exception as e:
        import traceback
        traceback.print_exc()

ChannelFactoryInitialize(0, 'enp8s0')

sub = ChannelSubscriber('rt/utlidar/cloud', PointCloud2_)
sub.Init(handler=on_pointcloud, queueLen=10)

while True:
    time.sleep(10.0)