import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

import numpy as np
# import pyrealsense2 as rs
import time
import torch
import threading
import queue
import argparse
from ultralytics import YOLO
import struct

# from cxn_010.api_object_extraction_transformer import ObjectExtractionAPI
# from cxn_010.api_language2motion_transformer import MotionPredictor

from models.api_object_extraction import SequenceToSequenceClassAPI
from models.api_language2mostion import MotionPredictor
from models.api_social_navigator import SocialNavigator


import logging
logging.getLogger('ultralytics').setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib.pyplot as plt
from crowd_sim.envs.utils.action import ActionXY

predictor = MotionPredictor(
    model_path="models/model_language2motion_n1000000_d128_h8_l4_f512_msl64_hold_success", 
    tokenizer_path="models/tokenizer_language2motion_n1000000")

# Example input ground truth
sample_input = {'mission_instruction_0': 'approach the bag at 0.5 m/', 
                'mission_instruction_1': 'approach the bag at 0.5 m/', 
                'predicted_object': 'handbag', 
                'confidence': [0.8823556303977966], 
                'object_xyn': [0.5, 0.5], 
                'object_whn': [0.4, 0.4],
                'motion_vector': [0.5, 0, 0], 
                'mission_state_in': 'running', 
                'search_state_in': 'running',
                "mission_state_out":"running",  # these aren't necessary
                "search_state_out":"running"}   #

# initial robot pose in world frame
x, y, theta = 0.0, 0.0, 0.0  # theta in radians

# storage for plotting
trajectory_x = []
trajectory_y = []

states = []

dt = 0.1
time_array = np.arange(0, 10, dt)
input = sample_input
for i in time_array:
    prediction = predictor.predict(sample_input)
    states.append(prediction)
    [v_rx, v_ry, om_r] = prediction["motion_vector"]

    # transform robot-relative velocity to world coordinates
    vx_world = v_rx * np.cos(theta) - v_ry * np.sin(theta)
    vy_world = v_rx * np.sin(theta) + v_ry * np.cos(theta)

    # update pose
    x += vx_world * dt
    y += vy_world * dt
    theta += om_r * dt

    trajectory_x.append(x)
    trajectory_y.append(y)


    input["motion_vector"] = prediction["motion_vector"]
    input["mission_state_in"] = prediction["predicted_state"]
    input["search_state_in"] = prediction["search_state"]

for state in states:
    print(state)

print(input)

# simple plot
plt.figure(figsize=(6,6))
plt.plot(trajectory_x, trajectory_y, '-o', markersize=2)
plt.xlabel("X (world)")
plt.ylabel("Y (world)")
plt.title("Robot Trajectory in World Coordinates")
plt.axis('equal')
plt.grid(True)
plt.show()

