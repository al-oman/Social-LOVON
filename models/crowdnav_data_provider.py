"""
Wraps a CrowdNav simulation and generates synthetic sensor data
(lidar point clouds, YOLO-style bounding boxes, object detections)
so deploy.py can run its full pipeline against simulated humans.
"""

import numpy as np
import math
import configparser
import gym
import cv2

from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.action import ActionXY, ActionRot, ActionXYRot
from crowd_sim.envs.policy.orca import ORCA


class CrowdNavDataProvider:

    HUMAN_WIDTH = 0.5       # meters, for bbox projection
    HUMAN_HEIGHT = 1.7      # meters, for bbox projection
    LIDAR_PTS_PER_HUMAN = 15
    LIDAR_NOISE_SCALE = 0.25 # fraction of human radius used as gaussian std dev

    def __init__(self, env_config_path, policy_config_path,
                 fov_deg=120.0, target_object="handbag"):
        # Camera model uses display dimensions so bounding boxes
        # align with the blank frame shown in the deploy GUI.

        self.image_width = self.DISPLAY_WIDTH
        self.image_height = self.DISPLAY_HEIGHT
        self.fov_deg = fov_deg
        half_fov = math.radians(fov_deg / 2.0)
        self._fx = (self.image_width / 2.0) / math.tan(half_fov)
        self._cx = self.image_width / 2.0
        self._cy = self.image_height / 2.0

        # self.target_object = target_object
        self.target_object = "handbag"

        # --- Read kinematics from policy config ---
        policy_config = configparser.RawConfigParser()
        policy_config.read(policy_config_path)
        if policy_config.has_option("lovon", "kinematics"):
            self.kinematics = policy_config.get("lovon", "kinematics")
        elif policy_config.has_option("action_space", "kinematics"):
            self.kinematics = policy_config.get("action_space", "kinematics")
        else:
            self.kinematics = "unicycle_xyrot"

        # --- CrowdNav environment setup ---
        env_config = configparser.RawConfigParser()
        env_config.read(env_config_path)
        self.env = gym.make('CrowdSim-v0')
        self.env.configure(env_config)

        self.time_step = float(env_config.get('env', 'time_step'))

        # Robot needs a policy for CrowdNav internals even though
        # we supply actions externally via step().
        robot = Robot(env_config, 'robot')
        robot.set_policy(ORCA())
        robot.kinematics = self.kinematics  # override ORCA's default
        self.env.set_robot(robot)
        self.robot = robot

        self.ob = None
        self.done = False

    # ------------------------------------------------------------------
    #  Episode control
    # ------------------------------------------------------------------

    def reset(self, phase='test', test_case=None):
        self.ob = self.env.reset(phase, test_case)
        self.done = False
        return self.ob

    # ------------------------------------------------------------------
    #  Main step — called lock-step from MotionControlThread
    # ------------------------------------------------------------------

    def step(self, motion_vector):
        """Advance CrowdNav one tick and return synthetic sensor data."""
        if self.done:
            return None

        action = self._to_action(motion_vector)
        self.ob, reward, self.done, info = self.env.step(action)

        self_state = self.robot.get_full_state()
        human_states = self.ob

        humans_rf = self._humans_to_robot_frame(self_state, human_states)

        return {
            "lidar": self._generate_synthetic_lidar(humans_rf),
            "pose_state": self._generate_synthetic_pose_state(humans_rf),
            "object_state": self._generate_synthetic_object_state(self_state),
            "done": self.done,
            "info": info,
        }

    # ------------------------------------------------------------------
    #  Action conversion (same logic as LOVONCrowdPolicy._to_action)
    # ------------------------------------------------------------------

    def _to_action(self, motion_vector):
        vx, vy, wz = [float(v) for v in motion_vector]
        if self.kinematics == "unicycle_xyrot":
            return ActionXYRot(vx=vx, vy=vy, wz=wz)
        elif self.kinematics == "holonomic":
            return ActionXY(vx=vx, vy=vy)
        else:  # unicycle
            speed = np.hypot(vx, vy)
            r = wz * self.time_step
            return ActionRot(v=speed, r=r)

    # ------------------------------------------------------------------
    #  Coordinate transforms (same math as LOVONCrowdPolicy)
    # ------------------------------------------------------------------

    def _world_to_robot_frame(self, hx, hy, rx, ry, rtheta):
        """World position → robot-frame [x_lateral, depth]."""
        dx, dy = hx - rx, hy - ry
        cos_t, sin_t = np.cos(-rtheta), np.sin(-rtheta)
        x_rot = dx * cos_t - dy * sin_t
        y_rot = dx * sin_t + dy * cos_t
        return -y_rot, x_rot  # x_lateral (+ = right), depth (+ = forward)

    def _humans_to_robot_frame(self, self_state, human_states):
        humans_rf = []
        for i, h in enumerate(human_states):
            x_lat, depth = self._world_to_robot_frame(
                h.px, h.py, self_state.px, self_state.py, self_state.theta)
            humans_rf.append({
                "track_id": i,
                "x_lateral": x_lat,
                "depth": depth,
                "distance": float(np.hypot(h.px - self_state.px,
                                           h.py - self_state.py)),
                "radius": h.radius,
            })
        return humans_rf

    # ------------------------------------------------------------------
    #  Synthetic lidar
    # ------------------------------------------------------------------

    def _generate_synthetic_lidar(self, humans_rf):
        """Point-cloud clusters at each human position.

        Lidar frame after deploy.py's x-flip: x=forward, y=left, z=up.
        """
        all_x, all_y, all_z = [], [], []

        for h in humans_rf:
            if h["depth"] <= 0:
                continue
            # lidar x = forward = depth,  lidar y = left = -x_lateral
            cx = h["depth"]
            cy = -h["x_lateral"]
            r = h["radius"]
            n = self.LIDAR_PTS_PER_HUMAN

            all_x.append(cx + np.random.normal(0, r * self.LIDAR_NOISE_SCALE, n))
            all_y.append(cy + np.random.normal(0, r * self.LIDAR_NOISE_SCALE, n))
            all_z.append(np.clip(np.random.normal(0.5, 0.25, n), 0.0, 1.0))

        if all_x:
            return {"x": np.concatenate(all_x),
                    "y": np.concatenate(all_y),
                    "z": np.concatenate(all_z)}
        return {"x": np.array([]), "y": np.array([]), "z": np.array([])}

    # ------------------------------------------------------------------
    #  Synthetic YOLO pose_state (bounding boxes only, no keypoints)
    # ------------------------------------------------------------------

    def _generate_synthetic_pose_state(self, humans_rf):
        """Project humans in camera FOV into pixel-space bounding boxes."""
        poses = []
        pose_boxes = []
        half_fov = math.radians(self.fov_deg / 2.0)

        for h in humans_rf:
            if h["depth"] <= 0.1:
                continue
            angle = math.atan2(h["x_lateral"], h["depth"])
            if abs(angle) > half_fov:
                continue

            u = self._fx * (h["x_lateral"] / h["depth"]) + self._cx
            v = self._cy

            w_px = self._fx * self.HUMAN_WIDTH / h["depth"]
            h_px = self._fx * self.HUMAN_HEIGHT / h["depth"]

            x1 = int(max(0, u - w_px / 2))
            x2 = int(min(self.image_width, u + w_px / 2))
            y1 = int(max(0, v - h_px / 2))
            y2 = int(min(self.image_height, v + h_px / 2))

            pose_boxes.append([x1, y1, x2, y2])
            poses.append({
                "keypoints": [[0, 0]] * 17,
                "keypoints_conf": [0.0] * 17,
                "confidence": 0.9,
            })

        return {"num_people": len(poses),
                "poses": poses,
                "pose_boxes": pose_boxes}

    # ------------------------------------------------------------------
    #  Synthetic object detection (goal → L2MM input)
    # ------------------------------------------------------------------

    def _generate_synthetic_object_state(self, self_state):
        """Project the CrowdNav goal into camera-normalised coords for L2MM."""
        dx = self_state.gx - self_state.px
        dy = self_state.gy - self_state.py
        goal_dist = np.hypot(dx, dy)
        goal_angle = np.arctan2(dy, dx) - self_state.theta
        goal_angle = (goal_angle + np.pi) % (2 * np.pi) - np.pi

        half_fov = np.radians(self.fov_deg / 2.0)
        goal_angle = np.clip(goal_angle, -half_fov, half_fov)

        x = 0.5 - np.tan(goal_angle) / (2 * np.tan(half_fov))
        xyn = [float(np.clip(x, 0.0, 1.0)), 0.5]
        whn = [0.4, 0.4] if goal_dist < 0.3 else [0.1, 0.1]

        return {
            "predicted_object": self.target_object,
            "confidence": [0.9],
            "object_xyn": xyn,
            "object_whn": whn,
        }

    # ------------------------------------------------------------------
    #  Blank camera frame for the deploy GUI
    # ------------------------------------------------------------------

    DISPLAY_WIDTH = 1280
    DISPLAY_HEIGHT = 720

    def get_blank_frame(self):
        """Return a black image sized for the deploy GUI."""
        return np.zeros((self.DISPLAY_HEIGHT, self.DISPLAY_WIDTH, 3),
                        dtype=np.uint8)

    # ------------------------------------------------------------------
    #  Live top-down matplotlib view (reuses crowd_test_live pattern)
    # ------------------------------------------------------------------

    def init_render(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib import patches
        self._plt = plt
        self._patches = patches
        self._fig, self._ax = plt.subplots(figsize=(5, 5))
        self._step_num = 0

    def render_frame(self):
        plt = self._plt
        patches = self._patches
        ax = self._ax
        ax.cla()
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_xlabel('x(m)')
        ax.set_ylabel('y(m)')

        cmap = plt.cm.get_cmap('hsv', 10)

        # Humans
        for i, human in enumerate(self.env.humans):
            hc = plt.Circle(human.get_position(), human.radius,
                            fill=False, color=cmap(i))
            ax.add_artist(hc)

        # Robot
        s = self.robot.get_full_state()
        rc = plt.Circle((s.px, s.py), self.robot.radius,
                        fill=True, color='yellow')
        ax.add_artist(rc)
        arrow = patches.FancyArrowPatch(
            (s.px, s.py),
            (s.px + self.robot.radius * np.cos(s.theta),
             s.py + self.robot.radius * np.sin(s.theta)),
            arrowstyle='->', color='red')
        ax.add_artist(arrow)

        # Goal
        ax.plot(s.gx, s.gy, 'r*', markersize=15)

        self._step_num += 1
        ax.set_title(f'Step {self._step_num}  t={self.env.global_time:.2f}s')

        # Render to buffer and display via cv2 (thread-safe)
        self._fig.canvas.draw()
        buf = self._fig.canvas.buffer_rgba()
        img = np.asarray(buf)[:, :, :3][:, :, ::-1].copy()  # RGBA → BGR
        cv2.imshow("CrowdNav", img)
        cv2.waitKey(1)
