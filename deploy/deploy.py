"""
Unified deploy script — GUI and headless modes.

Usage:
    # GUI mode (original behavior):
    python deploy/deploy.py --crowdnav_sim_mode --socialnav_enabled

    # Headless batch evaluation:
    python deploy/deploy.py --headless --crowdnav_sim_mode --socialnav_enabled --num_episodes 100
"""

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

import numpy as np
import time
import math
import torch
import threading
import queue
import argparse
import struct
import csv
import datetime
import re
import tempfile

import logging
logging.getLogger('ultralytics').setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ── Lazy / conditional imports ────────────────────────────────────────
# Heavy GUI / camera imports are deferred so that headless mode works
# on machines without displays or unitree SDK.

DTYPE_TO_STRUCT = {
    1: 'b', 2: 'B', 3: 'h', 4: 'H',
    5: 'i', 6: 'I', 7: 'f', 8: 'd',
}


def _import_gui_deps():
    """Import GUI-only dependencies (tkinter, PIL, cv2, YOLO, unitree)."""
    global Tk, Entry, Button, Label, Frame
    global Image, ImageTk
    global cv2, YOLO
    global ChannelFactoryInitialize, ChannelSubscriber, PointCloud2_
    global Go2VideoClient, Go2SportClient, H1SportClient
    global B2SportClient, B2FrontVideoClient, B2BackVideoClient
    global LidarWindowSide

    from tkinter import Tk, Entry, Button, Label, Frame
    from PIL import Image, ImageTk
    import cv2 as _cv2
    globals()['cv2'] = _cv2
    from ultralytics import YOLO as _YOLO
    globals()['YOLO'] = _YOLO
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize as _CFI, ChannelSubscriber as _CS
    from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_ as _PC2
    globals()['ChannelFactoryInitialize'] = _CFI
    globals()['ChannelSubscriber'] = _CS
    globals()['PointCloud2_'] = _PC2
    from unitree_sdk2py.go2.video.video_client import VideoClient as _GVC
    from unitree_sdk2py.go2.sport.sport_client import SportClient as _GSC
    from unitree_sdk2py.h1.loco.h1_loco_client import LocoClient as _HLC
    from unitree_sdk2py.b2.sport.sport_client import SportClient as _BSC
    from unitree_sdk2py.b2.front_video.front_video_client import FrontVideoClient as _BFC
    from unitree_sdk2py.b2.back_video.back_video_client import BackVideoClient as _BBC
    globals()['Go2VideoClient'] = _GVC
    globals()['Go2SportClient'] = _GSC
    globals()['H1SportClient'] = _HLC
    globals()['B2SportClient'] = _BSC
    globals()['B2FrontVideoClient'] = _BFC
    globals()['B2BackVideoClient'] = _BBC
    from tools.lidar import LidarWindowSide as _LWS
    globals()['LidarWindowSide'] = _LWS


def _import_headless_deps():
    """Minimal imports for headless mode."""
    import cv2 as _cv2
    globals()['cv2'] = _cv2


# Always-needed model imports
from models.api_object_extraction import SequenceToSequenceClassAPI
from models.api_language2mostion import MotionPredictor
from models.api_social_navigator import SocialNavigator

# CrowdNav policy choices (non-VLA)
CROWDNAV_POLICIES = ("orca", "sarl", "lstm_rl", "cadrl")


# ═══════════════════════════════════════════════════════════════════════
#  Utility
# ═══════════════════════════════════════════════════════════════════════

def pointcloud2_to_array(msg):
    """Parse a PointCloud2_ message into a dict of numpy arrays."""
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


# ═══════════════════════════════════════════════════════════════════════
#  Shared CrowdNav policy logic (used by both HeadlessRunner and GUI)
# ═══════════════════════════════════════════════════════════════════════

class CrowdNavPolicyMixin:
    """Methods for loading, validating, and stepping CrowdNav policies.

    Expects the consumer to provide:
        self.device, self.robot_policy_name, self.crowdnav_policy,
        self.crowdnav_provider, self.JointState
    """

    def _validate_policy(self):
        """Verify the requested policy was actually loaded and is functional."""
        name = self.robot_policy_name

        if name == "vla":
            if self.motion_predictor is None:
                raise RuntimeError("robot_policy='vla' but MotionPredictor failed to load")
            print(f"[PolicyCheck] VLA policy active  "
                  f"(object_extractor={self.object_extractor is not None}, "
                  f"motion_predictor={self.motion_predictor is not None})")
            return

        # Non-VLA: crowdnav policy must exist
        if self.crowdnav_policy is None:
            raise RuntimeError(
                f"robot_policy='{name}' but crowdnav_policy is None — "
                f"policy failed to load silently")

        # Verify the loaded policy class matches what was requested
        actual_name = getattr(self.crowdnav_policy, 'name', None)
        if actual_name and actual_name.lower() != name.lower():
            raise RuntimeError(
                f"Requested policy '{name}' but got '{actual_name}' — "
                f"check policy_factory registration")

        # Verify trainable policies have loaded weights (model params are non-zero)
        if self.crowdnav_policy.trainable:
            model = self.crowdnav_policy.get_model()
            total_params = sum(p.numel() for p in model.parameters())
            nonzero_params = sum((p != 0).sum().item() for p in model.parameters())
            if nonzero_params == 0:
                raise RuntimeError(
                    f"Policy '{name}' model has {total_params} params but ALL are zero — "
                    f"weights likely failed to load")
            print(f"[PolicyCheck] {name} policy active  "
                  f"trainable=True  params={total_params}  nonzero={nonzero_params}  "
                  f"kinematics={getattr(self.crowdnav_policy, 'kinematics', '?')}")
        else:
            print(f"[PolicyCheck] {name} policy active  "
                  f"trainable=False  "
                  f"kinematics={getattr(self.crowdnav_policy, 'kinematics', '?')}")

        # Warn about kinematics mismatch with environment
        policy_kin = getattr(self.crowdnav_policy, 'kinematics', None)
        env_kin = self.crowdnav_provider.kinematics
        if policy_kin and policy_kin != env_kin:
            print(f"[PolicyCheck] WARNING: policy kinematics '{policy_kin}' "
                  f"overrode env kinematics '{env_kin}'")

    def _load_crowdnav_policy(self, name, args):
        """Instantiate and configure a CrowdNav policy by name."""
        import configparser
        from crowd_nav.policy.policy_factory import policy_factory

        policy = policy_factory[name]()

        cfg_path = getattr(args, "crowdnav_policy_config", None) or args.policy_config
        policy_config = configparser.RawConfigParser()
        policy_config.read(cfg_path)
        policy.configure(policy_config)

        if hasattr(policy, 'time_step') and policy.time_step is None:
            policy.time_step = self.crowdnav_provider.time_step

        if policy.trainable:
            model_path = getattr(args, "crowdnav_model_path", None)
            if model_path is None:
                raise ValueError(
                    f"--crowdnav_model_path is required for trainable policy '{name}'")
            policy.set_device(self.device)
            policy.set_phase("test")
            policy.get_model().load_state_dict(torch.load(model_path, map_location=self.device))
            policy.get_model().eval()
            if getattr(policy, 'query_env', False):
                policy.set_env(self.crowdnav_provider.env)

        policy_kin = getattr(policy, 'kinematics', None)
        if policy_kin and policy_kin != self.crowdnav_provider.kinematics:
            print(f"  Overriding robot kinematics: "
                  f"{self.crowdnav_provider.kinematics} → {policy_kin}")
            self.crowdnav_provider.robot.kinematics = policy_kin
            self.crowdnav_provider.kinematics = policy_kin

        print(f"Loaded CrowdNav policy: {name}  trainable={policy.trainable}  "
              f"kinematics={getattr(policy, 'kinematics', '?')}")
        return policy

    def _crowdnav_policy_action(self):
        """Get action from CrowdNav policy -> body-frame [vx, vy, wz]."""
        robot = self.crowdnav_provider.robot
        self_state = robot.get_full_state()
        human_states = self.crowdnav_provider.ob
        if human_states is None:
            self._last_crowdnav_action = None
            return [0.0, 0.0, 0.0]

        joint_state = self.JointState(self_state, human_states)
        action = self.crowdnav_policy.predict(joint_state)
        self._last_crowdnav_action = action

        from crowd_sim.envs.utils.action import ActionXY, ActionRot, ActionXYRot
        if isinstance(action, ActionXYRot):
            return [action.vx, action.vy, action.wz]
        elif isinstance(action, ActionXY):
            cos_t = np.cos(-self_state.theta)
            sin_t = np.sin(-self_state.theta)
            vx_body = action.vx * cos_t - action.vy * sin_t
            vy_body = action.vx * sin_t + action.vy * cos_t
            return [float(vx_body), float(vy_body), 0.0]
        elif isinstance(action, ActionRot):
            wz = action.r / self.crowdnav_provider.time_step if self.crowdnav_provider.time_step > 0 else 0.0
            return [float(action.v), 0.0, float(wz)]
        else:
            print(f"[PolicyCheck] WARNING: unknown action type {type(action).__name__} "
                  f"from policy '{self.robot_policy_name}' — returning zero motion")
            return [0.0, 0.0, 0.0]

    def _crowdnav_step(self, motion_vector):
        """Step the CrowdNav env using a body-frame motion_vector."""
        from crowd_sim.envs.utils.action import ActionXY, ActionRot, ActionXYRot

        env = self.crowdnav_provider.env
        robot = self.crowdnav_provider.robot
        self_state = robot.get_full_state()
        kin = getattr(self.crowdnav_policy, 'kinematics', 'holonomic')

        vx_b, vy_b, wz = motion_vector
        if kin == "holonomic":
            cos_t = np.cos(self_state.theta)
            sin_t = np.sin(self_state.theta)
            vx_w = vx_b * cos_t - vy_b * sin_t
            vy_w = vx_b * sin_t + vy_b * cos_t
            action = ActionXY(float(vx_w), float(vy_w))
        elif kin == "unicycle":
            r = wz * self.crowdnav_provider.time_step
            action = ActionRot(float(np.hypot(vx_b, vy_b)), float(r))
        else:
            action = ActionXYRot(float(vx_b), float(vy_b), float(wz))

        ob, reward, done, info = env.step(action)
        self.crowdnav_provider.ob = ob
        self.crowdnav_provider.done = done

        human_states = ob
        self.crowdnav_provider._robot_trajectory.append(
            (self_state.px, self_state.py))
        humans_rf = self.crowdnav_provider._humans_to_robot_frame(
            robot.get_full_state(), human_states)

        return {
            "lidar": self.crowdnav_provider._generate_synthetic_lidar(humans_rf),
            "pose_state": self.crowdnav_provider._generate_synthetic_pose_state(humans_rf),
            "object_state": self.crowdnav_provider._generate_synthetic_object_state(
                robot.get_full_state()),
            "done": done,
            "info": info,
        }


# ═══════════════════════════════════════════════════════════════════════
#  HEADLESS RUNNER  (no threads, no GUI, synchronous tight loop)
# ═══════════════════════════════════════════════════════════════════════

class HeadlessRunner(CrowdNavPolicyMixin):
    """Runs CrowdNav episodes as fast as possible without any GUI."""

    def __init__(self, args):
        from crowd_sim.envs.utils.info import Collision, Danger, ReachGoal, Timeout
        from crowd_sim.envs.utils.state import JointState
        self.Collision = Collision
        self.Danger = Danger
        self.ReachGoal = ReachGoal
        self.Timeout = Timeout
        self.JointState = JointState

        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.robot_policy_name = getattr(args, "robot_policy", "vla")

        # ── Models (only load VLA when needed) ──
        if self.robot_policy_name == "vla":
            self.object_extractor = SequenceToSequenceClassAPI(
                model_path=args.object_extraction_model_path,
                tokenizer_path=args.tokenizer_path,
            )
            self.motion_predictor = MotionPredictor(
                model_path=args.language2motion_model_path,
                tokenizer_path=args.tokenizer_path,
            )
            self.mission_instruction_0 = args.mission_instruction
            self.mission_instruction_1 = args.mission_instruction
            self.extracted_object = self.object_extractor.predict(self.mission_instruction_1)
        else:
            self.object_extractor = None
            self.motion_predictor = None
            self.mission_instruction_0 = args.mission_instruction
            self.mission_instruction_1 = args.mission_instruction
            self.extracted_object = "handbag"

        # ── State dicts (same keys the GUI controller uses) ──
        self.state = {
            "predicted_object": "NULL",
            "confidence": [0.00],
            "object_xyn": [0.00, 0.00],
            "object_whn": [0.00, 0.00],
            "mission_state_in": "success",
            "search_state_in": "had_searching_1",
            "bounding_box": None,
        }
        self.pose_state = {"num_people": 0, "poses": [], "pose_boxes": []}
        self.motion_vector = [0.0, 0.0, 0.0]
        self._success_frame_count = 0

        # ── CrowdNav provider ──
        from models.crowdnav_data_provider import CrowdNavDataProvider
        self.crowdnav_provider = CrowdNavDataProvider(
            env_config_path=args.env_config,
            policy_config_path=args.policy_config,
            target_object=self.extracted_object,
        )

        # ── CrowdNav robot policy (non-VLA) ──
        self.crowdnav_policy = None
        if self.robot_policy_name in CROWDNAV_POLICIES:
            self.crowdnav_policy = self._load_crowdnav_policy(
                self.robot_policy_name, args)

        # ── Validate policy actually loaded ──
        self._validate_policy()

        # ── Social Navigator ──
        sn_kwargs = {
            "image_width": self.crowdnav_provider.image_width,
            "show_bezier_pts": False,
            "use_lidar_depth": True,
            "time_step": self.crowdnav_provider.time_step,
            "image_height": self.crowdnav_provider.image_height,
            "fov_deg": self.crowdnav_provider.fov_deg,
            "fov_v_deg": self.crowdnav_provider.fov_deg,
            "lidar_cam_z_offset": 0.0,
            "lidar_cam_pitch_offset": 0.0,
            "lidar_cam_yaw_offset": 0.0,
            "mono_k": self.crowdnav_provider._fx * 0.3,
            "human_traj_pred": not args.disable_human_traj_pred,
        }
        for key in ("safety_sigma", "safety_h", "safety_gamma",
                    "safety_sigma_spread", "safety_h_traj_scale",
                    "shield_thresh_on", "shield_thresh_off",
                    "vx_sfm_gain", "human_pred_s",
                    "traj_gradient_gain", "traj_goal_gain", "traj_step_size",
                    "vx_min"):
            val = getattr(args, key, None)
            if val is not None:
                sn_kwargs[key] = val
        if args.traj_direct_step:
            sn_kwargs["traj_normalize_step"] = False
        self.social_nav = SocialNavigator(
            enabled=args.socialnav_enabled, **sn_kwargs
        )

    # ------------------------------------------------------------------ #
    #  Single episode
    # ------------------------------------------------------------------ #

    def _run_episode(self, episode_idx):
        """Run one CrowdNav episode. Returns a result dict."""
        self.crowdnav_provider.reset(robot_theta=self.args.robot_theta)
        self.motion_vector = [0.0, 0.0, 0.0]
        self.state["mission_state_in"] = "running"

        self.social_nav._predictor.reset()
        self.social_nav._tracked_humans.clear()
        self.social_nav._ego_velocity = None
        self.social_nav._frame_count = 0

        step_count = 0
        max_steps = self.args.max_steps
        t0 = time.perf_counter()

        collision = False
        min_distance_episode = float('inf')
        danger_count = 0
        near_miss_count = 0
        in_near_miss = False
        near_miss_thresh = 0.2  # meters (edge-to-edge)
        termination_reason = "max_steps"
        policy_times = []

        while step_count < max_steps:
            if self.crowdnav_policy is not None:
                synthetic = self._crowdnav_step(self.motion_vector)
            else:
                synthetic = self.crowdnav_provider.step(self.motion_vector)
            if synthetic is None:
                break
            step_count += 1

            info = synthetic["info"]
            if isinstance(info, self.Collision):
                collision = True
                termination_reason = "collision"
            elif isinstance(info, self.Danger):
                danger_count += 1
            elif isinstance(info, self.ReachGoal):
                termination_reason = "goal"
            elif isinstance(info, self.Timeout):
                termination_reason = "timeout"

            robot_state = self.crowdnav_provider.robot.get_full_state()
            humans = self.crowdnav_provider.env.humans
            if humans:
                dmin_step = min(
                    np.hypot(h.px - robot_state.px, h.py - robot_state.py)
                    - h.radius - robot_state.radius
                    for h in humans
                )
                min_distance_episode = min(min_distance_episode, dmin_step)

                # Near-miss: entered danger zone without collision
                if dmin_step < near_miss_thresh and dmin_step > 0:
                    if not in_near_miss:
                        near_miss_count += 1
                        in_near_miss = True
                else:
                    in_near_miss = False

            self.pose_state = synthetic["pose_state"]
            self.state.update(synthetic["object_state"])

            state_copy = {**self.state}
            _t = time.perf_counter()
            self._update_motion_control(state_copy, lidar_cloud=synthetic["lidar"])
            policy_times.append(time.perf_counter() - _t)

            if synthetic["done"]:
                break

        elapsed = time.perf_counter() - t0
        robot = self.crowdnav_provider.robot.get_full_state()
        goal_dist = np.hypot(robot.gx - robot.px, robot.gy - robot.py)
        reached = goal_dist < self.crowdnav_provider.robot.radius + 0.1

        return {
            "episode": episode_idx,
            "steps": step_count,
            "time_s": elapsed,
            "sim_time": self.crowdnav_provider.env.global_time,
            "reached_goal": reached,
            "final_goal_dist": goal_dist,
            "final_px": robot.px,
            "final_py": robot.py,
            "collision": collision,
            "min_distance": min_distance_episode,
            "danger_count": danger_count,
            "near_miss_count": near_miss_count,
            "termination_reason": termination_reason,
            "avg_policy_ms": 1000 * np.mean(policy_times) if policy_times else 0.0,
        }

    # ------------------------------------------------------------------ #
    #  Motion control (same logic as VisualLanguageController)
    # ------------------------------------------------------------------ #

    def _update_motion_control(self, state, lidar_cloud=None):
        if self.crowdnav_policy is not None:
            self.motion_vector = self._crowdnav_policy_action()
            self.state["mission_state_in"] = "running"
        else:
            input_data = {
                "mission_instruction_0": self.mission_instruction_0,
                "mission_instruction_1": self.mission_instruction_1,
                **state,
            }
            prediction = self.motion_predictor.predict(input_data)
            self.state["search_state_in"] = prediction["search_state"]

            if prediction["predicted_state"] == "success":
                self._success_frame_count += 1
            else:
                self._success_frame_count = 0

            if self._success_frame_count >= 3:
                self.state["mission_state_in"] = "success"
                self.motion_vector = [0.0, 0.0, 0.0]
            else:
                self.state["mission_state_in"] = prediction["predicted_state"]
                self.motion_vector = prediction["motion_vector"]

        bbox = self.state.get("bounding_box")
        bbox_h = (bbox[3] - bbox[1]) if bbox else None
        self.social_nav.update_goal(self.state["object_xyn"], bbox_h)

        self.motion_vector = self.social_nav.step(
            motion_vector=self.motion_vector,
            pose_state=self.pose_state,
            mission_state=self.state["mission_state_in"],
            lidar_ranges=lidar_cloud,
        )

    # ------------------------------------------------------------------ #
    #  Batch run
    # ------------------------------------------------------------------ #

    def run(self):
        num_episodes = self.args.num_episodes
        results = []

        print(f"Running {num_episodes} episodes (headless) ...")
        batch_t0 = time.perf_counter()

        for ep in range(num_episodes):
            res = self._run_episode(ep)
            results.append(res)
            status = "GOAL" if res["reached_goal"] else ("COLL" if res["collision"] else "FAIL")
            reason = f"  reason={res['termination_reason']}" if status != "GOAL" and self.args.verbose_failures else ""
            print(
                f"  [{ep+1}/{num_episodes}] {status}  "
                f"steps={res['steps']}  sim_t={res['sim_time']:.2f}s  "
                f"wall={res['time_s']:.3f}s  goal_dist={res['final_goal_dist']:.3f}  "
                f"dmin={res['min_distance']:.3f}  danger={res['danger_count']}{reason}"
            )

        batch_elapsed = time.perf_counter() - batch_t0

        # ── Summary ──
        goals = sum(1 for r in results if r["reached_goal"])
        collisions = sum(1 for r in results if r["collision"])
        avg_min_dist = np.mean([r["min_distance"] for r in results]) if results else 0
        avg_danger = np.mean([r["danger_count"] for r in results]) if results else 0
        avg_near_miss = np.mean([r["near_miss_count"] for r in results]) if results else 0
        print(f"\n{'='*60}")
        print(f"  Episodes:   {num_episodes}")
        print(f"  Success:    {goals}/{num_episodes}  ({100*goals/max(num_episodes,1):.1f}%)")
        print(f"  Collisions: {collisions}/{num_episodes}  ({100*collisions/max(num_episodes,1):.1f}%)")
        print(f"  Avg min distance: {avg_min_dist:.3f} m")
        print(f"  Avg danger count: {avg_danger:.1f} steps/episode")
        print(f"  Avg near misses:  {avg_near_miss:.1f} events/episode")
        print(f"  Wall time: {batch_elapsed:.2f}s  "
              f"({batch_elapsed/max(num_episodes,1):.3f}s / episode)")
        avg_policy_ms = np.mean([r["avg_policy_ms"] for r in results]) if results else 0.0
        print(f"  Avg policy time: {avg_policy_ms:.1f} ms/step")
        print(f"{'='*60}")

        # ── Append batch summary row to CSV ──
        csv_path = self.args.csv_path
        avg_steps = np.mean([r["steps"] for r in results]) if results else 0
        avg_sim_time = np.mean([r["sim_time"] for r in results]) if results else 0
        avg_goal_dist = np.mean([r["final_goal_dist"] for r in results]) if results else 0

        env_cfg = self.crowdnav_provider.env.config
        print("── env config ─────────────────────────────────────────")
        for section in env_cfg.sections():
            for key, val in env_cfg.items(section):
                print(f"  [{section}] {key:<20s} = {val}")
        print(f"{'='*60}")

        # Record the effective social-nav params (defaults or CLI overrides)
        sn_params = self.social_nav.params if self.social_nav.enabled else {}
        batch_row = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "num_episodes": num_episodes,
            "robot_policy": self.robot_policy_name,
            "socialnav_enabled": self.args.socialnav_enabled,
            "human_traj_pred": not self.args.disable_human_traj_pred,
            "robot_theta": self.args.robot_theta,
            "human_num": env_cfg.getint("sim", "human_num"),
            "human_v_pref": env_cfg.getfloat("humans", "v_pref"),
            "human_policy": env_cfg.get("humans", "policy"),
            "mission_instruction": self.args.mission_instruction,
            "safety_sigma": sn_params.get("safety_sigma", ""),
            "safety_h": sn_params.get("safety_h", ""),
            "safety_gamma": sn_params.get("safety_gamma", ""),
            "safety_sigma_spread": sn_params.get("safety_sigma_spread", ""),
            "safety_h_traj_scale": sn_params.get("safety_h_traj_scale", ""),
            "shield_thresh_on": sn_params.get("shield_thresh_on", ""),
            "shield_thresh_off": sn_params.get("shield_thresh_off", ""),
            "vx_sfm_gain": sn_params.get("vx_sfm_gain", ""),
            "human_pred_s": sn_params.get("human_pred_s", ""),
            "traj_gradient_gain": sn_params.get("traj_gradient_gain", ""),
            "traj_goal_gain": sn_params.get("traj_goal_gain", ""),
            "traj_step_size": sn_params.get("traj_step_size", ""),
            "vx_min": sn_params.get("vx_min", ""),
            "success_rate": goals / max(num_episodes, 1),
            "collision_rate": collisions / max(num_episodes, 1),
            "avg_min_distance": avg_min_dist,
            "avg_danger_count": avg_danger,
            "avg_near_miss": avg_near_miss,
            "avg_steps": avg_steps,
            "avg_sim_time": avg_sim_time,
            "avg_goal_dist": avg_goal_dist,
            "wall_time": batch_elapsed,
            "avg_policy_ms": avg_policy_ms,
        }
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=batch_row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(batch_row)
        print(f"Batch summary appended to {csv_path}")


# ═══════════════════════════════════════════════════════════════════════
#  GUI THREAD CLASSES  (only instantiated when --headless is NOT used)
# ═══════════════════════════════════════════════════════════════════════

class LiDARGetterThread(threading.Thread):
    """LiDAR Point Cloud Acquisition Thread

    Accumulates the last ``accumulate_n`` scans so that ``get_cloud()``
    returns a denser merged point cloud instead of a single sparse frame.
    """

    ACCUMULATE_N = 5  # number of recent scans to merge

    def __init__(self, controller):
        super().__init__()
        from collections import deque
        self.controller = controller
        self.running = True
        self.lidar_lock = threading.Lock()
        self.latest_cloud = None
        self._cloud_buffer = deque(maxlen=self.ACCUMULATE_N)
        self.freq_start = time.time()
        self.freq_count = 0
        # polling-rate diagnostics
        self._last_cb_time = None
        self._gap_max = 0.0
        self._parse_total = 0.0
        self._parse_count = 0
        # full refresh timing
        self._refresh_scan_count = 0
        self._refresh_last_time = None
        self.full_refresh_dt = 0.0

    def run(self):
        self._sub = ChannelSubscriber('rt/utlidar/cloud_base', PointCloud2_)
        self._sub.Init(handler=self._on_pointcloud, queueLen=10)
        while self.running:
            time.sleep(0.5)

    def _on_pointcloud(self, msg):
        try:
            now = time.time()
            if self._last_cb_time is not None:
                gap = now - self._last_cb_time
                if gap > self._gap_max:
                    self._gap_max = gap
            self._last_cb_time = now

            t0 = time.perf_counter()
            cloud = pointcloud2_to_array(msg)
            parse_ms = (time.perf_counter() - t0) * 1000
            self._parse_total += parse_ms
            self._parse_count += 1

            with self.lidar_lock:
                self.latest_cloud = cloud
                self._cloud_buffer.append(cloud)

            # Track full buffer refresh time
            self._refresh_scan_count += 1
            if self._refresh_scan_count >= self.ACCUMULATE_N:
                if self._refresh_last_time is not None:
                    self.full_refresh_dt = now - self._refresh_last_time
                self._refresh_last_time = now
                self._refresh_scan_count = 0

            self.freq_count += 1
            if now - self.freq_start >= 1.0:
                freq = self.freq_count / (now - self.freq_start)
                fields = list(cloud.keys())
                n_pts = len(next(iter(cloud.values()))) if cloud else 0
                avg_parse = self._parse_total / max(self._parse_count, 1)
                accum_pts = sum(len(next(iter(c.values()))) for c in self._cloud_buffer)
                diag = (f"[LiDARGetter] {freq:.1f} Hz | {n_pts} pts/scan | "
                        f"accum={accum_pts} pts ({len(self._cloud_buffer)} scans) | "
                        f"parse={avg_parse:.1f}ms | gap_max={self._gap_max*1000:.0f}ms | "
                        f"full_refresh={self.full_refresh_dt*1000:.0f}ms")
                for k, v in cloud.items():
                    if len(v) > 0:
                        diag += f" | {k}:[{v.min():.2f}, {v.max():.2f}]"
                print(diag)
                if self._gap_max > 0.5:
                    print(f"[LiDARGetter] WARNING: max inter-message gap {self._gap_max*1000:.0f}ms — "
                          "messages may be dropping. Check network or queueLen.")
                if avg_parse > 50:
                    print(f"[LiDARGetter] WARNING: avg parse time {avg_parse:.0f}ms is high — "
                          "consider optimising pointcloud2_to_array.")
                with self.controller.freq_lock:
                    self.controller.lidar_getter_freq = freq
                self.freq_start = now
                self.freq_count = 0
                self._gap_max = 0.0
                self._parse_total = 0.0
                self._parse_count = 0

            from tools.lidar import full_refresh_dt                                       
            print(f"Full lidar refresh: {full_refresh_dt:.3f}s") 
            
        except Exception as e:
            print(f"LiDARGetter Error: {e}")

    def get_cloud(self):
        """Return the accumulated (merged) point cloud from recent scans."""
        with self.lidar_lock:
            if not self._cloud_buffer:
                return None
            keys = self._cloud_buffer[0].keys()
            merged = {k: np.concatenate([c[k] for c in self._cloud_buffer])
                      for k in keys}
            return merged

    def stop(self):
        self.running = False
        self.join()


class ImageGetterThread(threading.Thread):
    """Image Acquisition Thread"""

    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.running = True
        self.image_queue = queue.Queue(maxsize=1)
        self.pose_image_queue = queue.Queue(maxsize=1)
        self.freq_start = time.time()
        self.freq_count = 0

    def run(self):
        while self.running:
            try:
                if self.controller.simulation_mode:
                    self.controller._update_image_from_webcam()
                elif self.controller.camera_type == "inner":
                    self.controller._update_image_from_video_client()
                elif self.controller.camera_type == "realsense":
                    self.controller._update_image_from_realsense()

                with self.controller.image_lock:
                    if hasattr(self.controller, 'image') and self.controller.image is not None:
                        current_image = self.controller.image.copy()
                        if not self.image_queue.empty():
                            try:
                                self.image_queue.get_nowait()
                            except queue.Empty:
                                pass
                        laplacian_var, is_blur = self.detect_blur(
                            current_image, threshold=self.controller.blur_threshold
                        )
                        if not is_blur:
                            self.image_queue.put(current_image)
                            if not self.pose_image_queue.empty():
                                try:
                                    self.pose_image_queue.get_nowait()
                                except queue.Empty:
                                    pass
                            self.pose_image_queue.put(current_image.copy())
                            self.last_image = current_image
                        else:
                            if hasattr(self, 'last_image'):
                                self.image_queue.put(self.last_image)
                                if not self.pose_image_queue.empty():
                                    try:
                                        self.pose_image_queue.get_nowait()
                                    except queue.Empty:
                                        pass
                                self.pose_image_queue.put(self.last_image.copy())
                            else:
                                print("No clear image available to use as fallback.")

                self.freq_count += 1
                if time.time() - self.freq_start >= 1:
                    freq = self.freq_count / (time.time() - self.freq_start)
                    print(f"[ImageGetter] Frequency: {freq:.2f} Hz")
                    with self.controller.freq_lock:
                        self.controller.image_getter_freq = freq
                    self.freq_start = time.time()
                    self.freq_count = 0

            except Exception as e:
                print(f"ImageGetter Error: {e}")
                time.sleep(0.1)

    @staticmethod
    def detect_blur(image, threshold=100.0):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        is_blur = laplacian_var < threshold
        return laplacian_var, is_blur

    def stop(self):
        self.running = False
        self.join()


class YoloProcessingThread(threading.Thread):
    """YOLO Object Detection Processing Thread"""

    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.running = True
        self.image_queue = controller.image_getter_thread.image_queue
        self.result_queue = queue.Queue()
        self.freq_start = time.time()
        self.freq_count = 0

    def run(self):
        while self.running:
            try:
                image = self.image_queue.get(timeout=1)
                with self.controller.yolo_lock:
                    results = self.controller.yolo_model(image, verbose=False)
                    self.controller._yolo_image_post_process(results, image)

                # Override goal detection with ArUco if enabled
                if self.controller.aruco_detector is not None:
                    aruco_result = self.controller.aruco_detector.detect(image)
                    self.controller.state["predicted_object"] = aruco_result["predicted_object"]
                    self.controller.state["confidence"] = aruco_result["confidence"]
                    self.controller.state["object_xyn"] = aruco_result["object_xyn"]
                    self.controller.state["object_whn"] = aruco_result["object_whn"]
                    self.controller.state["bounding_box"] = aruco_result["bounding_box"]
                    self.controller.state["goal_depth"] = aruco_result["goal_depth"]

                self.result_queue.put(self.controller.state.copy())

                self.freq_count += 1
                if time.time() - self.freq_start >= 1:
                    freq = self.freq_count / (time.time() - self.freq_start)
                    print(f"[YoloProcessor] Frequency: {freq:.2f} Hz")
                    with self.controller.freq_lock:
                        self.controller.yolo_processor_freq = freq
                    self.freq_start = time.time()
                    self.freq_count = 0

            except queue.Empty:
                continue
            except Exception as e:
                print(f"YoloProcessing Error: {e}")
                time.sleep(0.1)

    def stop(self):
        self.running = False
        self.join()


class YoloPoseProcessingThread(threading.Thread):
    """YOLO Pose Detection Processing Thread"""

    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.running = True
        self.image_queue = controller.image_getter_thread.pose_image_queue
        self.result_queue = queue.Queue()
        self.freq_start = time.time()
        self.freq_count = 0

    def run(self):
        while self.running:
            try:
                image = self.image_queue.get(timeout=1)
                with self.controller.yolo_pose_lock:
                    results = self.controller.yolo_pose_model(image, verbose=False)
                    self.controller._yolo_pose_post_process(results, image)

                self.result_queue.put(self.controller.pose_state.copy())

                self.freq_count += 1
                if time.time() - self.freq_start >= 1:
                    freq = self.freq_count / (time.time() - self.freq_start)
                    print(f"[YoloPoseProcessor] Frequency: {freq:.2f} Hz")
                    with self.controller.freq_lock:
                        self.controller.yolo_pose_processor_freq = freq
                    self.freq_start = time.time()
                    self.freq_count = 0

            except queue.Empty:
                continue
            except Exception as e:
                print(f"YoloPoseProcessing Error: {e}")
                time.sleep(0.1)

    def stop(self):
        self.running = False
        self.join()


class MotionControlThread(threading.Thread):
    """Robot Motion Control Thread"""

    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.running = True
        if not controller.crowdnav_sim_mode:
            self.result_queue = controller.yolo_processing_thread.result_queue
        self.freq_start = time.time()
        self.freq_count = 0

    def run(self):
        while self.running:
            try:
                if self.controller.crowdnav_sim_mode:
                    tick_start = time.perf_counter()
                    self._crowdnav_tick()
                    elapsed = time.perf_counter() - tick_start
                    remaining = self.controller.crowdnav_provider.time_step - elapsed
                    if remaining > 0:
                        time.sleep(remaining)
                else:
                    try:
                        state = self.result_queue.get(timeout=0.05)  # 20Hz tick
                        with self.controller.motion_lock:
                            self.controller._update_motion_control(state)
                    except queue.Empty:
                        pass  # No new detection — keep sending last velocity
                    with self.controller.motion_lock:
                        self.controller._control_robot()

                self.freq_count += 1
                if time.time() - self.freq_start >= 1:
                    freq = self.freq_count / (time.time() - self.freq_start)
                    print(f"[MotionControl] Frequency: {freq:.2f} Hz")
                    with self.controller.freq_lock:
                        self.controller.motion_control_freq = freq
                    self.freq_start = time.time()
                    self.freq_count = 0

            except Exception as e:
                import traceback
                print(f"MotionControl Error: {e}")
                traceback.print_exc()
                time.sleep(0.1)

    def _crowdnav_tick(self):
        c = self.controller
        if not getattr(c, 'sim_started', False):
            time.sleep(0.05)
            return

        if getattr(c, 'sim_paused', False):
            time.sleep(0.05)
            return

        with c.motion_lock:
            mv = c.motion_vector if hasattr(c, 'motion_vector') else [0.0, 0.0, 0.0]
            if getattr(c, 'crowdnav_policy', None) is not None:
                synthetic = c._crowdnav_step(mv)
            else:
                synthetic = c.crowdnav_provider.step(mv)
            if synthetic is None or synthetic.get("done", False):
                if synthetic is not None and getattr(c.args, 'verbose_failures', False):
                    info = synthetic.get("info")
                    from crowd_sim.envs.utils.info import Collision, ReachGoal, Timeout
                    if isinstance(info, Collision):
                        reason = "collision"
                    elif isinstance(info, ReachGoal):
                        reason = "goal"
                    elif isinstance(info, Timeout):
                        reason = "timeout"
                    else:
                        reason = "unknown"
                    robot = c.crowdnav_provider.robot.get_full_state()
                    humans = c.crowdnav_provider.env.humans
                    if humans:
                        dists = [
                            np.hypot(h.px - robot.px, h.py - robot.py) - h.radius - robot.radius
                            for h in humans
                        ]
                        dmin = min(dists)
                        closest_idx = dists.index(dmin)
                        h = humans[closest_idx]
                        print(f"CrowdNav episode finished: {reason}  "
                              f"dmin={dmin:.3f}m (human {closest_idx} at "
                              f"({h.px:.2f},{h.py:.2f}), robot at ({robot.px:.2f},{robot.py:.2f}))")
                    else:
                        print(f"CrowdNav episode finished: {reason}")
                else:
                    print("CrowdNav episode finished.")
                c.sim_started = False
                return

            c.pose_state = synthetic["pose_state"]
            c.state.update(synthetic["object_state"])

            frame = c.crowdnav_provider.get_blank_frame()
            img_q = c.image_getter_thread.image_queue
            if not img_q.empty():
                try:
                    img_q.get_nowait()
                except queue.Empty:
                    pass
            img_q.put(frame)

            c.crowdnav_sim_frame = c.crowdnav_provider.render_frame()

            state = {**c.state}
            c._update_motion_control(state, lidar_cloud=synthetic["lidar"])

            if c._planned_trajectory_world is None and c.social_nav._goal_rf is not None:
                robot_state = c.crowdnav_provider.robot.get_full_state()
                path_rf = c.social_nav._extrapolate_robot_trajectory(c.motion_vector)
                if path_rf:
                    cos_t = math.cos(robot_state.theta)
                    sin_t = math.sin(robot_state.theta)
                    c._planned_trajectory_world = []
                    for pt in path_rf:
                        x_lat, depth = pt[0], pt[1]
                        wx = robot_state.px + depth * cos_t + x_lat * sin_t
                        wy = robot_state.py + depth * sin_t - x_lat * cos_t
                        c._planned_trajectory_world.append((wx, wy))
                    c.crowdnav_provider.planned_trajectory = c._planned_trajectory_world

    def stop(self):
        self.running = False
        self.join()


# ═══════════════════════════════════════════════════════════════════════
#  GUI CONTROLLER  (original deploy.py VisualLanguageController)
# ═══════════════════════════════════════════════════════════════════════

class VisualLanguageController(CrowdNavPolicyMixin):
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.robot_policy_name = getattr(args, 'robot_policy', 'vla')
        self.crowdnav_sim_mode = args.crowdnav_sim_mode

        # ── Models: only load what the chosen policy needs ──
        self.object_extractor = SequenceToSequenceClassAPI(
            model_path=args.object_extraction_model_path,
            tokenizer_path=args.tokenizer_path
        )
        if not self.crowdnav_sim_mode:
            self.yolo_model = YOLO(args.yolo_model_dir)
            self.yolo_pose_model = YOLO(args.yolo_pose_model_dir)
        if self.robot_policy_name == "vla":
            self.motion_predictor = MotionPredictor(
                model_path=args.language2motion_model_path,
                tokenizer_path=args.tokenizer_path
            )
        else:
            self.motion_predictor = None

        # JointState needed by CrowdNavPolicyMixin
        from crowd_sim.envs.utils.state import JointState
        self.JointState = JointState

        self.show_video = args.show_video
        self.show_max_result = args.show_max_result
        self.show_arrowed = args.show_arrowed
        self.camera_type = args.camera_type
        self.robot_type = args.robot_type
        self.blur_threshold = args.threshold
        self.lengthen_filter = args.lengthen_filter
        self.simulation_mode = args.simulation_mode
        self.socialnav_enabled = args.socialnav_enabled
        self.button_update_inst = False
        self.manual_stop = False
        self.network_device = args.network_device
        self.robot_theta = args.robot_theta
        self.human_traj_pred = not args.disable_human_traj_pred

        # Non-VLA policies require crowdnav_sim_mode
        if self.robot_policy_name in CROWDNAV_POLICIES and not self.crowdnav_sim_mode:
            raise ValueError(
                f"--robot_policy '{self.robot_policy_name}' requires --crowdnav_sim_mode in GUI mode.")

        # Initialize Unitree SDK components
        if not self.simulation_mode and not self.crowdnav_sim_mode:
            self._init_channel_factory()
            self.video_client = self._init_camera()
            self.sport_client = self._init_sport()

        # Initialize mission instructions and state
        self.mission_instruction_0 = "run to the person at speed of 0.36 m/s"
        self.mission_instruction_1 = self.mission_instruction_0
        self.state = {
            "predicted_object": "NULL",
            "confidence": [0.00],
            "object_xyn": [0.00, 0.00],
            "object_whn": [0.00, 0.00],
            "mission_state_in": "success",
            "search_state_in": "had_searching_1",
            "bounding_box": None,
        }
        self.extracted_object = self.object_extractor.predict(self.mission_instruction_1)

        self.pose_state = {
            "num_people": 0,
            "poses": [],
            "pose_boxes": [],
        }

        # Thread locks
        self.image_lock = threading.Lock()
        self.yolo_lock = threading.Lock()
        self.yolo_pose_lock = threading.Lock()
        self.motion_lock = threading.Lock()
        self.freq_lock = threading.Lock()
        self._lidar_estop_count = 0
        self._success_frame_count = 0

        # Frequency monitoring
        self.image_getter_freq = 0.0
        self.yolo_processor_freq = 0.0
        self.yolo_pose_processor_freq = 0.0
        self.motion_control_freq = 0.0
        self.lidar_getter_freq = 0.0

        # CrowdNav sim provider
        if self.crowdnav_sim_mode:
            from models.crowdnav_data_provider import CrowdNavDataProvider
            self.crowdnav_provider = CrowdNavDataProvider(
                env_config_path=args.env_config,
                policy_config_path=args.policy_config,
                target_object=self.extracted_object,
            )
            self.crowdnav_provider.reset(robot_theta=self.robot_theta)
            self.crowdnav_provider.init_render()
            self.motion_vector = [0.0, 0.0, 0.0]
            self._lidar_estop_count = 0
            self._success_frame_count = 0
            self.sim_started = False
            self.sim_paused = False
            self._planned_trajectory_world = None
            self.crowdnav_sim_frame = self.crowdnav_provider.render_frame()

        # ── CrowdNav robot policy (non-VLA) ──
        self.crowdnav_policy = None
        if self.robot_policy_name in CROWDNAV_POLICIES:
            self.crowdnav_policy = self._load_crowdnav_policy(
                self.robot_policy_name, args)

        # ── Validate policy actually loaded ──
        self._validate_policy()

        # Worker threads
        self.image_getter_thread = ImageGetterThread(self)
        if not self.crowdnav_sim_mode:
            self.yolo_processing_thread = YoloProcessingThread(self)
            self.yolo_pose_processing_thread = YoloPoseProcessingThread(self)
        self.motion_control_thread = MotionControlThread(self)
        if not self.simulation_mode and not self.crowdnav_sim_mode:
            self.lidar_getter_thread = LiDARGetterThread(self)
        else:
            self.lidar_getter_thread = None

        # LiDAR proximity e-stop constants
        self.LIDAR_ESTOP_DISTANCE = 0.35  # metres
        self.LIDAR_ESTOP_Z_MIN = -0.2
        self.LIDAR_ESTOP_Z_MAX = 0.8

        # Social navigation
        sn_width = self.crowdnav_provider.image_width if self.crowdnav_sim_mode else args.image_width
        sn_kwargs = {"image_width": sn_width,
                     "show_bezier_pts": args.show_bezier_pts}
        if self.crowdnav_sim_mode:
            sn_kwargs["use_lidar_depth"] = True
            sn_kwargs["time_step"] = self.crowdnav_provider.time_step
            sn_kwargs["image_height"] = self.crowdnav_provider.image_height
            sn_kwargs["fov_deg"] = self.crowdnav_provider.fov_deg
            sn_kwargs["fov_v_deg"] = self.crowdnav_provider.fov_deg
            sn_kwargs["lidar_cam_z_offset"] = 0.0
            sn_kwargs["lidar_cam_pitch_offset"] = 0.0
            sn_kwargs["lidar_cam_yaw_offset"] = 0.0
            sn_kwargs["mono_k"] = self.crowdnav_provider._fx * 0.3
            sn_kwargs["human_traj_pred"] = self.human_traj_pred
        for key in ("safety_sigma", "safety_h", "safety_gamma",
                    "safety_sigma_spread", "safety_h_traj_scale",
                    "shield_thresh_on", "shield_thresh_off",
                    "vx_sfm_gain", "human_pred_s",
                    "traj_gradient_gain", "traj_goal_gain", "traj_step_size",
                    "vx_min"):
            val = getattr(args, key, None)
            if val is not None:
                sn_kwargs[key] = val
        if args.traj_direct_step:
            sn_kwargs["traj_normalize_step"] = False
        self.social_nav = SocialNavigator(enabled=self.socialnav_enabled,
                                          **sn_kwargs)

        # ArUco goal detector (physical deployment)
        self.goal_mode = getattr(args, 'goal_mode', 'yolo')
        self.aruco_detector = None
        if self.goal_mode == 'aruco':
            from aruco_goal_detector import ArucoGoalDetector
            aruco_ids = tuple(int(x) for x in getattr(args, 'aruco_ids', '0,1').split(','))
            # Build camera matrix from social_nav intrinsics
            camera_matrix = np.array([
                [self.social_nav._fx, 0, self.social_nav._cx],
                [0, self.social_nav._fy, self.social_nav._cy],
                [0, 0, 1],
            ], dtype=np.float64)
            self.aruco_detector = ArucoGoalDetector(
                marker_ids=aruco_ids,
                marker_size_m=getattr(args, 'aruco_marker_size', 0.15),
                camera_matrix=camera_matrix,
            )
            print(f"[ArUco] Goal detection via ArUco markers {aruco_ids}")

        # Initialize UI
        self.root = Tk()
        self.root.title("Visual Language Motion Controller")
        self.font_style = ("Arial", 16, "bold")
        self.small_font = ("Arial", 14, "bold")

        self.image_frame = Frame(self.root)
        self.image_frame.pack(side='left', fill='both', expand=False)

        self.bev_frame = Frame(self.root)
        self.bev_frame.pack(side='left', anchor='se', padx=5, pady=5)

        self.instruction_frame = Frame(self.root)
        self.instruction_frame.pack(side='top', anchor='ne', expand=False)

        self.init_ui()

        if self.show_video:
            self.image_label = Label(self.image_frame)
            self.image_label.pack(fill='both', expand=True)

            if self.crowdnav_sim_mode:
                self.crowdnav_label = Label(self.bev_frame)
                self.crowdnav_label.pack(pady=(0, 5))
                self.crowdnav_sim_frame = None

            self.bev_label = Label(self.bev_frame)
            self.bev_label.pack()

            self.update_image()
        if self.simulation_mode:
            self.webcam = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
            if not self.webcam.isOpened():
                self.webcam = cv2.VideoCapture(0)
            if not self.webcam.isOpened():
                print("ERROR: Could not open webcam. Check camera permissions.")
            else:
                for _ in range(5):
                    self.webcam.read()

    def init_ui(self):
        """Initialize UI Interface"""
        screen_width = self.root.winfo_screenwidth()
        window_width = 1400
        window_height = 800
        self.root.geometry(f"{window_width}x{window_height}+{0}+20")

        control_frame = Frame(self.instruction_frame)
        control_frame.pack(pady=10, padx=10, anchor='n')
        self.stop_button = Button(control_frame, text="Stop",
               command=self._toggle_manual_stop,
               font=self.font_style, width=15, bg="red", fg="white")
        self.stop_button.pack(side='left', padx=5)
        Button(control_frame, text="Damp",
               command=self._damp_robot,
               font=self.font_style, width=15).pack(side='left', padx=5)
        Button(control_frame, text="Recovery Stand",
               command=lambda: print("RecoveryStand command") if self.simulation_mode else self.sport_client.RecoveryStand(),
               font=self.font_style, width=15).pack(side='left', padx=5)

        if self.crowdnav_sim_mode:
            sim_control_frame = Frame(self.instruction_frame)
            sim_control_frame.pack(pady=5, padx=10, anchor='n')
            Button(sim_control_frame, text="Start Sim",
                   command=self._start_sim,
                   font=self.font_style, width=12).pack(pady=5)
            Button(sim_control_frame, text="Reset Sim",
                   command=self._reset_sim,
                   font=self.font_style, width=12).pack(pady=5)
            Button(sim_control_frame, text="Save Paths",
                   command=self._save_paths,
                   font=self.font_style, width=12).pack(pady=5)

            self.pause_button = Button(sim_control_frame, text="Pause",
                   command=self._toggle_pause,
                   font=self.font_style, width=12)
            self.pause_button.pack(pady=5)

        initial_instructions = [
            args.mission_instruction
        ]

        self.instruction_entries = []
        for idx, instr in enumerate(initial_instructions):
            entry = Entry(self.instruction_frame, width=60, font=self.font_style)
            entry.insert(0, instr)
            entry.pack(pady=5)
            self.instruction_entries.append(entry)

            button = Button(
                self.instruction_frame,
                text=f"Submit Mission {idx + 1}",
                command=lambda e=entry: self.update_instruction(e),
                font=self.font_style
            )
            button.pack(pady=2)

        self.mission_label = Label(self.instruction_frame, text="Current Mission: ", font=self.font_style)
        self.mission_label.pack(pady=10)
        self.object_label = Label(self.instruction_frame, text="Extracted Object: ", font=self.font_style)
        self.object_label.pack(pady=10)
        self.state_label = Label(self.instruction_frame, text="Mission State: ", font=self.font_style)
        self.state_label.pack(pady=10)
        self.motion_label = Label(self.instruction_frame, text="Motion Vector: ", font=self.font_style)
        self.motion_label.pack(pady=10)

        freq_display_frame = Frame(self.instruction_frame, bd=1, relief='sunken', padx=10, pady=5)
        freq_display_frame.pack(side='bottom', fill='both', expand=True, padx=10, pady=10)

        self.freq_image_label = Label(freq_display_frame, text="[ImageGetter] Frequency: 0.00 Hz",
                                      font=self.small_font, anchor='w', fg='red')
        self.freq_image_label.pack(anchor='w', pady=2)

        self.freq_yolo_label = Label(freq_display_frame, text="[YoloProcessor] Frequency: 0.00 Hz",
                                     font=self.small_font, anchor='w', fg='red')
        self.freq_yolo_label.pack(anchor='w', pady=2)

        self.freq_yolo_pose_label = Label(freq_display_frame, text="[YoloPoseProcessor] Frequency: 0.00 Hz",
                                     font=self.small_font, anchor='w', fg='red')
        self.freq_yolo_pose_label.pack(anchor='w', pady=2)

        self.freq_motion_label = Label(freq_display_frame, text="[MotionControl] Frequency: 0.00 Hz",
                                       font=self.small_font, anchor='w', fg='red')
        self.freq_motion_label.pack(anchor='w', pady=2)

        self.freq_lidar_label = Label(freq_display_frame, text="[LiDARGetter] Frequency: 0.00 Hz",
                                      font=self.small_font, anchor='w', fg='red')
        self.freq_lidar_label.pack(anchor='w', pady=2)

        self.update_ui_labels()

    def update_ui_labels(self):
        """Update UI Status Labels"""
        self.mission_label.config(text=f"Current Mission: {self.mission_instruction_1}")
        self.object_label.config(text=f"Extracted Object: {self.extracted_object}")

        if self.button_update_inst:
            self.button_update_inst = False
        else:
            self.mission_instruction_0 = self.mission_instruction_1

        self.state_label.config(text=f"Mission State: {self.state['mission_state_in']}")
        motion_text = f"Motion Vector: {self.motion_vector}" if hasattr(self, 'motion_vector') else "Motion Vector: Not Available"
        self.motion_label.config(text=motion_text)

        self.root.after(1000, self.update_ui_labels)

    def update_freq_display(self):
        """Update Frequency Display Labels"""
        with self.freq_lock:
            img_freq = f"{self.image_getter_freq:.2f}"
            yolo_freq = f"{self.yolo_processor_freq:.2f}"
            pose_freq = f"{self.yolo_pose_processor_freq:.2f}"
            motion_freq = f"{self.motion_control_freq:.2f}"
            lidar_freq = f"{self.lidar_getter_freq:.2f}"

        self.freq_image_label.config(text=f"[ImageGetter] Frequency: {img_freq} Hz")
        self.freq_yolo_label.config(text=f"[YoloProcessor] Frequency: {yolo_freq} Hz")
        self.freq_yolo_pose_label.config(text=f"[YoloPoseProcessor] Frequency: {pose_freq} Hz")
        self.freq_motion_label.config(text=f"[MotionControl] Frequency: {motion_freq} Hz")
        self.freq_lidar_label.config(text=f"[LiDARGetter] Frequency: {lidar_freq} Hz")

        self.root.after(100, self.update_freq_display)

    def update_instruction(self, entry):
        """Process Mission Instruction Submission"""
        new_instr = entry.get()
        if new_instr:
            self.mission_instruction_0 = self.mission_instruction_1
            self.mission_instruction_1 = new_instr
            self.extracted_object = self.object_extractor.predict(new_instr)
            self.button_update_inst = True
            self.state["mission_state_in"] = "running"
            print(f"Updated Mission Instruction: {self.mission_instruction_1}")
            self.update_ui_labels()

    def _start_sim(self):
        """Start the CrowdNav simulation."""
        self.sim_started = True
        print("Simulation started.")

    def _reset_sim(self):
        """Reset the CrowdNav simulation for a new trial."""
        self.sim_started = False
        self.sim_paused = False
        with self.motion_lock:
            self.crowdnav_provider.reset(robot_theta=self.robot_theta)
            self.motion_vector = [0.0, 0.0, 0.0]
            self.state["mission_state_in"] = "success"
            self._planned_trajectory_world = None
            self.social_nav._predictor.reset()
            self.social_nav._tracked_humans.clear()
            self.social_nav._ego_velocity = None
            self.social_nav._frame_count = 0
            self.crowdnav_sim_frame = self.crowdnav_provider.render_frame()
        self.pause_button.config(text="Pause")
        print("Simulation reset. Press Start to begin.")

    def _toggle_pause(self):
        """Toggle pause state of the simulation."""
        if not self.sim_started:
            print("Simulation not started. Click Start Sim first.")
            return

        self.sim_paused = not self.sim_paused
        if self.sim_paused:
            self.pause_button.config(text="Resume")
            print("Simulation paused.")
        else:
            self.pause_button.config(text="Pause")
            print("Simulation resumed.")

    def _save_paths(self):
        """Save planned Bezier trajectory and actual robot path to a timestamped txt file."""
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"paths_{ts}.txt"

        actual = self.crowdnav_provider._robot_trajectory
        planned = self._planned_trajectory_world

        with open(filepath, 'w') as f:
            f.write("# Planned trajectory (Bezier, world coords)\n")
            f.write("# x y\n")
            if planned:
                for x, y in planned:
                    f.write(f"{x:.6f} {y:.6f}\n")
            else:
                f.write("# (no planned trajectory captured)\n")

            f.write("\n# Actual robot trajectory (world coords)\n")
            f.write("# x y\n")
            for x, y in actual:
                f.write(f"{x:.6f} {y:.6f}\n")

        print(f"Paths saved to {filepath}")

    def _init_channel_factory(self):
        # """Initialize Unitree Channel Factory"""
        # if len(sys.argv) > 1:
        #     ChannelFactoryInitialize(0, self.network_device)
        # else:
        #     ChannelFactoryInitialize(0)
        ChannelFactoryInitialize(0, self.network_device)

    def _init_camera(self):
        """Initialize Robot Camera Client Based on Robot Type"""
        if self.robot_type == "go2":
            client = Go2VideoClient()
        elif self.robot_type == "h1":
            client = Go2VideoClient()
        elif self.robot_type == "b2":
            client = B2FrontVideoClient()
        else:
            raise ValueError("Unsupported robot type. Supported types are: go2, h1, b2.")
        client.SetTimeout(3.0)
        client.Init()
        return client

    def _init_sport(self):
        """Initialize Robot Motion Control Client Based on Robot Type"""
        if self.robot_type == "go2":
            sport_client = Go2SportClient()
        elif self.robot_type == "h1":
            sport_client = H1SportClient()
        elif self.robot_type == "b2":
            sport_client = B2SportClient()
        else:
            raise ValueError("Unsupported robot type. Supported types are: go2, h1, b2.")
        sport_client.SetTimeout(10.0)
        sport_client.Init()
        return sport_client

    def _update_image_from_video_client(self):
        """Update Image from Robot's Built-in Camera"""
        code, data = self.video_client.GetImageSample()
        if code != 0:
            print("Failed to get image, error code:", code)
            return
        if isinstance(data, list):
            data = bytes(data)
        if len(data) == 0:
            return
        self.image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

    def _update_image_from_realsense(self):
        """Update Image from RealSense Camera"""
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if color_frame:
            self.image = np.asanyarray(color_frame.get_data())

    def _update_image_from_webcam(self):
        """Update Image from built-in webcam"""
        if not hasattr(self, 'webcam'):
            self.webcam = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
            if not self.webcam.isOpened():
                self.webcam = cv2.VideoCapture(0)
            if not self.webcam.isOpened():
                print("ERROR: Could not open webcam")
                return
        ret, frame = self.webcam.read()
        if ret:
            self.image = frame

    def _yolo_image_post_process(self, results, original_image):
        """Process YOLO Detection Results"""
        if not hasattr(self, '_wcheck'):
            self._wcheck = True
            h, w = original_image.shape[:2]
            if w != self.social_nav.params["image_width"]:
                print(f"[SocialNav] auto-correcting image_width: {self.social_nav.params['image_width']} -> {w}")
                self.social_nav.params["image_width"] = w
                half_fov = np.radians(self.social_nav.params["fov_deg"] / 2.0)
                self.social_nav._fx = (w / 2.0) / np.tan(half_fov)
                self.social_nav._cx = w / 2.0
            if h != self.social_nav.params["image_height"]:
                print(f"[SocialNav] auto-correcting image_height:{self.social_nav.params['image_height']} -> {h}")
                self.social_nav.params["image_height"] = h
                half_fov_v = np.radians(self.social_nav.params["fov_v_deg"] / 2.0)
                self.social_nav._fy = (h / 2.0) / np.tan(half_fov_v)
                self.social_nav._cy = h / 2.0




        detections = []
        for result in results:
            for box in result.boxes:
                class_name = result.names[int(box.cls)]
                if class_name == self.extracted_object:
                    img_height, img_width = original_image.shape[:2]
                    x_center_n, y_center_n = box.xywhn[0][0], box.xywhn[0][1]
                    width_n, height_n = box.xywhn[0][2], box.xywhn[0][3]
                    x1 = int((x_center_n - width_n/2) * img_width)
                    y1 = int((y_center_n - height_n/2) * img_height)
                    x2 = int((x_center_n + width_n/2) * img_width)
                    y2 = int((y_center_n + height_n/2) * img_height)

                    detections.append({
                        "object": class_name,
                        "confidence": float(box.conf),
                        "xyn": box.xywhn[0][:2].tolist(),
                        "whn": box.xywhn[0][2:].tolist(),
                        "xyxy": (x1, y1, x2, y2)
                    })

        if not hasattr(self, 'history_confidence'):
                self.history_object = []
                self.history_confidence = []
                self.history_xyn = []
                self.history_whn = []
                self.history_xyxy = []
                self.last_best = None

        if detections:
            best = max(detections, key=lambda x: x["confidence"])
            self.last_best = best
            self.history_object.append(best["object"])
            self.history_confidence.append(best["confidence"])
            self.history_xyn.append(best["xyn"])
            self.history_whn.append(best["whn"])
            self.history_xyxy.append(best["xyxy"])

            if len(self.history_object) > self.lengthen_filter:
                self.history_object.pop(0)
                self.history_confidence.pop(0)
                self.history_xyn.pop(0)
                self.history_whn.pop(0)
                self.history_xyxy.pop(0)
        else:
            self.history_object.append("NULL")
            self.history_confidence.append(0.00)
            self.history_xyn.append(self.last_best["xyn"] if self.last_best else [0.00, 0.00])
            self.history_whn.append(self.last_best["whn"] if self.last_best else [0.00, 0.00])
            self.history_xyxy.append(self.last_best["xyxy"] if self.last_best else [0, 0, 0, 0])

            if len(self.history_object) > self.lengthen_filter:
                self.history_object.pop(0)
                self.history_confidence.pop(0)
                self.history_xyn.pop(0)
                self.history_whn.pop(0)
                self.history_xyxy.pop(0)

        avg_confidence = np.mean(self.history_confidence)
        avg_xyn = np.mean(self.history_xyn, axis=0).tolist()
        avg_whn = np.mean(self.history_whn, axis=0).tolist()
        avg_xyxy = np.mean(self.history_xyxy, axis=0).tolist()
        avg_xyxy = [int(coord) for coord in avg_xyxy]

        most_common_object = max(set(self.history_object), key=self.history_object.count)

        if most_common_object == "NULL":
            avg_confidence = 0.00
            avg_xyn = [0.00, 0.00]
            avg_whn = [0.00, 0.00]
            avg_xyxy = None

        self.state.update({
            "predicted_object": most_common_object,
            "confidence": [avg_confidence],
            "object_xyn": avg_xyn,
            "object_whn": avg_whn,
            "bounding_box": avg_xyxy
        })

    def _yolo_pose_post_process(self, results, original_image):
        """Process YOLO Pose Detection Results"""
        poses = []
        pose_boxes = []

        for result in results:
            if result.keypoints is not None:
                for idx, keypoints in enumerate(result.keypoints):
                    if result.boxes is not None and idx < len(result.boxes):
                        box = result.boxes[idx]
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        confidence = float(box.conf)

                        kpts = keypoints.xy[0].cpu().numpy()
                        kpts_conf = keypoints.conf[0].cpu().numpy() if hasattr(keypoints, 'conf') else None

                        poses.append({
                            "keypoints": kpts.tolist(),
                            "keypoints_conf": kpts_conf.tolist() if kpts_conf is not None else None,
                            "confidence": confidence
                        })

                        pose_boxes.append([int(x1), int(y1), int(x2), int(y2)])

        self.pose_state.update({
            "num_people": len(poses),
            "poses": poses,
            "pose_boxes": pose_boxes
        })

    def _update_motion_control(self, state, lidar_cloud=None):
        """Update Motion Control Parameters Based on Detection Results"""
        if self.crowdnav_policy is not None:
            # ── CrowdNav policy path ──
            self.motion_vector = self._crowdnav_policy_action()
            self.state["mission_state_in"] = "running"
        else:
            # ── VLA (L2MM) path ──
            input_data = {
                "mission_instruction_0": self.mission_instruction_0,
                "mission_instruction_1": self.mission_instruction_1,
                **state
            }
            prediction = self.motion_predictor.predict(input_data)
            self.state["search_state_in"] = prediction["search_state"]

            if prediction["predicted_state"] == "success":
                self._success_frame_count += 1
            else:
                self._success_frame_count = 0

            if self._success_frame_count >= 3:
                self.state["mission_state_in"] = "success"
                self.motion_vector = [0.0, 0.0, 0.0]
            else:
                self.state["mission_state_in"] = prediction["predicted_state"]
                self.motion_vector = prediction["motion_vector"]

        if lidar_cloud is None:
            lidar_cloud = self.lidar_getter_thread.get_cloud() if self.lidar_getter_thread else None

        bbox = self.state.get("bounding_box")
        bbox_h = (bbox[3] - bbox[1]) if bbox else None
        self.social_nav.update_goal(self.state["object_xyn"], bbox_h,
                                    goal_depth=self.state.get("goal_depth"))

        self.motion_vector = self.social_nav.step(
            motion_vector=self.motion_vector,
            pose_state=self.pose_state,
            mission_state=self.state["mission_state_in"],
            lidar_ranges=lidar_cloud,
        )

    def _lidar_too_close(self) -> bool:
        """Return True if any LiDAR point is dangerously close in front of the robot."""
        if self.lidar_getter_thread is None:
            return False
        cloud = self.lidar_getter_thread.get_cloud()
        if cloud is None or len(next(iter(cloud.values()), [])) == 0:
            return False
        x, y, z = cloud['x'], cloud['y'], cloud['z']
        mask = (
            np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
            & (x > 0)
            & (z > self.LIDAR_ESTOP_Z_MIN)
            & (z < self.LIDAR_ESTOP_Z_MAX)
        )
        if not np.any(mask):
            return False
        dist = np.sqrt(x[mask] ** 2 + y[mask] ** 2)
        return float(dist.min()) < self.LIDAR_ESTOP_DISTANCE

    def _damp_robot(self):
        """Lay the robot down and zero ego-velocity."""
        if self.simulation_mode:
            print("StandDown command")
        else:
            self.sport_client.StandDown()
        self.manual_stop = True
        self.stop_button.config(bg="green", text="Resume")
        self.social_nav._ego_velocity = [0.0, 0.0, 0.0]

    def _toggle_manual_stop(self):
        """Toggle manual stop state."""
        self.manual_stop = not self.manual_stop
        if self.manual_stop:
            self.stop_button.config(bg="green", text="Resume")
            if not self.simulation_mode:
                self.sport_client.Move(0, 0, 0)
            print("[STOP] Manual stop activated")
        else:
            self.stop_button.config(bg="red", text="Stop")
            print("[STOP] Manual stop released")

    def _control_robot(self):
        """Send Motion Commands to Robot"""
        if hasattr(self, 'motion_vector'):
            v_x, v_y, w_z = [float(val) for val in self.motion_vector]
            if self.simulation_mode:
                print(f"vx={v_x:.4f}, vy={v_y:.4f}, wz={w_z:.4f}")
            elif self.manual_stop:
                self.sport_client.Move(0, 0, 0)
                self.social_nav._ego_velocity = [0.0, 0.0, 0.0]
            elif self._lidar_too_close():
                self._lidar_estop_count += 1
                if self._lidar_estop_count >= 3:
                    self.sport_client.Move(0, 0, 0)
                    self.social_nav._ego_velocity = [0.0, 0.0, 0.0]
                    print("[E-STOP] LiDAR proximity halt")
                else:
                    self.sport_client.Move(v_x, v_y, w_z)
            else:
                self._lidar_estop_count = 0
                self.sport_client.Move(v_x, v_y, w_z)

    def _show_results(self, image):
        """Draw Detection Results and Information on Image"""
        bbox = self.state["bounding_box"]
        if self.state["predicted_object"] != "NULL" and bbox is not None and self.show_max_result:
            x1, y1, x2, y2 = bbox
            confidence = self.state["confidence"][0]
            class_name = self.state["predicted_object"]
            object_cxy = self.state["object_xyn"]

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"{class_name}: {confidence:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x1, y1 - text_height - 5), (x1 + text_width, y1), (0, 255, 0), -1)
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            if self.show_arrowed:
                image_center = (image.shape[1] // 2, image.shape[0] // 2)
                object_center = (int(object_cxy[0] * image.shape[1]), int(object_cxy[1] * image.shape[0]))
                cv2.arrowedLine(image, image_center, object_center, (255, 0, 255), 2)

                cv2.circle(image, image_center, 10, (0, 255, 0), -1)
                cv2.circle(image, object_center, 10, (0, 255, 255), 2)

        if self.pose_state["num_people"] > 0:
            skeleton = [
                [0, 1], [0, 2], [1, 3], [2, 4],
                [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
                [5, 11], [6, 12], [11, 12],
                [11, 13], [13, 15], [12, 14], [14, 16]
            ]

            for idx, pose in enumerate(self.pose_state["poses"]):
                keypoints = pose["keypoints"]
                keypoints_conf = pose["keypoints_conf"]
                confidence = pose["confidence"]

                if idx < len(self.pose_state["pose_boxes"]):
                    x1, y1, x2, y2 = self.pose_state["pose_boxes"][idx]
                    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

                    dist_str = ""
                    if hasattr(self, 'social_nav') and self.social_nav.enabled:
                        h = self.social_nav._tracked_humans.get(idx)
                        if h and h.distance is not None:
                            dist_str = f" {h.distance:.1f}m"
                            if h.position_rf is not None:
                                dist_str += f" [{h.position_rf[0]:+.1f}, {h.position_rf[1]:.1f}]"
                            dist_str += f" Npt:{h.lidar_npts}"
                    label = f"Person: {confidence:.2f}{dist_str}"

                    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(image, (x1, y1 - text_height - 5), (x1 + text_width, y1), (255, 0, 0), -1)
                    cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                for i, (x, y) in enumerate(keypoints):
                    conf = keypoints_conf[i] if keypoints_conf else 1.0
                    if conf > 0.5:
                        cv2.circle(image, (int(x), int(y)), 4, (0, 255, 255), -1)

                for connection in skeleton:
                    pt1_idx, pt2_idx = connection
                    if (keypoints_conf is None or
                        (keypoints_conf[pt1_idx] > 0.5 and keypoints_conf[pt2_idx] > 0.5)):
                        pt1 = tuple(map(int, keypoints[pt1_idx]))
                        pt2 = tuple(map(int, keypoints[pt2_idx]))
                        cv2.line(image, pt1, pt2, (0, 255, 0), 2)

        # =========DRAW POINTS USED FOR HUMAN DISTANCE MEASUREMENT======
        # only writing this for 1 person right now!
        # if self.pose_state["num_people"] > 0:
        #     torso_pts = self.social_nav._human_torso
        #     ls_rs = [torso_pts[0], torso_pts[1]]
        #     rs_rh = [torso_pts[1], torso_pts[3]]
        #     rh_lh = [torso_pts[3], torso_pts[2]]
        #     lh_ls = [torso_pts[2], torso_pts[0]]

        #     torso_lines = [
        #         ls_rs,
        #         rs_rh, 
        #         rh_lh,
        #         lh_ls
        #     ]
        #     for line in torso_lines:
        #         cv2.line(image,
        #                  (int(line[0][0]), int(line[0][1])), 
        #                  (int(line[1][0]), int(line[1][1])), 
        #                  (0, 0, 0), 4) 
        # human_pts = self.social_nav._lidar_pts_in_torso
        # if human_pts is not None:
        #     for pt in human_pts:
        #         # cv2.circle(image, (int(pt[0]), int(pt[1])), 5, (0,0,0), -1)
        #         cv2.drawMarker(image, (int(pt[0]), int(pt[1])), (0, 0, 0), cv2.MARKER_STAR, 16, 2)


        texts = [
            f"Mission Instruction 1: {self.mission_instruction_1}",
            f"Mission Instruction 0: {self.mission_instruction_0}",
            f"Extracted Mission Object: {self.extracted_object}",
            f"Mission State In: {self.state['mission_state_in']}"
        ]
        if hasattr(self, 'motion_vector'):
            texts.append(f"Motion Vector: {self.motion_vector}")
        else:
            texts.append("Motion Vector: Not Available")

        y_positions = [30, 60, 90, 120, 150]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.9
        font_color = (255, 255, 0)
        font_thickness = 2
        padding = 5

        for text, y in zip(texts, y_positions):
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
            x = 10
            rect_x = x - padding
            rect_y = y - text_height - padding
            rect_width = text_width + 2 * padding
            rect_height = text_height + baseline + 2 * padding
            cv2.rectangle(image, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (0, 0, 0), -1)
            cv2.putText(image, text, (x, y), font, font_scale, font_color, font_thickness, cv2.LINE_AA)

        safety_texts = [
            f"SocialNav Enabled: {self.socialnav_enabled}",
            ]
        if hasattr(self, 'social_nav') and self.social_nav.enabled:
            min_d = self.social_nav.diag["min_distance"]
            n_humans = self.social_nav.diag["num_humans"]
            safety_score = self.social_nav.safety_score
            sheild_active = self.social_nav.shield_active
            min_hist = self.social_nav.diag.get("min_hist_len", 0)

            safety_texts.append(f"minimum distance: {min_d:.2f} m" if min_d is not None else "minimum distance: n/a")
            safety_texts.append(f"number of humans: {n_humans}")
            safety_texts.append(f"safety score: {safety_score:.2f}")
            safety_texts.append(f"shield active: {sheild_active}")
            lidar_npts = self.social_nav.diag.get("lidar_npts", {})
            total_npts = sum(lidar_npts.values()) if lidar_npts else 0
            safety_texts.append(f"lidar measurement pts: {total_npts}")
            safety_texts.append(f"min pred history: {min_hist}")
            min_pred = self.social_nav.diag.get("min_pred_len", 0)
            safety_texts.append(f"min pred future: {min_pred}")

        traj_score = self.social_nav.diag["traj_score"]
        safety_texts.append(f"traj score: {traj_score:.2f}")
        best_score = self.social_nav.diag["best_traj_score"]
        safety_texts.append(f"best score: {best_score:.2f}")

        safety_y_positions = [30 + i * 30 for i in range(len(safety_texts))]
        for safety_text, y in zip(safety_texts, safety_y_positions):
            (text_width, text_height), baseline = cv2.getTextSize(safety_text, font, font_scale, font_thickness)
            x = image.shape[1] - text_width - 10
            rect_x = x - padding
            rect_y = y - text_height - padding
            rect_width = text_width + 2 * padding
            rect_height = text_height + baseline + 2 * padding
            cv2.rectangle(image, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (0, 0, 0), -1)
            cv2.putText(image, safety_text, (x, y), font, font_scale, (0, 255, 255), font_thickness, cv2.LINE_AA)

        return image

    def update_image(self):
        """Update Video Display in UI"""
        try:
            if hasattr(self.image_getter_thread, 'image_queue'):
                img = self.image_getter_thread.image_queue.get(timeout=1)
                img = self._show_results(img)
                img = self.social_nav.overlay_lidar(img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                img = img.resize((800, 600), Image.LANCZOS)
                photo = ImageTk.PhotoImage(image=img)
                self.image_label.config(image=photo)
                self.image_label.image = photo

                if hasattr(self, 'social_nav'):
                    bev = self.social_nav.render_bev(show_heatmap=True)
                    bev = cv2.cvtColor(bev, cv2.COLOR_BGR2RGB)
                    bev_photo = ImageTk.PhotoImage(image=Image.fromarray(bev))
                    self.bev_label.config(image=bev_photo)
                    self.bev_label.image = bev_photo

                if hasattr(self, 'lidar_window') and hasattr(self, 'social_nav'):
                    cloud = self.social_nav._lidar_ranges
                    if cloud is not None:
                        self.lidar_window.update(cloud)
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Image update error: {e}")

        if self.crowdnav_sim_mode and getattr(self, 'crowdnav_sim_frame', None) is not None:
            sim_rgb = cv2.cvtColor(self.crowdnav_sim_frame, cv2.COLOR_BGR2RGB)
            sim_img = Image.fromarray(sim_rgb).resize((400, 400), Image.LANCZOS)
            sim_photo = ImageTk.PhotoImage(image=sim_img)
            self.crowdnav_label.config(image=sim_photo)
            self.crowdnav_label.image = sim_photo

        self.root.after(100, self.update_image)

    def start_threads(self):
        """Start All Worker Threads"""
        if not self.crowdnav_sim_mode:
            self.image_getter_thread.start()
            self.yolo_processing_thread.start()
            self.yolo_pose_processing_thread.start()
        self.motion_control_thread.start()
        if self.lidar_getter_thread:
            self.lidar_getter_thread.start()

    def stop_threads(self):
        """Stop All Worker Threads"""
        if not self.crowdnav_sim_mode:
            self.image_getter_thread.stop()
            self.yolo_processing_thread.stop()
            self.yolo_pose_processing_thread.stop()
        self.motion_control_thread.stop()
        if self.lidar_getter_thread:
            self.lidar_getter_thread.stop()
        if hasattr(self, 'webcam'):
            self.webcam.release()
        if self.camera_type == "realsense":
            self.pipeline.stop()

    def run(self):
        """Main Run Method"""
        self.start_threads()
        self.root.after(100, self.update_ui_labels)
        self.root.after(100, self.update_freq_display)
        self.root.mainloop()
        self.stop_threads()


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visual Language Motion Controller — GUI or headless batch evaluation"
    )

    # ── Headless-specific ──
    parser.add_argument("--headless", action="store_true", default=False,
                        help="Run without GUI for fast batch evaluation")
    parser.add_argument("--num_episodes", type=int, default=100,
                        help="Number of episodes to run (headless only)")
    parser.add_argument("--max_steps", type=int, default=200,
                        help="Max sim steps per episode before timeout")
    parser.add_argument("--csv_path", type=str, default="eval_results.csv",
                        help="Path to append batch summary rows to")
    parser.add_argument("--mission_instruction", type=str,
                        default="move to the handbag at speed of 0.5 m/s",
                        help="Mission instruction for all episodes")

    # ── Model paths ──
    parser.add_argument('--yolo_model_dir', type=str,
                        default="models/yolo-models/yolo11x.pt")
    parser.add_argument('--yolo_pose_model_dir', type=str,
                        default="models/yolo-models/yolo26n-pose.pt")
    parser.add_argument('--tokenizer_path', type=str,
                        default="models/tokenizer_language2motion_n1000000")
    parser.add_argument('--object_extraction_model_path', type=str,
                        default="models/model_object_extraction_n1000000_d64_h4_l2_f256_msl64_hold_success")
    parser.add_argument('--language2motion_model_path', type=str,
                        default="models/model_language2motion_n1000000_d128_h8_l4_f512_msl64_hold_success")

    # ── Hardware config ──
    parser.add_argument('--camera_type', type=str, default='inner',
                        choices=['inner', 'realsense'])
    parser.add_argument('--robot_type', type=str, default='go2',
                        choices=['go2', 'h1', 'b2'])

    # ── Display (ignored in headless) ──
    parser.add_argument('--show_video', action='store_true', default=True)
    parser.add_argument('--show_max_result', action='store_true', default=True)
    parser.add_argument('--show_arrowed', action='store_true', default=False)

    # ── Algorithm ──
    parser.add_argument('--threshold', type=float, default=10.0)
    parser.add_argument('--lengthen_filter', type=int, default=1)
    parser.add_argument('--simulation_mode', action='store_true', default=False)
    parser.add_argument('--socialnav_enabled', action='store_true', default=False)
    parser.add_argument('--image_width', type=int, default=640)
    parser.add_argument('--network_device', type=str, default="enx00e06c79d1cb")
    parser.add_argument('--crowdnav_sim_mode', action='store_true', default=False)
    parser.add_argument('--env_config', type=str, default='configs/env_lovon.config')
    parser.add_argument('--policy_config', type=str, default='configs/policy_lovon.config')
    parser.add_argument('--robot_theta', type=float, default=None)
    parser.add_argument('--show_bezier_pts', action='store_true', default=False)
    parser.add_argument('--disable_human_traj_pred', action='store_true', default=False,
                        help='Disable human trajectory prediction, use only gaussian for safety calculation')
    parser.add_argument('--verbose_failures', action='store_true', default=False,
                        help='Print termination reason for non-GOAL episodes')

    # ── Robot policy selection ──
    parser.add_argument('--robot_policy', type=str, default='vla',
                        choices=['vla', 'orca', 'sarl', 'lstm_rl', 'cadrl'],
                        help='Robot navigation policy. "vla" uses L2MM model, others use CrowdNav policies')
    parser.add_argument('--crowdnav_model_path', type=str, default=None,
                        help='Path to trained .pth weights for SARL/LSTM_RL/CADRL')
    parser.add_argument('--crowdnav_policy_config', type=str, default=None,
                        help='Policy config matching the trained weights (kinematics, network dims). '
                             'Defaults to --policy_config if not set.')

    # ── Safety Gaussian shape params ──
    parser.add_argument('--safety_sigma', type=float, default=None,
                        help='Gaussian width at human current position (meters). Default: 2.0')
    parser.add_argument('--safety_h', type=float, default=None,
                        help='Peak danger amplitude at distance=0 (0..1). Default: 1.0')
    parser.add_argument('--safety_gamma', type=float, default=None,
                        help='Per-step H multiplier along predicted trajectory. '
                             '<1 = danger fades, 1 = constant, >1 = danger grows. Default: 1.01')
    parser.add_argument('--safety_sigma_spread', type=float, default=None,
                        help='Sigma growth per prediction step (meters/step). Default: 0.1')
    parser.add_argument('--safety_h_traj_scale', type=float, default=None,
                        help='Per-step H multiplier along trajectory. '
                             '<1 = peak shrinks, 1 = unchanged, >1 = peak grows. Default: 1.0')

    # ── Social navigator params (shield / SFM / prediction) ──
    parser.add_argument('--shield_thresh_on', type=float, default=None,
                        help='Safety score below this activates shield. Default: 0.7')
    parser.add_argument('--shield_thresh_off', type=float, default=None,
                        help='Safety score above this deactivates shield. Default: 0.8')
    parser.add_argument('--vx_sfm_gain', type=float, default=None,
                        help='SFM forward velocity gain. Default: 3.0')
    parser.add_argument('--human_pred_s', type=float, default=None,
                        help='Human trajectory prediction horizon in seconds. Default: 5.0')

    # ── Trajectory planner params ──
    parser.add_argument('--traj_gradient_gain', type=float, default=None,
                        help='How strongly the safety gradient nudges each trajectory step. Default: 1.0')
    parser.add_argument('--traj_goal_gain', type=float, default=None,
                        help='Attractive force toward goal during gradient walk. Default: 0.3')
    parser.add_argument('--traj_step_size', type=float, default=None,
                        help='Step size in meters for gradient walk. Default: 0.2')
    parser.add_argument('--vx_min', type=float, default=None,
                        help='Minimum vx multiplier floor; negative allows reversing. Default: 0.0')
    parser.add_argument('--traj_direct_step', action='store_true', default=False,
                        help='Use direct force displacement (nxt = cur + grad + goal) instead of '
                             'normalized heading+grad+goal * step_size')

    # ── ArUco goal detection ──
    parser.add_argument('--goal_mode', type=str, default='yolo',
                        choices=['yolo', 'aruco'],
                        help='Goal detection method: yolo (default) or aruco markers')
    parser.add_argument('--aruco_marker_size', type=float, default=0.15,
                        help='ArUco marker physical size in meters (default 0.10)')
    parser.add_argument('--aruco_ids', type=str, default='0,1',
                        help='Comma-separated ArUco marker IDs (default "0,1")')

    # ── Environment overrides ──
    parser.add_argument('--robot_speed', type=float, default=None,
                        help='Robot speed in m/s. Overrides speed in --mission_instruction.')
    parser.add_argument('--human_speed', type=float, default=None,
                        help='Human preferred speed (v_pref) in m/s. Overrides env config.')
    parser.add_argument('--human_policy', type=str, default=None,
                        choices=['linear', 'orca'],
                        help='Human navigation policy. Overrides env config.')

    args = parser.parse_args()

    # ── Apply speed overrides ──
    if args.robot_speed is not None:
        args.mission_instruction = re.sub(
            r"(\d+\.?\d*)\s*m/s",
            f"{args.robot_speed} m/s",
            args.mission_instruction,
        )

    # Apply env config overrides (each reads the current env_config,
    # which may already be a temp file from a previous override)
    for attr, pattern in [
        ("human_speed",  r"^(v_pref\s*=).*$"),
        ("human_policy", r"^(policy\s*=).*$"),
    ]:
        val = getattr(args, attr, None)
        if val is not None:
            with open(args.env_config, "r") as f:
                text = f.read()
            text = re.sub(pattern, rf"\g<1> {val}", text, flags=re.MULTILINE)
            fd, tmp_path = tempfile.mkstemp(prefix="env_lovon_", suffix=".config")
            with os.fdopen(fd, "w") as f:
                f.write(text)
            args.env_config = tmp_path

    mode = "HEADLESS" if args.headless else "GUI"
    print("=" * 60)
    print(f"  deploy.py — {mode}")
    print("=" * 60)
    print(f"  mission:          {args.mission_instruction}")
    print(f"  robot_policy:     {args.robot_policy}")
    print(f"  robot_theta:      {args.robot_theta}")
    print(f"  socialnav:        {args.socialnav_enabled}")
    print(f"  traj_pred:        {not args.disable_human_traj_pred}")
    if args.headless:
        print(f"  num_episodes:     {args.num_episodes}")
        print(f"  max_steps:        {args.max_steps}")
        print(f"  csv_path:         {args.csv_path}")
    print("── files ──────────────────────────────────────────────")
    print(f"  env_config:       {args.env_config}")
    print(f"  policy_config:    {args.policy_config}")
    print(f"  yolo:             {args.yolo_model_dir}")
    print(f"  yolo_pose:        {args.yolo_pose_model_dir}")
    print(f"  tokenizer:        {args.tokenizer_path}")
    print(f"  obj_extraction:   {args.object_extraction_model_path}")
    print(f"  l2mm:             {args.language2motion_model_path}")
    if args.crowdnav_model_path:
        print(f"  crowdnav_model:   {args.crowdnav_model_path}")
    if args.crowdnav_policy_config:
        print(f"  crowdnav_policy:  {args.crowdnav_policy_config}")
    print("── sn params ──────────────────────────────────────────")
    sn_param_args = [
        ("shield_thresh_on",  args.shield_thresh_on),
        ("shield_thresh_off", args.shield_thresh_off),
        ("safety_sigma",      args.safety_sigma),
        ("safety_gamma",      args.safety_gamma),
        ("safety_sigma_spread", args.safety_sigma_spread),
        ("vx_sfm_gain",       args.vx_sfm_gain),
        ("vx_min",            args.vx_min),
        ("traj_gradient_gain", args.traj_gradient_gain),
        ("traj_goal_gain",    args.traj_goal_gain),
        ("traj_step_size",    args.traj_step_size),
        ("human_pred_s",      args.human_pred_s),
    ]
    for name, val in sn_param_args:
        flag = f"{val}" if val is not None else "(default)"
        print(f"  {name:<22s}  {flag}")
    print("=" * 60)

    if args.headless:
        args.crowdnav_sim_mode = True
        _import_headless_deps()
        runner = HeadlessRunner(args)
        runner.run()
    else:
        _import_gui_deps()
        controller = VisualLanguageController(args)
        controller.run()

    print("Program terminated.")
