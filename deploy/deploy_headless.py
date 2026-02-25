"""
Headless batch-evaluation variant of deploy.py.

Usage:
    python deploy/deploy_headless.py --crowdnav_sim_mode --socialnav_enabled --headless --num_episodes 100

When --headless is passed:
  - All tkinter / GUI / rendering code is skipped
  - The simulation loop runs synchronously (no threads, no real-time pacing)
  - Episodes auto-start, auto-reset, and results are collected to a CSV

When --headless is NOT passed, behaviour is identical to deploy.py.
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
from crowd_sim.envs.utils.info import Collision, Danger, ReachGoal, Timeout
from crowd_sim.envs.utils.state import JointState

# CrowdNav policy choices (non-VLA)
CROWDNAV_POLICIES = ("orca", "sarl", "lstm_rl", "cadrl")


# ═══════════════════════════════════════════════════════════════════════
#  HEADLESS RUNNER  (no threads, no GUI, synchronous tight loop)
# ═══════════════════════════════════════════════════════════════════════

class HeadlessRunner:
    """Runs CrowdNav episodes as fast as possible without any GUI."""

    def __init__(self, args):
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

        # ── State dicts (same keys deploy.py uses) ──
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

        # ── CrowdNav provider ──
        from models.crowdnav_data_provider import CrowdNavDataProvider
        self.crowdnav_provider = CrowdNavDataProvider(
            env_config_path=args.env_config,
            policy_config_path=args.policy_config,
            target_object=self.extracted_object,
        )
        # Skip init_render() — we don't need matplotlib in headless mode

        # ── CrowdNav robot policy (non-VLA) ──
        self.crowdnav_policy = None
        if self.robot_policy_name in CROWDNAV_POLICIES:
            self.crowdnav_policy = self._load_crowdnav_policy(
                self.robot_policy_name, args)

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
        # Forward any safety gaussian overrides from CLI
        for key in ("safety_sigma", "safety_h", "safety_gamma",
                    "safety_sigma_spread", "safety_h_traj_scale"):
            val = getattr(args, key, None)
            if val is not None:
                sn_kwargs[key] = val
        self.social_nav = SocialNavigator(
            enabled=args.socialnav_enabled, **sn_kwargs
        )

    # ------------------------------------------------------------------ #
    #  CrowdNav policy loader
    # ------------------------------------------------------------------ #

    def _load_crowdnav_policy(self, name, args):
        """Instantiate and configure a CrowdNav policy by name."""
        import configparser
        from crowd_nav.policy.policy_factory import policy_factory

        policy = policy_factory[name]()

        # Use dedicated policy config if provided (must match trained weights),
        # otherwise fall back to the main policy config.
        cfg_path = getattr(args, "crowdnav_policy_config", None) or args.policy_config
        policy_config = configparser.RawConfigParser()
        policy_config.read(cfg_path)
        policy.configure(policy_config)

        # ORCA needs time_step set explicitly
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
            # RL policies with query_env need the env reference
            if getattr(policy, 'query_env', False):
                policy.set_env(self.crowdnav_provider.env)

        # Sync robot kinematics with the policy so env.step() and
        # onestep_lookahead() accept the same action type the policy emits.
        policy_kin = getattr(policy, 'kinematics', None)
        if policy_kin and policy_kin != self.crowdnav_provider.kinematics:
            print(f"  Overriding robot kinematics: "
                  f"{self.crowdnav_provider.kinematics} → {policy_kin}")
            self.crowdnav_provider.robot.kinematics = policy_kin
            self.crowdnav_provider.kinematics = policy_kin

        print(f"Loaded CrowdNav policy: {name}  trainable={policy.trainable}  "
              f"kinematics={getattr(policy, 'kinematics', '?')}")
        return policy

    # ------------------------------------------------------------------ #
    #  Single episode
    # ------------------------------------------------------------------ #

    def _run_episode(self, episode_idx):
        """Run one CrowdNav episode. Returns a result dict."""
        self.crowdnav_provider.reset(robot_theta=self.args.robot_theta)
        self.motion_vector = [0.0, 0.0, 0.0]
        self.state["mission_state_in"] = "running"

        # Reset social nav state
        self.social_nav._predictor.reset()
        self.social_nav._tracked_humans.clear()
        self.social_nav._ego_velocity = None
        self.social_nav._frame_count = 0

        step_count = 0
        max_steps = self.args.max_steps
        t0 = time.perf_counter()

        # Per-episode near-miss accumulators
        collision = False
        min_distance_episode = float('inf')
        danger_count = 0
        termination_reason = "max_steps"

        while step_count < max_steps:
            if self.crowdnav_policy is not None:
                synthetic = self._crowdnav_step(self.motion_vector)
            else:
                synthetic = self.crowdnav_provider.step(self.motion_vector)
            if synthetic is None:
                break
            step_count += 1

            # Track near-miss metrics from CrowdNav info
            info = synthetic["info"]
            if isinstance(info, Collision):
                collision = True
                termination_reason = "collision"
            elif isinstance(info, Danger):
                danger_count += 1
            elif isinstance(info, ReachGoal):
                termination_reason = "goal"
            elif isinstance(info, Timeout):
                termination_reason = "timeout"

            # Compute true min distance from robot/human positions
            robot_state = self.crowdnav_provider.robot.get_full_state()
            humans = self.crowdnav_provider.env.humans
            if humans:
                dmin_step = min(
                    np.hypot(h.px - robot_state.px, h.py - robot_state.py)
                    - h.radius - robot_state.radius
                    for h in humans
                )
                min_distance_episode = min(min_distance_episode, dmin_step)

            # Update controller state from synthetic data
            self.pose_state = synthetic["pose_state"]
            self.state.update(synthetic["object_state"])

            # Build L2MM input and get motion vector
            state_copy = {**self.state}
            self._update_motion_control(state_copy, lidar_cloud=synthetic["lidar"])

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
            "termination_reason": termination_reason,
        }

    # ------------------------------------------------------------------ #
    #  Motion control (same logic as deploy.py VisualLanguageController)
    # ------------------------------------------------------------------ #

    def _update_motion_control(self, state, lidar_cloud=None):
        if self.crowdnav_policy is not None:
            # ── CrowdNav policy path ──
            self.motion_vector = self._crowdnav_policy_action()
            # Keep mission state as "running" so the shield can engage
            self.state["mission_state_in"] = "running"
        else:
            # ── VLA (L2MM) path ──
            input_data = {
                "mission_instruction_0": self.mission_instruction_0,
                "mission_instruction_1": self.mission_instruction_1,
                **state,
            }
            prediction = self.motion_predictor.predict(input_data)
            self.state["mission_state_in"] = prediction["predicted_state"]
            self.state["search_state_in"] = prediction["search_state"]
            self.motion_vector = prediction["motion_vector"]
            if self.state["mission_state_in"] == "success":
                self.motion_vector = [0.0, 0.0, 0.0]

        bbox = self.state.get("bounding_box")
        bbox_h = (bbox[3] - bbox[1]) if bbox else None
        self.social_nav.update_goal(self.state["object_xyn"], bbox_h)

        self.motion_vector = self.social_nav.step(
            motion_vector=self.motion_vector,
            pose_state=self.pose_state,
            mission_state=self.state["mission_state_in"],
            lidar_ranges=lidar_cloud,
        )

    def _crowdnav_policy_action(self):
        """Get action from CrowdNav policy → body-frame [vx, vy, wz].

        Also stores self._last_crowdnav_action so _crowdnav_step() can
        feed the *original* action type directly to env.step(), avoiding
        a lossy world→body→world round-trip.
        """
        robot = self.crowdnav_provider.robot
        self_state = robot.get_full_state()
        human_states = self.crowdnav_provider.ob
        if human_states is None:
            self._last_crowdnav_action = None
            return [0.0, 0.0, 0.0]

        joint_state = JointState(self_state, human_states)
        action = self.crowdnav_policy.predict(joint_state)
        self._last_crowdnav_action = action

        # Convert to body-frame [vx, vy, wz] for the SocialNav shield
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
            return [0.0, 0.0, 0.0]

    def _crowdnav_step(self, motion_vector):
        """Step the CrowdNav env using a body-frame motion_vector.

        When socialnav is disabled (motion_vector unchanged from policy output),
        we feed the original CrowdNav action directly to env.step() to avoid
        conversion artifacts.  When socialnav modulated the vector, we convert
        the body-frame vector back to a world-frame action for the env.
        """
        from crowd_sim.envs.utils.action import ActionXY, ActionRot, ActionXYRot

        env = self.crowdnav_provider.env
        robot = self.crowdnav_provider.robot
        self_state = robot.get_full_state()
        kin = getattr(self.crowdnav_policy, 'kinematics', 'holonomic')

        # Convert body-frame [vx, vy, wz] → world-frame CrowdNav action
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
        else:  # unicycle_xyrot
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
            print(
                f"  [{ep+1}/{num_episodes}] {status}  "
                f"steps={res['steps']}  sim_t={res['sim_time']:.2f}s  "
                f"wall={res['time_s']:.3f}s  goal_dist={res['final_goal_dist']:.3f}  "
                f"dmin={res['min_distance']:.3f}  danger={res['danger_count']}"
            )

        batch_elapsed = time.perf_counter() - batch_t0

        # ── Summary ──
        goals = sum(1 for r in results if r["reached_goal"])
        collisions = sum(1 for r in results if r["collision"])
        avg_min_dist = np.mean([r["min_distance"] for r in results]) if results else 0
        avg_danger = np.mean([r["danger_count"] for r in results]) if results else 0
        print(f"\n{'='*60}")
        print(f"  Episodes:   {num_episodes}")
        print(f"  Success:    {goals}/{num_episodes}  ({100*goals/max(num_episodes,1):.1f}%)")
        print(f"  Collisions: {collisions}/{num_episodes}  ({100*collisions/max(num_episodes,1):.1f}%)")
        print(f"  Avg min distance: {avg_min_dist:.3f} m")
        print(f"  Avg danger count: {avg_danger:.1f} steps/episode")
        print(f"  Wall time: {batch_elapsed:.2f}s  "
              f"({batch_elapsed/max(num_episodes,1):.3f}s / episode)")
        print(f"{'='*60}")

        # ── Append batch summary row to CSV ──
        csv_path = self.args.csv_path
        avg_steps = np.mean([r["steps"] for r in results]) if results else 0
        avg_sim_time = np.mean([r["sim_time"] for r in results]) if results else 0
        avg_goal_dist = np.mean([r["final_goal_dist"] for r in results]) if results else 0

        # Read env config params for the CSV row
        env_cfg = self.crowdnav_provider.env.config
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
            "success_rate": goals / max(num_episodes, 1),
            "collision_rate": collisions / max(num_episodes, 1),
            "avg_min_distance": avg_min_dist,
            "avg_danger_count": avg_danger,
            "avg_steps": avg_steps,
            "avg_sim_time": avg_sim_time,
            "avg_goal_dist": avg_goal_dist,
            "wall_time": batch_elapsed,
        }
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=batch_row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(batch_row)
        print(f"Batch summary appended to {csv_path}")


# ═══════════════════════════════════════════════════════════════════════
#  GUI RUNNER  (identical to original deploy.py)
# ═══════════════════════════════════════════════════════════════════════
#  Only imported / instantiated when --headless is NOT used.
#  All the thread classes and VisualLanguageController from deploy.py
#  are pulled in lazily to avoid importing tkinter on headless boxes.


def _run_gui(args):
    """Fall back to the original deploy.py GUI path.

    Since deploy.py reads args from its own argparse at module level,
    we delegate by exec-ing it as a new process with the same CLI args
    (minus headless-only flags).
    """
    cmd = [sys.executable, os.path.join(current_dir, "deploy.py")] + _strip_headless_args(sys.argv[1:])
    os.execvp(sys.executable, cmd)


def _strip_headless_args(argv):
    """Remove headless-only arguments before forwarding to deploy.py."""
    skip_next = False
    out = []
    headless_flags = {
        "--headless", "--num_episodes", "--max_steps", "--mission_instruction",
        "--csv_path", "--robot_policy", "--crowdnav_model_path", "--crowdnav_policy_config",
    }
    for i, arg in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if arg in headless_flags:
            if arg == "--headless":
                continue  # boolean flag, just skip
            else:
                skip_next = True  # skip the flag and its value
                continue
        out.append(arg)
    return out


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visual Language Motion Controller — headless batch evaluation"
    )

    # ── Headless-specific ──
    parser.add_argument("--headless", action="store_true", default=False,
                        help="Run without GUI for fast batch evaluation")
    parser.add_argument("--num_episodes", type=int, default=100,
                        help="Number of episodes to run (headless only)")
    parser.add_argument("--max_steps", type=int, default=500,
                        help="Max sim steps per episode before timeout")
    parser.add_argument("--csv_path", type=str, default="eval_results.csv",
                        help="Path to append batch summary rows to")
    parser.add_argument("--mission_instruction", type=str,
                        default="move to the handbag at speed of 0.5 m/s",
                        help="Mission instruction for all episodes")

    # ── Model paths (same as deploy.py) ──
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

    args = parser.parse_args()

    if args.headless:
        # Force crowdnav_sim_mode on — headless only makes sense with the sim
        args.crowdnav_sim_mode = True
        _import_headless_deps()
        runner = HeadlessRunner(args)
        runner.run()
    else:
        _run_gui(args)

    print("Program terminated.")
