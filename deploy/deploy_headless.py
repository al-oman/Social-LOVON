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


# ═══════════════════════════════════════════════════════════════════════
#  HEADLESS RUNNER  (no threads, no GUI, synchronous tight loop)
# ═══════════════════════════════════════════════════════════════════════

class HeadlessRunner:
    """Runs CrowdNav episodes as fast as possible without any GUI."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── Models ──
        self.object_extractor = SequenceToSequenceClassAPI(
            model_path=args.object_extraction_model_path,
            tokenizer_path=args.tokenizer_path,
        )
        self.motion_predictor = MotionPredictor(
            model_path=args.language2motion_model_path,
            tokenizer_path=args.tokenizer_path,
        )

        # ── Mission ──
        self.mission_instruction_0 = args.mission_instruction
        self.mission_instruction_1 = args.mission_instruction
        self.extracted_object = self.object_extractor.predict(self.mission_instruction_1)

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
                    "safety_sigma_spread", "safety_h_decay"):
            val = getattr(args, key, None)
            if val is not None:
                sn_kwargs[key] = val
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
    """Fall back to the original deploy.py GUI path."""
    # Re-use the original deploy.py module directly
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "deploy_gui", os.path.join(current_dir, "deploy.py")
    )
    mod = importlib.util.find_module
    # Simpler: just exec deploy.py's __main__ block via subprocess-like reimport
    # But the cleanest approach is to just call the original file.
    # Since deploy.py reads `args` from its own argparse at module level,
    # we delegate by running it as a subprocess with the same CLI args.
    import subprocess
    cmd = [sys.executable, os.path.join(current_dir, "deploy.py")] + _strip_headless_args(sys.argv[1:])
    os.execvp(sys.executable, cmd)


def _strip_headless_args(argv):
    """Remove headless-only arguments before forwarding to deploy.py."""
    skip_next = False
    out = []
    headless_flags = {"--headless", "--num_episodes", "--max_steps", "--mission_instruction"}
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
    parser.add_argument('--safety_h_decay', type=float, default=None,
                        help='Per-step H decay multiplier along trajectory. '
                             '<1 = peak shrinks, 1 = unchanged. Default: 1.0')

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
