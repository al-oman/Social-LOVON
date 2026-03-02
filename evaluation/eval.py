#!/usr/bin/env python3
"""
Social-LOVON headless evaluation sweep.

Modes:
  Full sweep (default):
    python evaluation/eval.py

  Side-by-side comparison (4 conditions):
    python evaluation/eval.py --compare

    Runs: VLA+snOFF, VLA+snON, ORCA, SARL
    across all human_num / human_speed / human_policy / theta combos.

Each combination gets a temporary env config with the parameter
overrides, then runs deploy.py --headless.
Results CSVs are collected into evaluation/results/<timestamp>/
"""

import os
import sys
import re
import subprocess
import tempfile
import itertools
import datetime
import argparse

# ── Paths ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DEPLOY = os.path.join(PROJECT_ROOT, "deploy", "deploy.py")
BASE_ENV_CONFIG = os.path.join(PROJECT_ROOT, "configs", "env_lovon.config")
POLICY_CONFIG = os.path.join(PROJECT_ROOT, "configs", "policy_lovon.config")

# SARL defaults (relative to project root)
SARL_MODEL_PATH = os.path.join(PROJECT_ROOT, "data", "output", "rl_model.pth")
SARL_POLICY_CONFIG = os.path.join(PROJECT_ROOT, "data", "output", "policy_non_holo.config")

# ── Full sweep parameters ──
ROBOT_POLICIES = ["vla", "orca"]
CROWDNAV_MODEL_PATH = os.path.join(PROJECT_ROOT, "data", "output", "rl_model.pth")
CROWDNAV_POLICY_CONFIG = os.path.join(PROJECT_ROOT, "configs", "policy.config")

SOCIALNAV_FLAGS = [False, True]
ROBOT_THETAS = [2.3562, 0.7854, 1.5708]
HUMAN_NUMS = [1, 2]
HUMAN_SPEEDS = [1.0]
HUMAN_POLICIES = ["orca"]
TRAJ_PRED_FLAGS = [True, False]
ROBOT_SPEEDS = [1.0]
TRAJ_DIRECT_STEP = False   # set True to use direct force method instead of normalized heading
DISABLE_VX_MOD  = False   # set True to disable vx modulation (sets vx_min=1.0)

# ── Per-run settings ──
DEFAULT_NUM_EPISODES = 20
DEFAULT_MAX_STEPS = 100


# ═══════════════════════════════════════════════════════════════════════
#  Shared helpers
# ═══════════════════════════════════════════════════════════════════════

def make_env_config(human_num, human_speed, human_policy):
    """Create a temporary env config with the given overrides."""
    with open(BASE_ENV_CONFIG, "r") as f:
        text = f.read()

    text = re.sub(r"^human_num\s*=.*$", f"human_num = {human_num}", text, flags=re.MULTILINE)
    text = re.sub(r"^v_pref\s*=.*$", f"v_pref = {human_speed}", text, flags=re.MULTILINE)
    text = re.sub(r"^policy\s*=.*$", f"policy = {human_policy}", text, flags=re.MULTILINE)

    fd, path = tempfile.mkstemp(prefix="env_lovon_", suffix=".config")
    with os.fdopen(fd, "w") as f:
        f.write(text)
    return path


def run_one(params, csv_path, num_episodes, max_steps):
    """Run a single headless evaluation. Returns exit code."""
    tmp_config = make_env_config(
        params["human_num"], params["human_speed"], params["human_policy"]
    )

    try:
        mission = f"move to the handbag at speed of {params['robot_speed']} m/s"

        cmd = [
            sys.executable, DEPLOY,
            "--headless",
            "--num_episodes", str(num_episodes),
            "--max_steps", str(max_steps),
            "--csv_path", csv_path,
            "--mission_instruction", mission,
            "--robot_theta", str(params["theta"]),
            "--env_config", tmp_config,
            "--policy_config", params.get("policy_config", POLICY_CONFIG),
            "--robot_policy", params["robot_policy"],
        ]

        # Trained CrowdNav policies need model weights
        model_path = params.get("crowdnav_model_path")
        policy_cfg = params.get("crowdnav_policy_config")
        if model_path:
            cmd += ["--crowdnav_model_path", model_path]
        if policy_cfg:
            cmd += ["--crowdnav_policy_config", policy_cfg]

        if params.get("socialnav"):
            cmd.append("--socialnav_enabled")

        if TRAJ_DIRECT_STEP:
            cmd.append("--traj_direct_step")

        if DISABLE_VX_MOD:
            cmd += ["--vx_min", "1.0"]

        if params.get("vx_min") is not None:
            cmd += ["--vx_min", str(params["vx_min"])]

        if not params.get("traj_pred", True):
            cmd.append("--disable_human_traj_pred")

        result = subprocess.run(cmd, cwd=PROJECT_ROOT)
        return result.returncode
    finally:
        os.unlink(tmp_config)


def run_sweep(sweep, csv_path, num_episodes, max_steps):
    """Execute a list of param dicts, printing progress."""
    total = len(sweep)
    failures = []

    for i, params in enumerate(sweep):
        pct = 100 * i // total
        print("-" * 56)
        print(f"  [{i+1}/{total}] ({pct}%)  {params['tag']}")
        print("-" * 56)

        rc = run_one(params, csv_path, num_episodes, max_steps)
        if rc != 0:
            failures.append(params["tag"])
            print(f"  WARNING: exited with code {rc}")
        print()

    return failures


# ═══════════════════════════════════════════════════════════════════════
#  Full sweep
# ═══════════════════════════════════════════════════════════════════════

def build_full_sweep():
    """Generate all parameter combinations for the full sweep."""
    combos = list(itertools.product(
        ROBOT_POLICIES,
        SOCIALNAV_FLAGS,
        ROBOT_THETAS,
        HUMAN_NUMS,
        HUMAN_SPEEDS,
        HUMAN_POLICIES,
        TRAJ_PRED_FLAGS,
        ROBOT_SPEEDS,
    ))

    sweep = []
    for rpolicy, socialnav, theta, nhumans, hspeed, hpolicy, traj_pred, rspeed in combos:
        sn_label = "on" if socialnav else "off"
        tp_label = "tpOn" if traj_pred else "tpOff"
        tag = (f"{rpolicy}_sn{sn_label}_theta{theta}_h{nhumans}"
               f"_hspd{hspeed}_{hpolicy}_{tp_label}_rspd{rspeed}")

        params = {
            "robot_policy": rpolicy,
            "socialnav": socialnav,
            "theta": theta,
            "human_num": nhumans,
            "human_speed": hspeed,
            "human_policy": hpolicy,
            "traj_pred": traj_pred,
            "robot_speed": rspeed,
            "tag": tag,
        }

        # Trained policies need weights
        if rpolicy not in ("vla", "orca"):
            params["crowdnav_model_path"] = CROWDNAV_MODEL_PATH
            params["crowdnav_policy_config"] = CROWDNAV_POLICY_CONFIG

        sweep.append(params)
    return sweep


# ═══════════════════════════════════════════════════════════════════════
#  --compare mode: VLA+snOFF vs VLA+snON vs ORCA vs SARL
# ═══════════════════════════════════════════════════════════════════════

# The four conditions to compare
COMPARE_CONDITIONS = [
    {
        "label": "vla_snON",
        "robot_policy": "vla",
        "socialnav": True,
        "traj_pred": True,
    },
    {
        "label": "vla_snON",
        "robot_policy": "vla",
        "socialnav": True,
        "traj_pred": False,
    },
    {
        "label": "vla_snOFF",
        "robot_policy": "vla",
        "socialnav": False,
        "traj_pred": True,
    },
    {
        "label": "vla_snON_novx",
        "robot_policy": "vla",
        "socialnav": True,
        "traj_pred": True,
        "vx_min": 1.0,
    },
    {
        "label": "orca",
        "robot_policy": "orca",
        "socialnav": False,
        "traj_pred": True,
    },
    {
        "label": "sarl",
        "robot_policy": "sarl",
        "socialnav": False,
        "traj_pred": True,
        "crowdnav_model_path": SARL_MODEL_PATH,
        "crowdnav_policy_config": SARL_POLICY_CONFIG,
    },
]

# Environment axes to sweep across in compare mode
COMPARE_THETAS = [2.356, 1.5708, 0.785]
COMPARE_HUMAN_NUMS = [1, 2, 3]
COMPARE_HUMAN_SPEEDS = [1.0]
COMPARE_HUMAN_POLICIES = ["orca"]
COMPARE_ROBOT_SPEEDS = [0.5, 1.0]


def build_compare_sweep():
    """Build sweep for --compare: 4 conditions x env combos."""
    env_combos = list(itertools.product(
        COMPARE_THETAS,
        COMPARE_HUMAN_NUMS,
        COMPARE_HUMAN_SPEEDS,
        COMPARE_HUMAN_POLICIES,
        COMPARE_ROBOT_SPEEDS,
    ))

    sweep = []
    for condition in COMPARE_CONDITIONS:
        for theta, nhumans, hspeed, hpolicy, rspeed in env_combos:
            tag = (f"{condition['label']}_theta{theta}_h{nhumans}"
                   f"_hspd{hspeed}_{hpolicy}_rspd{rspeed}")

            params = {
                "robot_policy": condition["robot_policy"],
                "socialnav": condition["socialnav"],
                "traj_pred": condition["traj_pred"],
                "theta": theta,
                "human_num": nhumans,
                "human_speed": hspeed,
                "human_policy": hpolicy,
                "robot_speed": rspeed,
                "tag": tag,
            }

            if "crowdnav_model_path" in condition:
                params["crowdnav_model_path"] = condition["crowdnav_model_path"]
            if "crowdnav_policy_config" in condition:
                params["crowdnav_policy_config"] = condition["crowdnav_policy_config"]
            if "vx_min" in condition:
                params["vx_min"] = condition["vx_min"]

            sweep.append(params)
    return sweep


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Social-LOVON headless evaluation sweep"
    )
    parser.add_argument("--compare", action="store_true",
                        help="Run side-by-side: VLA+snOFF vs VLA+snON vs ORCA vs SARL")
    parser.add_argument("--num_episodes", type=int, default=DEFAULT_NUM_EPISODES,
                        help=f"Episodes per configuration (default: {DEFAULT_NUM_EPISODES})")
    parser.add_argument("--max_steps", type=int, default=DEFAULT_MAX_STEPS,
                        help=f"Max sim steps per episode (default: {DEFAULT_MAX_STEPS})")
    parser.add_argument("--sarl_model_path", type=str, default=SARL_MODEL_PATH,
                        help="Path to SARL .pth weights")
    parser.add_argument("--sarl_policy_config", type=str, default=SARL_POLICY_CONFIG,
                        help="Path to SARL policy config (non-holonomic)")
    args = parser.parse_args()

    # Update SARL paths if overridden
    if args.compare:
        for cond in COMPARE_CONDITIONS:
            if cond["robot_policy"] == "sarl":
                cond["crowdnav_model_path"] = args.sarl_model_path
                cond["crowdnav_policy_config"] = args.sarl_policy_config

    if args.compare:
        sweep = build_compare_sweep()
        mode_label = "compare (VLA+snOFF vs VLA+snON vs ORCA vs SARL)"
    else:
        sweep = build_full_sweep()
        mode_label = "full sweep"

    total = len(sweep)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(SCRIPT_DIR, "results", timestamp)
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "eval_results.csv")

    print("=" * 60)
    print(f"  Social-LOVON evaluation — {mode_label}")
    print(f"  {total} configurations x {args.num_episodes} episodes each")
    print(f"  Results:      {results_dir}")
    print(f"  Deploy:       {DEPLOY}")
    print(f"  Env config:   {BASE_ENV_CONFIG}")
    print(f"  Policy config:{POLICY_CONFIG}")
    print(f"  CrowdNav model:  {CROWDNAV_MODEL_PATH}")
    print(f"  CrowdNav policy: {CROWDNAV_POLICY_CONFIG}")
    print(f"  SARL model:      {SARL_MODEL_PATH}")
    print(f"  SARL policy:     {SARL_POLICY_CONFIG}")
    print("=" * 60)
    print()

    failures = run_sweep(sweep, csv_path, args.num_episodes, args.max_steps)

    print("=" * 60)
    print(f"  Sweep complete.  {total} runs finished.")
    if failures:
        print(f"  {len(failures)} failures: {', '.join(failures)}")
    print(f"  Results saved to: {csv_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
