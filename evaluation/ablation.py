#!/usr/bin/env python3
"""
Social-LOVON VLA hyperparameter ablation study.

Varies one social-navigation parameter at a time while holding the rest
at their defaults.  Every configuration runs 20 episodes (headless).

Usage:
  python evaluation/ablation.py                  # full ablation
  python evaluation/ablation.py --num_episodes 5 # quick test
  python evaluation/ablation.py --params sigma_spread gamma  # subset

Results are saved to evaluation/results/ablation_<timestamp>/
"""

import os
import sys
import re
import subprocess
import tempfile
import datetime
import argparse
import itertools

# ── Paths ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DEPLOY = os.path.join(PROJECT_ROOT, "deploy", "deploy.py")
BASE_ENV_CONFIG = os.path.join(PROJECT_ROOT, "configs", "env_lovon.config")
POLICY_CONFIG = os.path.join(PROJECT_ROOT, "configs", "policy_lovon.config")

# ── Per-run settings ──
DEFAULT_NUM_EPISODES = 10
DEFAULT_MAX_STEPS = 100

# ── Fixed environment for ablation ──
ENV_HUMAN_SPEED = 1.0
ENV_HUMAN_POLICY = "orca"
ROBOT_SPEED = 0.5
TRAJ_DIRECT_STEP = True   # set True to use direct force method instead of normalized heading

# ── Simulation config axes (cross-product with every hyperparam config) ──
SIM_CONFIGS = {
    "robot_theta": [0.7854],
    "human_num":   [2],
}

# ═══════════════════════════════════════════════════════════════════════
#  Ablation axes — one parameter varied at a time
#
#  Each entry: (cli_flag, [values_to_test])
#  The first value in each list should be (or be near) the default.
# ═══════════════════════════════════════════════════════════════════════

ABLATION_AXES = {
    "sigma_spread": {
        "flag": "--safety_sigma_spread",
        "values": [0.01, 0.05,0.1, 0.2],
        "default": 0.1,
    },
    "gamma": {
        "flag": "--safety_gamma",
        "values": [0.90, 0.95, 0.98, 1.0, 1.01],
        "default": 1.01,
    },
    "shield_thresh_on": {
        "flag": "--shield_thresh_on",
        "values": [0.7, 0.8, 0.9 ],
        "default": 0.7,
    },
    "shield_thresh_off": {
        "flag": "--shield_thresh_off",
        "values": [0.8, 0.9, 0.95],
        "default": 0.8,
    },
    "vx_sfm_gain": {
        "flag": "--vx_sfm_gain",
        "values": [4.0, 5.0, 6.0, 7.0],
        "default": 5.0,
    },
    "human_pred_s": {
        "flag": "--human_pred_s",
        "values": [4.0, 6.0, 8.0],
        "default": 5.0,
    },
    "traj_gradient_gain": {
        "flag": "--traj_gradient_gain",
        "values": [1, 2, 3, 4],
        "default": 0.5,
    },
    "traj_step_size": {
        "flag": "--traj_step_size",
        "values": [0.1, 0.2, 0.4],
        "default": 0.2,
    },
    "vx_min": {
        "flag": "--vx_min",
        "values": [0.1, 0.0, -0.1],
        "default": 0.0,
    },
    "traj_goal_gain": {
        "flag": "--traj_goal_gain",
        "values": [1, 2, 3, 4],
        "default": 0.3,
    },
}


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════

def make_env_config(human_num):
    """Create a temporary env config with ablation-fixed overrides."""
    with open(BASE_ENV_CONFIG, "r") as f:
        text = f.read()

    text = re.sub(r"^human_num\s*=.*$", f"human_num = {human_num}",
                  text, flags=re.MULTILINE)
    text = re.sub(r"^v_pref\s*=.*$", f"v_pref = {ENV_HUMAN_SPEED}",
                  text, flags=re.MULTILINE)
    text = re.sub(r"^policy\s*=.*$", f"policy = {ENV_HUMAN_POLICY}",
                  text, flags=re.MULTILINE)

    fd, path = tempfile.mkstemp(prefix="env_ablation_", suffix=".config")
    with os.fdopen(fd, "w") as f:
        f.write(text)
    return path


def run_one(csv_path, num_episodes, max_steps, extra_flags, robot_theta, human_num):
    """Run a single headless evaluation with extra CLI flags."""
    tmp_config = make_env_config(human_num)

    try:
        mission = f"move to the handbag at speed of {ROBOT_SPEED} m/s"

        cmd = [
            sys.executable, DEPLOY,
            "--headless",
            "--num_episodes", str(num_episodes),
            "--max_steps", str(max_steps),
            "--csv_path", csv_path,
            "--mission_instruction", mission,
            "--robot_theta", str(robot_theta),
            "--env_config", tmp_config,
            "--policy_config", POLICY_CONFIG,
            "--robot_policy", "vla",
            "--socialnav_enabled",
        ] + (["--traj_direct_step"] if TRAJ_DIRECT_STEP else []) + extra_flags

        result = subprocess.run(cmd, cwd=PROJECT_ROOT)
        return result.returncode
    finally:
        os.unlink(tmp_config)


def _default_flags():
    flags = []
    for spec in ABLATION_AXES.values():
        flags.extend([spec["flag"], str(spec["default"])])
    return flags


def _sim_combos():
    """All (robot_theta, human_num) combinations from SIM_CONFIGS."""
    return list(itertools.product(
        SIM_CONFIGS["robot_theta"],
        SIM_CONFIGS["human_num"],
    ))


def _expand_sim(hyperparam_sweep):
    """Cross-product a (tag, extra_flags) sweep with all sim config combos."""
    result = []
    for tag, extra_flags in hyperparam_sweep:
        for theta, nhumans in _sim_combos():
            sim_tag = f"{tag}__theta{theta}_h{nhumans}"
            result.append((sim_tag, extra_flags, theta, nhumans))
    return result


def build_ablation_sweep(param_names=None):
    """Build list of (tag, extra_flags, theta, human_num) for each ablation run.

    Varies one parameter at a time, holding others at defaults.
    """
    axes = ABLATION_AXES
    if param_names:
        axes = {k: v for k, v in axes.items() if k in param_names}

    base = _default_flags()
    hyperparam_sweep = [("baseline", base)]

    for param_name, spec in axes.items():
        for val in spec["values"]:
            if val == spec["default"]:
                tag = f"{param_name}_{val}_DEFAULT"
            else:
                tag = f"{param_name}_{val}"
            extra = base + [spec["flag"], str(val)]
            hyperparam_sweep.append((tag, extra))

    return _expand_sim(hyperparam_sweep)


def build_grid_sweep(param_names=None):
    """Build list of (tag, extra_flags, theta, human_num) for every combination."""
    axes = ABLATION_AXES
    if param_names:
        axes = {k: v for k, v in axes.items() if k in param_names}

    names = list(axes.keys())
    specs = [axes[n] for n in names]
    value_lists = [s["values"] for s in specs]

    base = _default_flags()
    hyperparam_sweep = []
    for combo in itertools.product(*value_lists):
        parts = []
        extra = []
        for name, spec, val in zip(names, specs, combo):
            parts.append(f"{name}={val}")
            extra.extend([spec["flag"], str(val)])
        tag = "  ".join(parts)
        hyperparam_sweep.append((tag, base + extra))

    return _expand_sim(hyperparam_sweep)


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Social-LOVON VLA hyperparameter ablation study"
    )
    parser.add_argument("--num_episodes", type=int, default=DEFAULT_NUM_EPISODES,
                        help=f"Episodes per configuration (default: {DEFAULT_NUM_EPISODES})")
    parser.add_argument("--max_steps", type=int, default=DEFAULT_MAX_STEPS,
                        help=f"Max sim steps per episode (default: {DEFAULT_MAX_STEPS})")
    parser.add_argument("--params", nargs="+", default=None,
                        choices=list(ABLATION_AXES.keys()),
                        help="Only ablate these parameters (default: all)")
    parser.add_argument("--grid", action="store_true",
                        help="Test all combinations of param values instead of one-at-a-time. "
                             "Use with --params to keep the count tractable.")
    args = parser.parse_args()

    if args.grid:
        sweep = build_grid_sweep(args.params)
    else:
        sweep = build_ablation_sweep(args.params)
    total = len(sweep)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_tag = "grid" if args.grid else "ablation"
    results_dir = os.path.join(SCRIPT_DIR, "results", f"{mode_tag}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "ablation_results.csv")

    param_label = ", ".join(args.params) if args.params else "all"
    mode_label = "grid search" if args.grid else "ablation (one-at-a-time)"

    print("=" * 60)
    print(f"  Social-LOVON VLA {mode_label}")
    print(f"  Parameters: {param_label}")
    print(f"  {total} configurations x {args.num_episodes} episodes each")
    print(f"  Fixed env: speed {ENV_HUMAN_SPEED}, policy {ENV_HUMAN_POLICY}")
    print(f"  Sim configs: theta {SIM_CONFIGS['robot_theta']}, "
          f"human_num {SIM_CONFIGS['human_num']}")
    print(f"  Results: {results_dir}")
    print("=" * 60)
    print()

    failures = []
    for i, (tag, extra_flags, theta, human_num) in enumerate(sweep):
        pct = 100 * i // total
        print("-" * 56)
        print(f"  [{i+1}/{total}] ({pct}%)  {tag}")
        if extra_flags:
            print(f"    {' '.join(extra_flags)}")
        print("-" * 56)

        rc = run_one(csv_path, args.num_episodes, args.max_steps, extra_flags, theta, human_num)
        if rc != 0:
            failures.append(tag)
            print(f"  WARNING: exited with code {rc}")
        print()

    print("=" * 60)
    print(f"  Ablation complete.  {total} runs finished.")
    if failures:
        print(f"  {len(failures)} failures: {', '.join(failures)}")
    print(f"  Results saved to: {csv_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
