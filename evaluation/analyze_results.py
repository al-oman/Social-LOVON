#!/usr/bin/env python3
"""
Analyze eval_results.csv from a Social-LOVON evaluation sweep.

Usage:
    python evaluation/analyze_results.py <path_to_eval_results.csv>

Displays success/collision/danger metrics grouped by socialnav enabled vs
disabled, then breaks down by each sweep variable.
"""

import sys
import os
import pandas as pd
import numpy as np

# ── Formatting helpers ────────────────────────────────────────────────

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
RESET = "\033[0m"

def color_pct(val, good_thresh=0.8, bad_thresh=0.4, invert=False):
    """Color a 0-1 value as a percentage string."""
    if invert:
        good_thresh, bad_thresh = 1 - good_thresh, 1 - bad_thresh
        color = GREEN if val <= good_thresh else (YELLOW if val <= bad_thresh else RED)
    else:
        color = GREEN if val >= good_thresh else (YELLOW if val >= bad_thresh else RED)
    return f"{color}{val*100:6.1f}%{RESET}"

def fmt_float(val, fmt=".2f"):
    return f"{val:{fmt}}"

def summary_row(label, grp):
    """Return a formatted summary string for a group of rows."""
    n = len(grp)
    succ = grp["success_rate"].mean()
    coll = grp["collision_rate"].mean()
    danger = grp["avg_danger_count"].mean()
    min_d = grp["avg_min_distance"].mean()
    steps = grp["avg_steps"].mean()
    goal_d = grp["avg_goal_dist"].mean()
    return (
        f"  {label:<32s}  n={n:>3d}  "
        f"success={color_pct(succ)}  "
        f"collision={color_pct(coll, invert=True)}  "
        f"danger={fmt_float(danger, '.1f'):>5s}  "
        f"min_dist={fmt_float(min_d):>5s}  "
        f"steps={fmt_float(steps, '.0f'):>4s}  "
        f"goal_dist={fmt_float(goal_d)}"
    )

def section(title):
    w = 100
    print(f"\n{BOLD}{CYAN}{'─'*w}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'─'*w}{RESET}")

def header_line():
    print(
        f"  {'':32s}  {'':>5s}  "
        f"{'success':>13s}  "
        f"{'collision':>15s}  "
        f"{'danger':>6s}  "
        f"{'min_d':>8s}  "
        f"{'steps':>5s}  "
        f"{'goal_d':>9s}"
    )

# ── Main ──────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        # Try to find the most recent results CSV
        eval_dir = os.path.join(os.path.dirname(__file__), "results")
        if os.path.isdir(eval_dir):
            subdirs = sorted(os.listdir(eval_dir))
            for d in reversed(subdirs):
                candidate = os.path.join(eval_dir, d, "eval_results.csv")
                if os.path.isfile(candidate):
                    csv_path = candidate
                    break
            else:
                print(f"Usage: {sys.argv[0]} <eval_results.csv>")
                sys.exit(1)
        else:
            print(f"Usage: {sys.argv[0]} <eval_results.csv>")
            sys.exit(1)
    else:
        csv_path = sys.argv[1]

    df = pd.read_csv(csv_path)
    print(f"\n{BOLD}Loaded {len(df)} configurations from:{RESET} {csv_path}")

    # ── 1. Overall: socialnav ON vs OFF ───────────────────────────────
    section("Social Navigation: ENABLED vs DISABLED")
    header_line()
    for enabled, grp in df.groupby("socialnav_enabled"):
        label = f"socialnav={'ON' if enabled else 'OFF'}"
        print(summary_row(label, grp))

    # Compute deltas
    on = df[df["socialnav_enabled"] == True]
    off = df[df["socialnav_enabled"] == False]
    if len(on) > 0 and len(off) > 0:
        ds = on["success_rate"].mean() - off["success_rate"].mean()
        dc = on["collision_rate"].mean() - off["collision_rate"].mean()
        dd = on["avg_danger_count"].mean() - off["avg_danger_count"].mean()
        dm = on["avg_min_distance"].mean() - off["avg_min_distance"].mean()
        print(f"\n  {BOLD}Delta (ON - OFF):{RESET}  "
              f"success={ds:+.1%}  collision={dc:+.1%}  "
              f"danger={dd:+.1f}  min_dist={dm:+.3f}")

    # ── 1b. By robot policy (if column exists) ─────────────────────────
    if "robot_policy" in df.columns and df["robot_policy"].nunique() > 1:
        section("By Robot Policy")
        header_line()
        for pol in sorted(df["robot_policy"].unique()):
            grp = df[df["robot_policy"] == pol]
            print(summary_row(f"policy={pol}", grp))

        # Further: robot_policy × socialnav
        section("Robot Policy x SocialNav")
        header_line()
        for pol in sorted(df["robot_policy"].unique()):
            subset = df[df["robot_policy"] == pol]
            for sn_enabled in [False, True]:
                grp = subset[subset["socialnav_enabled"] == sn_enabled]
                if len(grp) == 0:
                    continue
                sn_tag = "ON " if sn_enabled else "OFF"
                print(summary_row(f"{pol}  sn={sn_tag}", grp))
            sn_on = subset[subset["socialnav_enabled"] == True]
            sn_off = subset[subset["socialnav_enabled"] == False]
            if len(sn_on) > 0 and len(sn_off) > 0:
                ds = sn_on["success_rate"].mean() - sn_off["success_rate"].mean()
                dc = sn_on["collision_rate"].mean() - sn_off["collision_rate"].mean()
                dd = sn_on["avg_danger_count"].mean() - sn_off["avg_danger_count"].mean()
                sign_s = GREEN if ds > 0 else (RED if ds < 0 else DIM)
                sign_c = GREEN if dc < 0 else (RED if dc > 0 else DIM)
                print(f"  {'  Δ (ON-OFF)':<32s}        "
                      f"{sign_s}{ds:+6.1%}{RESET}         "
                      f"{sign_c}{dc:+6.1%}{RESET}  "
                      f"{dd:+5.1f}")
            print()

    # ── 2. Breakdown by each sweep variable, split by socialnav ───────
    sweep_cols = [
        ("robot_policy",    "Robot Policy"),
        ("human_policy",    "Human Policy"),
        ("human_num",       "Human Count"),
        ("human_v_pref",    "Human Speed"),
        ("robot_theta",     "Robot Theta"),
        ("human_traj_pred", "Traj Prediction"),
    ]

    # Infer robot speed from mission instruction
    if "mission_instruction" in df.columns:
        df["robot_speed"] = df["mission_instruction"].str.extract(r"(\d+\.?\d*)\s*m/s").astype(float)
        sweep_cols.append(("robot_speed", "Robot Speed"))

    for col, label in sweep_cols:
        if col not in df.columns:
            continue
        section(f"By {label} (socialnav ON vs OFF)")
        header_line()
        for val in sorted(df[col].unique()):
            subset = df[df[col] == val]
            for sn_enabled in [False, True]:
                grp = subset[subset["socialnav_enabled"] == sn_enabled]
                if len(grp) == 0:
                    continue
                sn_tag = "ON " if sn_enabled else "OFF"
                row_label = f"{label}={val}  sn={sn_tag}"
                print(summary_row(row_label, grp))
            # delta line
            sn_on = subset[subset["socialnav_enabled"] == True]
            sn_off = subset[subset["socialnav_enabled"] == False]
            if len(sn_on) > 0 and len(sn_off) > 0:
                ds = sn_on["success_rate"].mean() - sn_off["success_rate"].mean()
                dc = sn_on["collision_rate"].mean() - sn_off["collision_rate"].mean()
                dd = sn_on["avg_danger_count"].mean() - sn_off["avg_danger_count"].mean()
                sign_s = GREEN if ds > 0 else (RED if ds < 0 else DIM)
                sign_c = GREEN if dc < 0 else (RED if dc > 0 else DIM)
                print(f"  {'  Δ (ON-OFF)':<32s}        "
                      f"{sign_s}{ds:+6.1%}{RESET}         "
                      f"{sign_c}{dc:+6.1%}{RESET}  "
                      f"{dd:+5.1f}")
            print()

    # ── 3. Worst / best configs ───────────────────────────────────────
    section("Top 5 Best Configs (by success rate, then min danger)")
    best = df.nlargest(5, ["success_rate", "avg_min_distance"])
    for _, row in best.iterrows():
        sn = "ON" if row["socialnav_enabled"] else "OFF"
        tp = "ON" if row["human_traj_pred"] else "OFF"
        tag = (f"sn={sn} h={int(row['human_num'])} "
               f"hspd={row['human_v_pref']} {row['human_policy']} tp={tp}")
        print(summary_row(tag, pd.DataFrame([row])))

    section("Top 10 Worst Configs (by success rate, then max danger)")
    worst = df.nsmallest(10, ["success_rate", "avg_min_distance"])
    for _, row in worst.iterrows():
        sn = "ON" if row["socialnav_enabled"] else "OFF"
        tp = "ON" if row["human_traj_pred"] else "OFF"
        tag = (f"sn={sn} h={int(row['human_num'])} "
               f"hspd={row['human_v_pref']} {row['human_policy']} tp={tp}")
        print(summary_row(tag, pd.DataFrame([row])))

    print()


if __name__ == "__main__":
    main()
