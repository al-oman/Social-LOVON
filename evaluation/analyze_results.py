#!/usr/bin/env python3
"""
Analyze eval_results.csv from a Social-LOVON evaluation sweep.

Usage:
    python evaluation/analyze_results.py [path_to_eval_results.csv] [--best-combo]

Displays success/collision/danger metrics grouped by socialnav enabled vs
disabled, then breaks down by each sweep variable.
"""

import sys
import os
import argparse
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
    near_miss = grp["avg_near_miss"].mean() if "avg_near_miss" in grp.columns else float("nan")
    min_d = grp["avg_min_distance"].mean()
    steps = grp["avg_steps"].mean()
    goal_d = grp["avg_goal_dist"].mean()
    return (
        f"  {label:<32s}  n={n:>3d}  "
        f"success={color_pct(succ)}  "
        f"collision={color_pct(coll, invert=True)}  "
        f"danger={fmt_float(danger, '.1f'):>5s}  "
        f"near_miss={fmt_float(near_miss, '.1f'):>5s}  "
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

METRIC_COLS = {"success_rate", "collision_rate", "avg_min_distance",
               "avg_danger_count", "avg_near_miss", "avg_steps",
               "avg_sim_time", "avg_goal_dist", "wall_time", "timestamp",
               "num_episodes", "mission_instruction", "robot_speed",
               "human_num"}

def _varying_param_cols(df):
    return [c for c in df.columns if c not in METRIC_COLS and df[c].nunique() > 1]

def find_best_combo(df):
    """Rank all tested parameter combinations by a composite score."""
    param_cols = _varying_param_cols(df)

    if not param_cols:
        print(f"\n  {YELLOW}No varying parameters found — only one configuration in data.{RESET}")
        return

    grouped = df.groupby(param_cols, dropna=False).agg(
        success_rate=("success_rate", "mean"),
        collision_rate=("collision_rate", "mean"),
        avg_danger_count=("avg_danger_count", "mean"),
        avg_min_distance=("avg_min_distance", "mean"),
        avg_steps=("avg_steps", "mean"),
        avg_goal_dist=("avg_goal_dist", "mean"),
        n=("success_rate", "count"),
    ).reset_index()

    # Composite score: reward success & safety distance, penalise collisions & danger
    grouped["score"] = (
        grouped["success_rate"]
        - grouped["collision_rate"]
        + 0.1 * grouped["avg_min_distance"]
        - 0.01 * grouped["avg_danger_count"]
    )
    grouped = grouped.sort_values("score", ascending=False).reset_index(drop=True)

    section("Best Parameter Combinations (by composite score)")
    print(f"  {DIM}score = success - collision + 0.1*min_dist - 0.01*danger{RESET}")
    print()
    header_line()
    n_show = min(10, len(grouped))
    for i in range(n_show):
        row = grouped.iloc[i]
        parts = [f"{c}={row[c]}" for c in param_cols]
        label = "  ".join(parts)
        # Truncate long labels
        if len(label) > 30:
            label = label[:27] + "..."
        score_str = f"{BOLD}{row['score']:+.3f}{RESET}"
        print(
            f"  {f'#{i+1}':<4s} {label:<30s}  n={int(row['n']):>3d}  "
            f"success={color_pct(row['success_rate'])}  "
            f"collision={color_pct(row['collision_rate'], invert=True)}  "
            f"danger={fmt_float(row['avg_danger_count'], '.1f'):>5s}  "
            f"min_dist={fmt_float(row['avg_min_distance']):>5s}  "
            f"steps={fmt_float(row['avg_steps'], '.0f'):>4s}  "
            f"goal_dist={fmt_float(row['avg_goal_dist'])}  "
            f"score={score_str}"
        )

    # Print full param breakdown of #1
    print()
    best = grouped.iloc[0]
    section("Best Combination — Full Parameters")
    for c in param_cols:
        print(f"  {c:<25s} = {best[c]}")
    print(f"  {'score':<25s} = {best['score']:.4f}")


def main():
    cli = argparse.ArgumentParser(
        description="Analyze eval_results.csv from a Social-LOVON evaluation sweep."
    )
    cli.add_argument("csv", nargs="?", default=None,
                     help="Path to eval_results.csv (auto-detects most recent if omitted)")
    cli.add_argument("--best-combo", action="store_true",
                     help="Find the highest-scoring parameter combination and exit")
    args = cli.parse_args()

    if args.csv:
        csv_path = args.csv
    else:
        # Try to find the most recent results CSV
        eval_dir = os.path.join(os.path.dirname(__file__), "results")
        csv_path = None
        if os.path.isdir(eval_dir):
            for d in sorted(os.listdir(eval_dir), reverse=True):
                for name in ("eval_results.csv", "ablation_results.csv"):
                    candidate = os.path.join(eval_dir, d, name)
                    if os.path.isfile(candidate):
                        csv_path = candidate
                        break
                if csv_path:
                    break
        if csv_path is None:
            cli.error("No eval_results.csv or ablation_results.csv found in evaluation/results/")

    df = pd.read_csv(csv_path)
    print(f"\n{BOLD}Loaded {len(df)} configurations from:{RESET} {csv_path}")

    # Infer robot speed early so it's available for best-combo
    if "mission_instruction" in df.columns:
        df["robot_speed"] = df["mission_instruction"].str.extract(r"(\d+\.?\d*)\s*m/s").astype(float)

    if args.best_combo:
        find_best_combo(df)
        print()
        return

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

    # ── 2. All configurations ─────────────────────────────────────────
    param_cols = _varying_param_cols(df)
    section("All Configurations")
    header_line()
    if param_cols:
        grouped = df.groupby(param_cols, dropna=False)
        for key, grp in grouped:
            vals = key if isinstance(key, tuple) else (key,)
            label = "  ".join(f"{c}={v}" for c, v in zip(param_cols, vals))
            print(summary_row(label, grp))
    else:
        print(summary_row("all", df))

    # ── 3. Worst / best configs ───────────────────────────────────────
    param_cols = _varying_param_cols(df)

    def row_tag(row):
        parts = [f"{c}={row[c]}" for c in param_cols if c in row.index]
        return "  ".join(parts)

    section("Top 5 Best Configs (by success rate, then min danger)")
    best = df.nlargest(5, ["success_rate", "avg_min_distance"])
    for _, row in best.iterrows():
        print(summary_row(row_tag(row), pd.DataFrame([row])))

    section("Top 10 Worst Configs (by success rate, then max danger)")
    worst = df.nsmallest(10, ["success_rate", "avg_min_distance"])
    for _, row in worst.iterrows():
        print(summary_row(row_tag(row), pd.DataFrame([row])))

    print()


if __name__ == "__main__":
    main()
