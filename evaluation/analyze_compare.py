#!/usr/bin/env python3
"""
Side-by-side comparison table for Social-LOVON evaluation results.

Displays SARL / ORCA / VLA+snOFF / VLA+snON as main column groups,
each with success, collision, danger, and steps sub-columns.
Rows are grouped by environment attributes (human count, speed, policy, theta).

Usage:
    python evaluation/analyze_compare.py [path_to_eval_results.csv]
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np

# ── ANSI helpers ──────────────────────────────────────────────────────

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
RESET = "\033[0m"


def color_pct(val):
    """Color a 0-1 value as a percentage."""
    if np.isnan(val):
        return f"{'—':>6s}"
    color = GREEN if val >= 0.8 else (YELLOW if val >= 0.4 else RED)
    return f"{color}{val*100:5.1f}%{RESET}"


def color_pct_inv(val):
    """Color a 0-1 value (lower is better)."""
    if np.isnan(val):
        return f"{'—':>6s}"
    color = GREEN if val <= 0.2 else (YELLOW if val <= 0.6 else RED)
    return f"{color}{val*100:5.1f}%{RESET}"


def fmt_num(val, width=5, decimals=1):
    if np.isnan(val):
        return f"{'—':>{width}s}"
    return f"{val:>{width}.{decimals}f}"


# ── Condition classification ──────────────────────────────────────────

CONDITIONS = [
    ("SARL",  lambda df: df["robot_policy"] == "sarl"),
    ("ORCA",  lambda df: df["robot_policy"] == "orca"),
    ("snOFF", lambda df: (df["robot_policy"] == "vla") & (df["socialnav_enabled"] == False)),
    ("snON",  lambda df: (df["robot_policy"] == "vla") & (df["socialnav_enabled"] == True)),
]

METRICS = ["success_rate", "collision_rate", "avg_danger_count", "avg_near_miss", "avg_min_distance", "avg_steps"]
METRIC_HEADERS = ["succ", "coll", "dngr", "nmiss", "dmin", "steps"]


# ── Table rendering ──────────────────────────────────────────────────

def format_metrics(row):
    """Format a single condition's metrics into a fixed-width string."""
    if row is None:
        return f"{'—':>6s} {'—':>6s} {'—':>5s} {'—':>5s} {'—':>5s} {'—':>5s}"
    nm = fmt_num(row.get('avg_near_miss', float('nan')))
    return (
        f"{color_pct(row['success_rate'])} "
        f"{color_pct_inv(row['collision_rate'])} "
        f"{fmt_num(row['avg_danger_count'])} "
        f"{nm} "
        f"{fmt_num(row['avg_min_distance'], width=5, decimals=2)} "
        f"{fmt_num(row['avg_steps'], width=5, decimals=0)}"
    )


def print_table(df):
    # ── Identify environment attribute columns ──
    env_cols = []
    for col, label in [("human_num", "h"), ("human_v_pref", "hspd"),
                        ("human_policy", "h_pol"), ("robot_theta", "theta")]:
        if col in df.columns and df[col].nunique() > 1:
            env_cols.append((col, label))

    # If no varying env cols, use a single summary row
    if not env_cols:
        env_cols = [("human_num", "h")]

    # ── Build row groups ──
    group_keys = [c for c, _ in env_cols]
    groups = df.groupby(group_keys, dropna=False)

    # ── Header ──
    attr_width = 30
    cond_width = 42  # per condition (raw, before ANSI)

    # Condition group header
    header_top = f"{'':>{attr_width}s}"
    for cond_name, _ in CONDITIONS:
        header_top += f"  │ {BOLD}{cond_name:^{cond_width}s}{RESET}"
    print()
    print(header_top)

    # Sub-headers (metric names)
    sub_header = f"{'':>{attr_width}s}"
    metric_hdr = f"{'succ':>6s} {'coll':>6s} {'dngr':>5s} {'nmiss':>5s} {'dmin':>5s} {'steps':>5s}"
    for _ in CONDITIONS:
        sub_header += f"  │ {DIM}{metric_hdr}{RESET}"
    print(sub_header)

    # ── Averages row ──
    sep = "─" * attr_width
    for _ in CONDITIONS:
        sep += "──┼" + "─" * cond_width
    print(sep)

    avg_label = f"{BOLD}{'AVERAGE':>{attr_width}s}{RESET}"
    avg_row = avg_label
    for _, cond_filter in CONDITIONS:
        subset = df[cond_filter(df)]
        if len(subset) == 0:
            avg_row += f"  │ {format_metrics(None)}"
        else:
            avg_row += f"  │ {format_metrics(subset[METRICS].mean())}"
    print(avg_row)
    print(sep)

    # ── Data rows ──
    for group_vals, group_df in sorted(groups, key=lambda x: x[0] if isinstance(x[0], tuple) else (x[0],)):
        if not isinstance(group_vals, tuple):
            group_vals = (group_vals,)

        # Build attribute label
        parts = []
        for (col, label), val in zip(env_cols, group_vals):
            if col == "robot_theta":
                parts.append(f"θ={val:.2f}")
            elif col == "human_num":
                parts.append(f"h={int(val)}")
            elif col == "human_v_pref":
                parts.append(f"hspd={val}")
            else:
                parts.append(f"{label}={val}")
        attr_str = f"{'  '.join(parts):>{attr_width}s}"

        row_str = attr_str
        for _, cond_filter in CONDITIONS:
            subset = group_df[cond_filter(group_df)]
            if len(subset) == 0:
                row_str += f"  │ {format_metrics(None)}"
            else:
                row_str += f"  │ {format_metrics(subset[METRICS].mean())}"
        print(row_str)

    print(sep)
    print()


# ── Main ──────────────────────────────────────────────────────────────

def main():
    cli = argparse.ArgumentParser(
        description="Side-by-side comparison table for Social-LOVON eval results."
    )
    cli.add_argument("csv", nargs="?", default=None,
                     help="Path to eval_results.csv (auto-detects most recent if omitted)")
    args = cli.parse_args()

    if args.csv:
        csv_path = args.csv
    else:
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
            cli.error("No results CSV found in evaluation/results/")

    df = pd.read_csv(csv_path)
    print(f"\n{BOLD}Loaded {len(df)} rows from:{RESET} {csv_path}")

    # Detect which conditions are present
    present = []
    for name, cond_filter in CONDITIONS:
        if len(df[cond_filter(df)]) > 0:
            present.append(name)
    print(f"{DIM}Conditions found: {', '.join(present)}{RESET}")

    print_table(df)


if __name__ == "__main__":
    main()
