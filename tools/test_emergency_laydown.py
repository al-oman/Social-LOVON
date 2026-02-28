#!/usr/bin/env python3
"""
Test emergency laydown strategies while the robot is moving forward.

Sends a forward Move command, then after a delay, interrupts with one of
several laydown strategies to see which works best as an emergency stop.

Usage:
    python tools/test_emergency_laydown.py                    # default: StandDown
    python tools/test_emergency_laydown.py --method damp
    python tools/test_emergency_laydown.py --method stop_then_down
    python tools/test_emergency_laydown.py --method all       # try each one sequentially

Methods:
    standdown       - StandDown() only
    damp            - Damp() only
    stopmove        - StopMove() only
    stop_then_down  - StopMove() then StandDown()
    stop_then_damp  - StopMove() then Damp()
    all             - Run each method one at a time with recovery between
"""

import time
import sys
import argparse

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
from unitree_sdk2py.go2.sport.sport_client import SportClient


METHODS = ["standdown", "damp", "stopmove", "stop_then_down", "stop_then_damp"]


def run_method(sport_client, method):
    """Execute a single emergency laydown method. Returns description of what was sent."""
    if method == "standdown":
        sport_client.StandDown()
        return "StandDown()"

    elif method == "damp":
        sport_client.Damp()
        return "Damp()"

    elif method == "stopmove":
        sport_client.StopMove()
        return "StopMove()"

    elif method == "stop_then_down":
        sport_client.StopMove()
        time.sleep(0.1)
        sport_client.StandDown()
        return "StopMove() -> StandDown()"

    elif method == "stop_then_damp":
        sport_client.StopMove()
        time.sleep(0.1)
        sport_client.Damp()
        return "StopMove() -> Damp()"


def run_test(sport_client, method, forward_speed, move_duration):
    """Send forward motion, wait, then execute emergency laydown."""
    print(f"\n{'='*60}")
    print(f"  Testing: {method}")
    print(f"  Forward speed: {forward_speed} m/s for {move_duration}s")
    print(f"{'='*60}")

    # Ensure standing first
    print("  [1] Standing up...")
    sport_client.RecoveryStand()
    time.sleep(2.0)

    # Send forward motion
    print(f"  [2] Moving forward at {forward_speed} m/s...")
    sport_client.Move(forward_speed, 0, 0)
    time.sleep(move_duration)

    # Emergency laydown
    print(f"  [3] EMERGENCY LAYDOWN -> {method}")
    t0 = time.time()
    desc = run_method(sport_client, method)
    elapsed = time.time() - t0
    print(f"      Sent: {desc} ({elapsed*1000:.0f}ms)")

    # Wait for robot to settle
    print("  [4] Waiting for robot to settle...")
    time.sleep(3.0)

    print(f"  Done. Observe robot behavior for '{method}'.")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Test emergency laydown strategies on Go2"
    )
    parser.add_argument("--method", type=str, default="standdown",
                        choices=METHODS + ["all"],
                        help="Laydown method to test (default: standdown)")
    parser.add_argument("--speed", type=float, default=0.3,
                        help="Forward speed in m/s (default: 0.3)")
    parser.add_argument("--duration", type=float, default=2.0,
                        help="How long to move forward before laydown in seconds (default: 2.0)")
    parser.add_argument("--network_device", type=str, default="enx00e06c79d1cb",
                        help="Network interface for robot connection")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  EMERGENCY LAYDOWN TEST")
    print("="*60)
    print("\n  WARNING: Ensure clear space around robot!")
    print(f"  Robot will move forward at {args.speed} m/s for {args.duration}s")
    print(f"  then attempt emergency laydown.\n")
    input("  Press Enter to continue (Ctrl+C to abort)...")

    # Initialize
    ChannelFactoryInitialize(0, args.network_device)

    sport_client = SportClient()
    sport_client.SetTimeout(5.0)
    sport_client.Init()
    print("  Sport client initialized.")

    methods = METHODS if args.method == "all" else [args.method]

    try:
        for method in methods:
            run_test(sport_client, method, args.speed, args.duration)

            if args.method == "all" and method != methods[-1]:
                print("  Recovering for next test...")
                sport_client.RecoveryStand()
                time.sleep(3.0)
                input("  Press Enter for next test (Ctrl+C to abort)...")

        print("\nAll tests complete.")

    except KeyboardInterrupt:
        print("\n\nAborted! Stopping robot...")
        sport_client.StopMove()
        sport_client.Damp()
        print("Robot damped.")


if __name__ == "__main__":
    main()
