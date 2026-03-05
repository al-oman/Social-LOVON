import time
import sys
import select
import termios
import tty
import threading
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.go2.robot_state.robot_state_client import RobotStateClient


MOTOR_NAMES = ["FR_hip", "FR_thigh", "FR_calf",
               "FL_hip", "FL_thigh", "FL_calf",
               "RR_hip", "RR_thigh", "RR_calf",
               "RL_hip", "RL_thigh", "RL_calf"]

class RobotState:
    def __init__(self):
        self.battery_voltage = 0.0
        self.battery_current = 0.0
        self.motor_temps = [0] * 12
        self.motor_positions = [0.0] * 12
        self.last_update = time.time()

robot_state = RobotState()

def lowstate_handler(msg: LowState_):
    global robot_state
    robot_state.battery_voltage = msg.power_v
    robot_state.battery_current = msg.power_a
    for i in range(12):
        robot_state.motor_positions[i] = msg.motor_state[i].q
        robot_state.motor_temps[i] = msg.motor_state[i].temperature
    robot_state.last_update = time.time()

def print_status():
    print("\n" + "="*60)
    print(f"Battery: {robot_state.battery_voltage:.2f}V @ {robot_state.battery_current:.2f}A")
    print("-"*60)
    print("Motor Temperatures:")
    for name, temp in zip(MOTOR_NAMES, robot_state.motor_temps):
        print(f"  {name:>10}: {temp}°C")
    age = time.time() - robot_state.last_update
    if age > 1.0:
        print(f" Data age: {age:.1f}s")
    print("="*60)

def print_help():
    print("\n" + "="*60)
    print("WASD CONTROLS (hold key, release to stop):")
    print("  W - Forward        S - Backward")
    print("  A - Strafe Left    D - Strafe Right")
    print("  Q - Turn Left      E - Turn Right")
    print("")
    print("OTHER COMMANDS (press once):")
    print("  u - Stand Up       U - Balance Stand")
    print("  l - Lay Down       L - Damp")
    print("  r - Recovery Stand")
    print("  i - Show Status")
    print("  h - Show this Help")
    print("  x - Exit")
    print("="*60 + "\n")


def get_pressed_keys():
    """Read all currently buffered keys from stdin (non-blocking)."""
    keys = set()
    while select.select([sys.stdin], [], [], 0)[0]:
        ch = sys.stdin.read(1)
        keys.add(ch)
    return keys


def main():
    ChannelFactoryInitialize(0, "enx00e06c79d1cb")

    print("Initializing state monitor...")
    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(lowstate_handler, 10)

    rsc = RobotStateClient()
    rsc.SetTimeout(3.0)
    rsc.Init()

    print("Initializing sport client...")
    sport_client = SportClient()
    sport_client.SetTimeout(5.0)
    sport_client.Init()

    print("\n Robot interface initialized!")
    print_help()

    linear_speed = 0.3   # m/s
    strafe_speed = 0.3   # m/s
    angular_speed = 1.0  # rad/s
    cmd_rate = 0.05      # 20 Hz command loop

    MOVE_KEYS = {'w', 'a', 's', 'd', 'q', 'e'}
    KEY_TIMEOUT = 0.6   # seconds — covers terminal initial key-repeat delay (~500ms)

    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())

        active_keys = {}  # key → timestamp of last seen

        while True:
            keys = get_pressed_keys()
            now = time.time()

            if 'x' in keys:
                print("\n→ Stopping robot and exiting...")
                sport_client.StopMove()
                break

            # One-shot commands
            if 'u' in keys:
                print("→ Standing up...")
                sport_client.StandUp()
            if 'U' in keys:
                print("→ Balance stand...")
                sport_client.BalanceStand()
            if 'l' in keys:
                print("→ Laying down...")
                sport_client.StandDown()
            if 'L' in keys:
                print("→ Damping...")
                sport_client.Damp()
            if 'r' in keys:
                print("→ Recovery stand...")
                sport_client.RecoveryStand()
            if 'i' in keys:
                print_status()
            if 'h' in keys:
                print_help()

            # Update timestamps for movement keys
            for k in keys & MOVE_KEYS:
                active_keys[k] = now

            # Expire keys not seen recently (key released)
            active_keys = {k: t for k, t in active_keys.items()
                           if now - t < KEY_TIMEOUT}

            # Build velocity from active keys
            vx, vy, wz = 0.0, 0.0, 0.0
            if 'w' in active_keys:
                vx += linear_speed
            if 's' in active_keys:
                vx -= linear_speed
            if 'a' in active_keys:
                vy += strafe_speed
            if 'd' in active_keys:
                vy -= strafe_speed
            if 'q' in active_keys:
                wz += angular_speed
            if 'e' in active_keys:
                wz -= angular_speed

            sport_client.Move(vx, vy, wz)

            time.sleep(cmd_rate)

    except KeyboardInterrupt:
        print("\n\n→ Ctrl+C detected. Stopping robot...")
        sport_client.StopMove()
        print("Exited safely.")
    except Exception as e:
        print(f"\n  Error: {e}")
        sport_client.StopMove()
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  UNITREE GO2 WASD CONTROLLER")
    print("="*60)
    print("\n  WARNING: Ensure clear space around robot!")
    input("Press Enter to continue...")

    main()
