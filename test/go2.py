import time
import sys
import threading
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.go2.robot_state.robot_state_client import RobotStateClient


# Robot state storage
class RobotState:
    def __init__(self):
        self.battery_voltage = 0.0
        self.battery_current = 0.0
        self.temperature = 0.0
        self.motor_positions = [0.0] * 12
        self.last_update = time.time()
       
robot_state = RobotState()

def lowstate_handler(msg: LowState_):
    """Handle incoming low-level state messages"""
    global robot_state
   
    robot_state.battery_voltage = msg.power_v
    robot_state.battery_current = msg.power_a
   
    # Store motor data
    for i in range(12):
        robot_state.motor_positions[i] = msg.motor_state[i].q

   
    # IMU temperature (if available)
    if hasattr(msg, 'imu_state'):
        robot_state.temperature = msg.imu_state.temperature
   
    robot_state.last_update = time.time()

# def rsc_handler():
#     global robot_state
#     robot_state.mode = 

def print_status():
    """Print current robot status"""
    print("\n" + "="*60)
    print(f"Battery: {robot_state.battery_voltage:.2f}V @ {robot_state.battery_current:.2f}A")
    print(f"Temperature: {robot_state.temperature:.1f}°C")
      
    # Show if data is stale
    age = time.time() - robot_state.last_update
    if age > 1.0:
        print(f" Data age: {age:.1f}s")
    print("="*60)

def print_help():
    """Print control instructions"""
    print("\n" + "="*60)
    print("KEYBOARD CONTROLS:")
    print("  w - Move Forward")
    print("  s - Move Backward")
    print("  a - Turn Left")
    print("  d - Turn Right")
    print("  q - Stop Movement")
    print("  u - Stand Up")
    print("  U - Stand (Balance)")
    print("  l - Lay Down")
    print("  L - Lay Down (Damp)")
    print("  r - Recovery Stand")
    print("  i - Show Status")
    print("  h - Show this Help")
    print("  x - Exit")
    print("="*60 + "\n")

def main():

    ChannelFactoryInitialize(0, "enp8s0")
   
    # Initialize subscriber for monitoring
    print("Initializing state monitor...")
    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(lowstate_handler, 10)

    # Initialiing Robot State Client
    rsc = RobotStateClient()
    rsc.SetTimeout(3.0)
    rsc.Init()
   
    # Initialize sport client for control
    print("Initializing sport client...")
    sport_client = SportClient()
    sport_client.SetTimeout(5.0)
    sport_client.Init()
   
    print("\n Robot interface initialized!")
    print_help()
   
    # Movement speed settings
    linear_speed = 0.3  # m/s
    angular_speed = 1.0  # rad/s
   
    try:
        while True:
            # Get user input
            cmd = input("Command (h for help): ").strip()
           
            if cmd == 'w':
                print("→ Moving forward...")
                ret = sport_client.Move(linear_speed, 0, 0)
                print(f"   Result: {ret}")
               
            elif cmd == 's':
                print("→ Moving backward...")
                ret = sport_client.Move(-linear_speed, 0, 0)
                print(f"   Result: {ret}")
               
            elif cmd == 'a':
                print("→ Turning left...")
                ret = sport_client.Move(0, 0, angular_speed)
                print(f"   Result: {ret}")
               
            elif cmd == 'd':
                print("→ Turning right...")
                ret = sport_client.Move(0, 0, -angular_speed)
                print(f"   Result: {ret}")
               
            elif cmd == 'q':
                print("→ Stopping movement...")
                ret = sport_client.StopMove()
                print(f"   Result: {ret}")
               
            elif cmd == 'u':
                print("→ Standing up...")
                ret = sport_client.StandUp()
                print(f"   Result: {ret}")

            elif cmd == 'U':
                print("→ Standing up (balancing)...")
                ret = sport_client.BalanceStand()
                print(f"   Result: {ret}")
               
            elif cmd == 'l':
                print("→ Laying down...")
                ret = sport_client.StandDown()
                print(f"   Result: {ret}")

            elif cmd == 'L':
                print("→ Laying down (damping)...")
                ret = sport_client.Damp()
                print(f"   Result: {ret}")
               
            elif cmd == 'r':
                print("→ Recovery stand...")
                ret = sport_client.RecoveryStand()
                print(f"   Result: {ret}")
               
            elif cmd == 'i':
                print_status()
               
            elif cmd == 'h':
                print_help()
               
            elif cmd == 'x':
                print("\n→ Stopping robot and exiting...")
                sport_client.StopMove()
                break
               
            else:
                print(f"Unknown command: '{cmd}'. Press 'h' for help.")
           
            time.sleep(0.1)
   
    except KeyboardInterrupt:
        print("\n\n→ Ctrl+C detected. Stopping robot...")
        sport_client.StopMove()
        print("✓ Exited safely.")
    except Exception as e:
        print(f"\n  Error: {e}")
        sport_client.StopMove()

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  UNITREE GO2 CONTROLLER")
    print("="*60)
    print("\n  WARNING: Ensure clear space around robot!")
    input("Press Enter to continue...")
   
    main()