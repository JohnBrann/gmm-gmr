import sys
sys.path.append('..')
import os
import re
import robosuite as suite
from robosuite.models.objects import BoxObject
from environments import pick_place_custom
import mujoco
import pygame
import h5py
import numpy as np
import time
import math
from enum import IntEnum
from dataclasses import dataclass

folder_path = "demonstrations"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Joystick setup
found_controller = False
pygame.init()
pygame.joystick.init()
if pygame.joystick.get_count() != 0:
    found_controller = True
    joystick = pygame.joystick.Joystick(0)
    joystick.init()

# Controls Class
@dataclass
class Controls:
    translate: tuple = (0.0, 0.0, 0.0)
    wrist: tuple = (0.0, 0.0)
    gripper: float = 0.0
    demo_toggle: bool = False
    demo_finish: bool = False
    demo_scrap: bool = False
    
    def __str__(this):
        yes = "\033[92mYES\033[0m"
        no = "\033[91mNO\033[0m"
        return (" ─────────────── Controller State ───────────────"
        f"\n   Translation:  ({this.translate[0]:.5f}, {this.translate[1]:.5f}, {this.translate[2]:.5f})"
        f"\n   Wrist:        ({this.wrist[0]:.5f}, {this.wrist[1]:.5f})"
        f"\n   Gripper:      ({this.gripper:.5f})"
        f"\n   Toggle Demo:  {yes if this.demo_toggle else no}"
        f"\n   Finish Demo:  {yes if this.demo_finish else no}"
        f"\n   Scrap Demo:   {yes if this.demo_scrap else no}"
        "\n ────────────────────────────────────────────────")

# Enum for tracking held buttons/keys
# (Also used to match controller button to functionality)
class Input(IntEnum):
    NOT_BOUND = 0,
    SCRAP_DEMO = 1,
    TOGGLE_DEMO = 2,
    FINISH_DEMO = 3

# Function for making console output a bit nicer
def print_inplace(output, clear_precise=False, margin_lower=1, margin_upper=1):
    line_count = output.count('\n') + 1
    height = margin_upper + line_count + margin_lower
    upper = "\n" * margin_upper
    lower = "\n" * margin_lower
    if height > print_inplace.inplace_line_count:
        for i in range(height - print_inplace.inplace_line_count + 1):
            print("\n", end="")
            print_inplace.inplace_line_count = height
    # ANSI nonsense follows; this might or might not work on other operating systems
    if clear_precise:
        print(f"\033[{height}F", end="")
        for i in range(line_count + margin_upper):
            print(f"\033[2K")
        print(f"\033[{line_count + margin_upper}F{upper}{output}\n{lower}", end=f"\033[{height}E", flush=True)
    else:
        print(f"\033[{height}F\033[0J{upper}{output}\n{lower}", end=f"\033[{height}E", flush=True)
print_inplace.inplace_line_count = 0

# Function to print some stuff above the in-place log
def above_inplace(output):
    margin = "\n" * output.count('\n')
    height = print_inplace.inplace_line_count
    print(f"\033[{height}F{output}", end=f"\033[0K\033[{height}E\n{margin}", flush=True)

# Create cubes
box_r = BoxObject(
    name="red-box",
    size=[0.02, 0.02, 0.02],
    rgba=[1, 0, 0, 1]
)
box_g = BoxObject(
    name="green-box",
    size=[0.02, 0.02, 0.02],
    rgba=[0, 1, 0, 1]
)
box_b = BoxObject(
    name="blue-box",
    size=[0.02, 0.02, 0.02],
    rgba=[0, 0, 1, 1]
)

# Create env
env = suite.make(
    env_name="PickPlaceCustom",
    robots="UR5e",
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    control_freq=20,
    initialization_noise=None,
    # Horizon is length of sim, we set it absurdly high since we want
    # the user to end the demo manually using controller input
    horizon=5000000,
    use_initializer=True,
    blocks=[box_r, box_g, box_b]
)

obs = env.reset()
running = True
start_time = time.time()
last_render_time = start_time
env.visualize(vis_settings = { "robots": False, "grippers": True, "env": False })

actuator_info = env.sim.data.qfrc_actuator
gripper_id = env.sim.model.actuator_name2id('gripper0_right_finger_1')
grip_strength = []
eef_positions = [] 
actions = []
timestamps = []

# Sensitivity multipliers (may change with different controllers)
arm_scaling = 0.125
wrist_scaling = 0.1
rotation_scaling = 0.1
trigger_scaling = 0.03
deadzone = 0.1

def apply_deadzone(value, threshold):
    if abs(value) < threshold:
        return 0.0
    return value * (abs(value) - threshold) / (1 - threshold)  # llm generated smoothing

button_held = [False] * 4
recording = False

# Demonstration loop
while running:
    # Prepare controls for input
    controls = Controls()
    
    # Break condition, event get as well
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            break
    # Extraneous error handling
    if not running:
        break

    # Construct action vector based on inputs
    action = np.zeros(env.action_dim)

    # If using controller, read controller inputs and apply deadzones
    if found_controller:
        left_stick_x = apply_deadzone(joystick.get_axis(0), deadzone)
        left_stick_y = apply_deadzone(joystick.get_axis(1), deadzone)
        right_stick_x = apply_deadzone(joystick.get_axis(3), deadzone)
        right_stick_y = apply_deadzone(joystick.get_axis(4), deadzone)
        left_trigger = apply_deadzone(1 + joystick.get_axis(2), deadzone)
        right_trigger = (joystick.get_axis(5) + 1) / 2
        button_A = joystick.get_button(Input.NOT_BOUND)               # A button
        controls.demo_scrap = joystick.get_button(Input.SCRAP_DEMO)   # B button
        controls.demo_toggle = joystick.get_button(Input.TOGGLE_DEMO) # X button
        controls.demo_finish = joystick.get_button(Input.FINISH_DEMO) # Y button
        left_bumper = joystick.get_button(4)  # Left Bumper
        right_bumper = joystick.get_button(4) # Right Bumper
        
        controls.translate = (left_stick_y * arm_scaling, left_stick_x * arm_scaling, -(right_stick_y * wrist_scaling))
        controls.wrist = (right_stick_x * rotation_scaling, (left_trigger * trigger_scaling) * (1 if left_bumper else -1))
        grip = (actuator_info[gripper_id] - 0.55) / 5.45 # Should convert observed range 0.55-6.0 to 0.0-1.0
        clamped = 1.0 if 1.0 < grip else 0.0 if grip < 0.0 else grip
        grip_ctl = right_trigger - clamped
        if right_trigger > 0.95:
            controls.gripper = 1.0
        elif abs(grip_ctl) >= 0.1: # Prevent erratic movement
            if grip_ctl < 0:
                controls.gripper = math.sin(5 * math.pi * grip_ctl / 9 + math.pi / 18)
            else:
                controls.gripper = math.sin(5 * math.pi * grip_ctl / 9 - math.pi / 18)


    # Left joystick controls movement within plane parallel to ground (If robosuite >1.5, otherwise non-composite controller)
    action[0] = controls.translate[0] # (Non-Composite: Base rotation)
    action[1] = controls.translate[1] # (Non-Composite: Shoulder)
    # Right joystick controls elevation and end effector left/right tilt (If robosuite >1.5, otherwise non-composite controller)
    action[2] = controls.translate[2] # (Non-Composite: Elbow)
    action[3] = controls.wrist[0]     # (Non-Composite: Wrist)
    # Left trigger controls end effector forward/backward tilt, left bumper controls tilt direction
    action[4] = controls.wrist[1]     # (Non-Composite: Wrist)
    # Right trigger controls gripper
    action[6] = controls.gripper

    # Step the environment with the computed action
    obs, reward, done, info = env.step(action)
    
    # Retrieve the end effector position from the observation
    # Adjust the key if needed; typically "robot0_eef_pos" or "eef_pos"
    eef_pos = obs.get("robot0_eef_pos", None)
    if eef_pos is None:
        eef_pos = obs.get("eef_pos", None)
    if eef_pos is None:
        raise ValueError("End-effector position not found in observation!")
    
    # Append the data for this timestep, if recording
    if controls.demo_toggle:
        if not button_held[Input.TOGGLE_DEMO]:
            button_held[Input.TOGGLE_DEMO] = True
            recording = not recording
            start_time = time.time()
            above_inplace(f"Recording {'started' if recording else 'ended'}")
    elif button_held[Input.TOGGLE_DEMO]:
        button_held[Input.TOGGLE_DEMO] = False
    if recording:
        actions.append(action)
        eef_positions.append(eef_pos)
        above_inplace(f"Current robot EE: {obs['robot0_eef_pos']}")
        grip_strength.append(clamped)
        above_inplace(f"Current robot grip strength: {clamped}")
        timestamps.append(time.time() - start_time)
        
    # Render at the control frequency
    current_time = time.time()
    if current_time - last_render_time >= 1/20:
        env.render()
        last_render_time = current_time
    
    # Print controller state at bottom of terminal
    print_inplace(f"{controls}")
    
    if controls.demo_scrap:
        if not button_held[Input.SCRAP_DEMO]:
            button_held[Input.SCRAP_DEMO] = True
            if recording:
                recording = False
                print(f"Recording scrapped, resetting...\n\n")
                print_inplace.inplace_line_count = 0
                actions.clear()
                eef_positions.clear()
                grip_strength.clear()
                timestamps.clear()
                start_time = time.time()
                env.reset()
                env.visualize(vis_settings = { "robots": False, "grippers": True, "env": False })
                actuator_info = env.sim.data.qfrc_actuator
                gripper_id = env.sim.model.actuator_name2id('gripper0_right_finger_1')
    elif button_held[Input.SCRAP_DEMO]:
        button_held[Input.SCRAP_DEMO] = False

    if controls.demo_finish:
        done = True
    if done:
        print("Episode complete")
        running = False
        obs = env.reset()

print("Reached end, closing...")
env.close()
pygame.quit()

# Determine next available file number in the demonstrations folder
existing_files = os.listdir(folder_path)
pattern = r"demo_(\d+)\.h5"
max_num = 0
for filename in existing_files:
    m = re.match(pattern, filename)
    if m:
        num = int(m.group(1))
        if num > max_num:
            max_num = num
new_file_num = max_num + 1
new_file_name = f"demo_{new_file_num}.h5"
full_path = os.path.join(folder_path, new_file_name)

# Save the demonstration data to the new file
with h5py.File(full_path, "w") as f:
    print("Writing demonstration data...")
    f.attrs["env_name"] = "Lift"
    f.attrs["robot"] = "UR5e"
    f.attrs["control_freq"] = env.control_freq
    f.create_dataset("timestamps", data=np.array(timestamps))
    f.create_dataset("grip_strength", data=np.array(grip_strength))
    f.create_dataset("eef_positions", data=np.array(eef_positions))
    f.create_dataset("actions", data=np.array(actions))

print(f"Trajectory saved to {full_path}")
