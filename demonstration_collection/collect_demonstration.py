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

folder_path = "demonstrations"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Joystick setup
pygame.init()
pygame.joystick.init()
if pygame.joystick.get_count() == 0:
    raise RuntimeError("No gamepad detected!")
joystick = pygame.joystick.Joystick(0)
joystick.init()

class Button(IntEnum):
    A = 0,
    B = 1,
    X = 2,
    Y = 3

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
    # Break condition, event get as well
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            break
    # Extraneous error handling
    if not running:
        break

    # Read inputs, apply deadzones
    left_stick_x = apply_deadzone(joystick.get_axis(0), deadzone)
    left_stick_y = apply_deadzone(joystick.get_axis(1), deadzone)
    right_stick_x = apply_deadzone(joystick.get_axis(3), deadzone)
    right_stick_y = apply_deadzone(joystick.get_axis(4), deadzone)
    left_trigger = apply_deadzone(1 + joystick.get_axis(2), deadzone)
    right_trigger = (joystick.get_axis(5) + 1) / 2
    button_A = joystick.get_button(0)           # A button
    button_B = joystick.get_button(1)           # B button
    record_button = joystick.get_button(2)      # X button
    end_button = joystick.get_button(3)         # Y button
    left_bumper = joystick.get_button(4)        # Left Bumper
    right_bumper = joystick.get_button(4)       # Right Bumper

    # Construct action vector based on your controller inputs
    action = np.zeros(env.action_dim)
    # Left joystick controls movement within plane parallel to ground (If robosuite >1.5, otherwise non-composite controller)
    action[0] = left_stick_y * arm_scaling        # (Non-Composite: Base rotation)
    action[1] = left_stick_x * arm_scaling        # (Non-Composite: Shoulder)
    # Right joystick controls elevation and end effector left/right tilt (If robosuite >1.5, otherwise non-composite controller)
    action[2] = -(right_stick_y * wrist_scaling)  # (Non-Composite: Elbow)
    action[3] = right_stick_x * rotation_scaling  # (Non-Composite: Wrist)
    # Left trigger controls end effector forward/backward tilt, left bumper controls tilt direction
    action[4] = (left_trigger * trigger_scaling) * (1 if left_bumper else -1)  # (Non-Composite: Wrist)
    # Right trigger controls gripper
    grip = (actuator_info[gripper_id] - 0.55) / 5.45 # Should convert observed range 0.55-6.0 to 0.0-1.0
    clamped = 1.0 if 1.0 < grip else 0.0 if grip < 0.0 else grip
    grip_ctl = right_trigger - clamped
    if right_trigger > 0.95:
        action[6] = 1.0
    elif abs(grip_ctl) >= 0.1: # Prevent erratic movement
        if grip_ctl < 0:
            action[6] = math.sin(5 * math.pi * grip_ctl / 9 + math.pi / 18)
        else:
            action[6] = math.sin(5 * math.pi * grip_ctl / 9 - math.pi / 18)

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
    if record_button:
        if not button_held[Button.X]:
            button_held[Button.X] = True
            recording = not recording
            start_time = time.time()
            print(f"Recording {'started' if recording else 'ended'}")
    elif button_held[Button.X]:
        button_held[Button.X] = False
    if recording:
        actions.append(action)
        eef_positions.append(eef_pos)
        print("Current robot EE:", obs["robot0_eef_pos"])
        grip_strength.append(clamped)
        print("Current robot grip strength:", clamped)
        timestamps.append(time.time() - start_time)
        
    # Render at the control frequency
    current_time = time.time()
    if current_time - last_render_time >= 1/20:
        env.render()
        last_render_time = current_time
    
    if button_B:
        if not button_held[Button.B]:
            button_held[Button.B] = True
            if recording:
                recording = False
                print(f"Recording scrapped, resetting...")
                actions.clear()
                eef_positions.clear()
                grip_strength.clear()
                timestamps.clear()
                start_time = time.time()
                env.reset()
                actuator_info = env.sim.data.qfrc_actuator
                gripper_id = env.sim.model.actuator_name2id('gripper0_right_finger_1')
    elif button_held[Button.B]:
        button_held[Button.B] = False

    if end_button:
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
