# import os
# import re
# import robosuite as suite
# import pygame
# import h5py
# import numpy as np
# import time

# # Ensure the demonstrations folder exists
# folder_path = "demonstrations"
# if not os.path.exists(folder_path):
#     os.makedirs(folder_path)

# # joystick setup
# pygame.init()
# pygame.joystick.init()
# if pygame.joystick.get_count() == 0:
#     raise RuntimeError("No gamepad detected!")
# joystick = pygame.joystick.Joystick(0)
# joystick.init()

# # suite setup
# env = suite.make(
#     env_name="Lift",
#     robots="UR5e",
#     has_renderer=True,
#     has_offscreen_renderer=False,
#     use_camera_obs=False,
#     control_freq=20,
#     initialization_noise=None,
#     # horizon is length of sim, measured in 1/100 of a second
#     horizon=2000,
# )

# # setup
# obs = env.reset()
# running = True
# start_time = time.time()
# last_render_time = start_time

# states = []
# actions = []
# timestamps = []

# # sensitivity multipliers (may change with different controllers)
# arm_scaling = 0.125
# wrist_scaling = 0.1
# rotation_scaling = 0.1
# trigger_scaling = 0.03

# deadzone = 0.1  

# def apply_deadzone(value, threshold):
#     if abs(value) < threshold:
#         return 0.0
#     return value * (abs(value) - threshold) / (1 - threshold)  # simple smoothing

# # loop for collecting demonstration data
# while running:
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             running = False
#             break

#     if not running:
#         break

#     # Read joystick inputs and apply deadzone
#     left_stick_x = apply_deadzone(joystick.get_axis(0), deadzone)
#     left_stick_y = apply_deadzone(joystick.get_axis(1), deadzone)
#     right_stick_x = apply_deadzone(joystick.get_axis(3), deadzone)
#     right_stick_y = apply_deadzone(joystick.get_axis(4), deadzone)
#     left_trigger = apply_deadzone(1 + joystick.get_axis(2), deadzone)
#     right_trigger = apply_deadzone(1 + joystick.get_axis(5), deadzone)
#     grip_button_close = joystick.get_button(0)  # A button
#     grip_button_open = joystick.get_button(1)   # B button
    
#     action = np.zeros(env.action_dim)
#     action[0] = right_stick_y * arm_scaling     # base rotation
#     action[1] = right_stick_x * arm_scaling     # shoulder
#     action[2] = -(left_stick_y * wrist_scaling)  # elbow
#     action[3] = left_stick_x * rotation_scaling  # wrist
#     action[4] = -(left_trigger * trigger_scaling) + (right_trigger * trigger_scaling)  # wrist

#     if grip_button_close:
#         action[-1] = 1.0
#     elif grip_button_open:
#         action[-1] = -1.0
    
#     # step the environment
#     obs, reward, done, info = env.step(action)
#     state = env.sim.get_state().flatten()

#     actions.append(action)
#     states.append(state)
#     timestamps.append(time.time() - start_time)

#     current_time = time.time()
#     if current_time - last_render_time >= 1/20:
#         env.render()
#         last_render_time = current_time
        
#     if done:
#         print("Episode complete")
#         running = False
#         obs = env.reset()

# print("Reached end, closing...")
# env.close()
# pygame.quit()

# # Determine the next available file number in the demonstrations folder
# existing_files = os.listdir(folder_path)
# pattern = r"demo_(\d+)\.h5"
# max_num = 0
# for filename in existing_files:
#     m = re.match(pattern, filename)
#     if m:
#         num = int(m.group(1))
#         if num > max_num:
#             max_num = num
# new_file_num = max_num + 1
# new_file_name = f"demo_{new_file_num}.h5"
# full_path = os.path.join(folder_path, new_file_name)

# # Save the demonstration to the new file
# with h5py.File(full_path, "w") as f:
#     print("Reached demo write")
#     f.attrs["env_name"] = "Lift"
#     f.attrs["robot"] = "UR5e"
#     f.attrs["control_freq"] = env.control_freq
#     f.create_dataset("timestamps", data=np.array(timestamps))
#     f.create_dataset("states", data=np.array(states))
#     f.create_dataset("actions", data=np.array(actions))

# print(f"Trajectory saved to {full_path}")



import os
import re
import robosuite as suite
import pygame
import h5py
import numpy as np
import time

folder_path = "demonstrations"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# joystick setup
pygame.init()
pygame.joystick.init()
if pygame.joystick.get_count() == 0:
    raise RuntimeError("No gamepad detected!")
joystick = pygame.joystick.Joystick(0)
joystick.init()

# env setup
env = suite.make(
    env_name="Lift",
    robots="UR5e",
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    control_freq=20,
    initialization_noise=None,
    # horizon is length of sim, measured in 1/100 of a second
    horizon=2000,  
)

obs = env.reset()
running = True
start_time = time.time()
last_render_time = start_time

eef_positions = [] 
actions = []
timestamps = []

# sensitivity multipliers (may change with different controllers)
arm_scaling = 0.125
wrist_scaling = 0.1
rotation_scaling = 0.1
trigger_scaling = 0.03
deadzone = 0.1  

def apply_deadzone(value, threshold):
    if abs(value) < threshold:
        return 0.0
    return value * (abs(value) - threshold) / (1 - threshold)  # llm generated smoothing

# # loop
while running:
    # break condition, event get as well
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            break
     # extraneous error handling
    if not running:
        break

    # read inputs, apply deadzones
    left_stick_x = apply_deadzone(joystick.get_axis(0), deadzone)
    left_stick_y = apply_deadzone(joystick.get_axis(1), deadzone)
    right_stick_x = apply_deadzone(joystick.get_axis(3), deadzone)
    right_stick_y = apply_deadzone(joystick.get_axis(4), deadzone)
    left_trigger = apply_deadzone(1 + joystick.get_axis(2), deadzone)
    right_trigger = apply_deadzone(1 + joystick.get_axis(5), deadzone)
    grip_button_close = joystick.get_button(0)  # A button
    grip_button_open = joystick.get_button(1)   # B button

    # Construct action vector based on your controller inputs
    action = np.zeros(env.action_dim)
    action[0] = right_stick_y * arm_scaling     # base rotation
    action[1] = right_stick_x * arm_scaling     # shoulder
    action[2] = -(left_stick_y * wrist_scaling)  # elbow
    action[3] = left_stick_x * rotation_scaling  # wrist
    action[4] = -(left_trigger * trigger_scaling) + (right_trigger * trigger_scaling)  # wrist movement

    if grip_button_close:
        action[-1] = 1.0
    elif grip_button_open:
        action[-1] = -1.0

    # Step the environment with the computed action
    obs, reward, done, info = env.step(action)
    
    # Retrieve the end-effector position from the observation
    # Adjust the key if needed; typically "robot0_eef_pos" or "eef_pos"
    eef_pos = obs.get("robot0_eef_pos", None)
    if eef_pos is None:
        eef_pos = obs.get("eef_pos", None)
    if eef_pos is None:
        raise ValueError("End-effector position not found in observation!")
    
    # Append the data for this timestep
    actions.append(action)
    eef_positions.append(eef_pos)
    print("Current robot EE:", obs["robot0_eef_pos"])
    # print("Commanded EE:", desired_abs)

    timestamps.append(time.time() - start_time)

    # Render at the control frequency
    current_time = time.time()
    if current_time - last_render_time >= 1/20:
        env.render()
        last_render_time = current_time

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
    f.create_dataset("eef_positions", data=np.array(eef_positions))
    f.create_dataset("actions", data=np.array(actions))

print(f"Trajectory saved to {full_path}")
