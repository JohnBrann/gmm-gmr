import sys
sys.path.append('..')
import time
import h5py
import numpy as np
import robosuite as suite
from robosuite.models.objects import BoxObject
from environments import pick_place_custom
import os
import sys

def load_skill_from_h5(file_path):
    with h5py.File(file_path, "r") as f:
        times = np.array(f["times"])
        trajectory = np.array(f["trajectory"])
        grip_strength = np.array(f["grip_strength"])
        print(f"Loaded skill from: {file_path}")
        print(f"Trajectory shape: {trajectory.shape}")
        print(f"Gripper Strength shape: {grip_strength.shape}")
    return times, trajectory, grip_strength

#  Incrementally moves the robot toward the target using delta commands
def move_to_target(env, target, grip_strength_target, control_interval=0.1, scaling=1.0, acceptance_threshold=0.02, max_steps=100):
    fixed_orientation = np.zeros(3)
    
    action_zero = np.zeros(env.action_dim)
    obs, _, _, _ = env.step(action_zero)
    current = np.array(obs["robot0_eef_pos"])
    
    step = 0
    while step < max_steps:
        error = target - current
        err_norm = np.linalg.norm(error)
        # print(f"Step {step}: Target: {target}, Current: {current}, Error norm: {err_norm:.5f}")
        if err_norm < acceptance_threshold:
            break
        delta = scaling * error

        grip_target_scalar = float(np.squeeze(grip_strength_target))
        
        # Now all arrays are 1D: delta and fixed_orientation have shape (3,)
        # and np.array([grip_target_scalar]) has shape (1,)
        action = np.concatenate((delta, fixed_orientation, np.array([grip_target_scalar])))


        # action = np.concatenate((delta, fixed_orientation, [grip_strength_target]))
        
        obs, _, _, _ = env.step(action)
        current = np.array(obs["robot0_eef_pos"])
        env.render()
        time.sleep(control_interval) 
        step += 1
    else:
        print("Max steps reached without converging to the target.")

def apply_skill_trajectory(skill_file, control_interval=0.1, scaling=1.0, acceptance_threshold=0.02):
    times, trajectory, grip_strength = load_skill_from_h5(skill_file)
    
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
    controller_config = suite.load_composite_controller_config(robot="UR5e") # Controller
    # Instruction commented out below may be useful, but further testing is needed
    # controller_config["body_parts"]["right"]["input_ref_frame"] = "world"
    print(controller_config)
    env = suite.make(
        env_name="PickPlaceCustom",
        robots="UR5e",
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
        controller_configs=controller_config,
        use_initializer=True,
        blocks=[box_r, box_g, box_b]
    )
    
    obs = env.reset()
    env.render()
    time.sleep(1.0)
    
    # Move to the starting point of the trajectory
    starting_point = trajectory[0]
    starting_grip = -0.00001 # TODO Gripper strength is discrete! I think... find solution to this. clip continuous strength?
    # starting_grip = grip_strength[0]
    print("Moving to starting position:", starting_point)
    print("Starting Gripper Strength:", starting_grip)
    move_to_target(env, starting_point, starting_grip, control_interval, scaling, acceptance_threshold=0.01)

    time.sleep(3.0)
    
    # Iterate through the trajectory points
    for i, point in enumerate(trajectory):
        print(f"Moving to trajectory point {i}: {point}")
        print(f"Gripper Strength {i}: {grip_strength[i]}")
        move_to_target(env, point, grip_strength[i], control_interval, scaling, acceptance_threshold)
    
    # Stay at last position of trajectory
    print("Reached final position.")
    # Keep arm in place at end
    hold_action = np.zeros(env.action_dim) 
    for _ in range(20):
        _, _, _, _ = env.step(hold_action)
        env.render()
        time.sleep(control_interval)
    
    time.sleep(2)
    env.close()

if __name__ == "__main__":
    skills_dir = "skills"
    if not os.path.exists(skills_dir):
        print("No skills available...")
        sys.exit()

    skill_file_path = os.path.join(skills_dir, "skill_1.h5")
    apply_skill_trajectory(skill_file_path, control_interval=0.1, scaling=5.0, acceptance_threshold=0.05)
