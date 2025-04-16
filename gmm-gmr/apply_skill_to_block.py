import time
import h5py
import numpy as np
import robosuite as suite
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

def adjust_trajectory(trajectory, block_position):
    """
    Adjust the entire trajectory so that the final waypoint matches the block_position.
    This simply computes the difference between the block_position and the last trajectory
    point and adds that offset to the entire trajectory.
    """
    original_final_point = trajectory[-1]
    offset = block_position - original_final_point
    print("Original final trajectory point:", original_final_point)
    print("Observed block position:", block_position)
    print("Computed offset:", offset)
    adjusted_trajectory = trajectory + offset
    return adjusted_trajectory

def move_to_target(env, target, grip_strength_target, control_interval=0.1, scaling=1.0, acceptance_threshold=0.02, max_steps=100):
    fixed_orientation = np.zeros(3)
    action_zero = np.zeros(env.action_dim)
    obs, _, _, _ = env.step(action_zero)
    current = np.array(obs["robot0_eef_pos"])
    
    step = 0
    while step < max_steps:
        error = target - current
        err_norm = np.linalg.norm(error)
        if err_norm < acceptance_threshold:
            break
        delta = scaling * error
        grip_target_scalar = float(np.squeeze(grip_strength_target))
        action = np.concatenate((delta, fixed_orientation, np.array([grip_target_scalar])))
        obs, _, _, _ = env.step(action)
        current = np.array(obs["robot0_eef_pos"])
        env.render()
        time.sleep(control_interval)
        step += 1
    else:
        print("Max steps reached without converging to the target.")

# Gets the block position 
def get_block_position(obs):
    # TODO: get posiiton of specified block (red, blue, ect.)
    block_position = np.array(obs["cube_pos"]) 
    
    return block_position

# Set the block position TODO: make this for every block that spawns in (semi random configuration of 3 blocks)
def set_block_position(env, new_position):
    try:
        body_id = env.sim.model.body_name2id("cube_main")
    except Exception as e:
        print("Error accessing the cube body. Please verify the body name in your model:", e)
        return

    # Get the qpos index corresponding to the body
    qpos_index = env.sim.model.body_jntadr[body_id]
    current_qpos = env.sim.data.qpos.copy()
    # Set the position (assumed to be the first three values)
    current_qpos[qpos_index:qpos_index+3] = new_position
    env.sim.data.qpos[:] = current_qpos
    env.sim.forward()
    print("Block position manually set to:", new_position)

def apply_skill_trajectory(skill_file, control_interval=0.1, scaling=1.0, acceptance_threshold=0.02):
    # Load the learned skill
    times, trajectory, grip_strength = load_skill_from_h5(skill_file)
    
    # Create environment using robosuite
    controller_config = suite.load_composite_controller_config(robot="UR5e")
    print("Controller configuration:", controller_config)
    env = suite.make(
        env_name="Lift",
        robots="UR5e",
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
        controller_configs=controller_config
    )
    
    # Reset environment and render
    obs = env.reset()
    env.render()
    time.sleep(1.0)
    
    try:
        block_position = get_block_position(obs)
    except KeyError as e:
        print("Error retrieving block position:", e)
        env.close()
        sys.exit(1)
    
    print("Block position before manual update:", block_position)
    
    # Manually set the block position to the desired coordinates
    desired_block_position = [0.01686829, -0.12840486, 0.83049447]  # Change these values as needed
    set_block_position(env, desired_block_position)
    
    # Take an extra step to update the observation after state modification.
    obs, _, _, _ = env.step(np.zeros(env.action_dim))
    block_position = get_block_position(obs)
    print("Block position after manual update:", block_position)
    env.render()
    time.sleep(1.0)
    
    # Adjust the skill trajectory based on the updated block position.
    adjusted_trajectory = adjust_trajectory(trajectory, block_position)
    
    # Move to the starting point of the adjusted trajectory
    starting_point = adjusted_trajectory[0]
    starting_grip = -0.00001  # Adjust as needed for your gripper configuration
    print("Moving to starting position:", starting_point)
    print("Starting Gripper Strength:", starting_grip)
    move_to_target(env, starting_point, starting_grip, control_interval, scaling, acceptance_threshold=0.01)

    time.sleep(3.0)
    
    # Iterate through the waypoints of the adjusted trajectory
    for i, point in enumerate(adjusted_trajectory):
        print(f"Moving to trajectory point {i}: {point}")
        print(f"Gripper Strength {i}: {grip_strength[i]}")
        move_to_target(env, point, grip_strength[i], control_interval, scaling, acceptance_threshold)
    
    # Hold the last position
    print("Reached final position.")
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
    apply_skill_trajectory(skill_file_path, control_interval=0.1, scaling=5.0, acceptance_threshold=0.04)
