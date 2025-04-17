import sys
sys.path.append('..')
import time
import h5py
import numpy as np
import robosuite as suite
from robosuite.models.objects import BoxObject
from environments import pick_place_custom
from environments.pick_place_custom import PickPlaceCustom
import os
import sys

# Create environment once

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


controller_config = suite.load_composite_controller_config(robot="UR5e")
env = suite.make(
    env_name="PickPlaceCustom",
    robots="UR5e",
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    control_freq=20,
    controller_configs=controller_config,
    blocks=[box_r, box_g, box_b]
)
obs = env.reset()
env.render()
time.sleep(1.0)


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
def get_block_position(env, block_name):
    candidates = [b for b in env.sim.model.body_names if b.startswith(block_name + "_")]
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one body for {block_name}, got {candidates!r}")
    body_id = env.sim.model.body_name2id(candidates[0])
    return env.sim.data.body_xpos[body_id].copy()

# Set the block position TODO: make this for every block that spawns in (semi random configuration of 3 blocks)
def set_block_position(env, block_name, new_position):
    candidates = [b for b in env.sim.model.body_names if b.startswith(block_name + "_")]
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one body for {block_name}, got {candidates!r}")
    body_id  = env.sim.model.body_name2id(candidates[0])
    jnt_addr = env.sim.model.body_jntadr[body_id]
    qpos = env.sim.data.qpos.copy()
    qpos[jnt_addr:jnt_addr+3] = new_position
    env.sim.data.qpos[:] = qpos
    env.sim.forward()
    print(f"Manually set {block_name} to {new_position}")

def apply_skill_trajectory(skill_file, block_name, control_interval=0.1, scaling=1.0, acceptance_threshold=0.02):
    # Load the learned skill
    times, trajectory, grip_strength = load_skill_from_h5(skill_file)
    
    block_position = get_block_position(env, block_name)
    
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

if __name__ == "__main__":
    skills_dir = "skills"
    if not os.path.exists(skills_dir):
        print("No skills available...")
        sys.exit()

    skill_file_path = os.path.join(skills_dir, "skill_1.h5")

    # pick one of: "red-box", "green-box", "blue-box"
    target_color = "green-box"
    # wherever you want to drop it
    drop_pos = np.array([0.1, -0.1, 0.85])
    apply_skill_trajectory(skill_file_path, target_color, control_interval=0.1, scaling=5.0, acceptance_threshold=0.04)

    target_color = "red-box"
    # wherever you want to drop it
    drop_pos = np.array([0.1, -0.1, 0.85])
    apply_skill_trajectory(skill_file_path, target_color, control_interval=0.1, scaling=5.0, acceptance_threshold=0.04)

    target_color = "blue-box"
    # wherever you want to drop it
    drop_pos = np.array([0.1, -0.1, 0.85])
    apply_skill_trajectory(skill_file_path, target_color, control_interval=0.1, scaling=5.0, acceptance_threshold=0.04)

