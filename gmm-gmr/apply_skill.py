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
        print(f"Loaded skill from: {file_path}")
        print(f"Trajectory shape: {trajectory.shape}")
    return times, trajectory

#  Incrementally moves the robot toward the target using delta commands
def move_to_target(env, target, control_interval=0.1, scaling=1.0, acceptance_threshold=0.02, max_steps=100):
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
        action = np.concatenate((delta, fixed_orientation, [0]))
        
        obs, _, _, _ = env.step(action)
        current = np.array(obs["robot0_eef_pos"])
        env.render()
        time.sleep(control_interval) 
        step += 1
    else:
        print("Max steps reached without converging to the target.")

def apply_skill_trajectory(skill_file, control_interval=0.1, scaling=1.0, acceptance_threshold=0.02):
    times, trajectory = load_skill_from_h5(skill_file)
    
    # Create env
    controller_config = suite.load_controller_config(default_controller="OSC_POSE") # Controller
    env = suite.make(
        env_name="Lift",
        robots="UR5e",
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
        controller_configs=controller_config
    )
    
    obs = env.reset()
    env.render()
    time.sleep(1.0)
    
    # Move to the starting point of the trajectory
    starting_point = trajectory[0]
    print("Moving to starting position:", starting_point)
    move_to_target(env, starting_point, control_interval, scaling, acceptance_threshold=0.01)

    time.sleep(2.0)
    
    # Iterate through the trajectory points
    for i, point in enumerate(trajectory):
        print(f"Moving to trajectory point {i}: {point}")
        move_to_target(env, point, control_interval, scaling, acceptance_threshold)
    
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
    apply_skill_trajectory(skill_file_path, control_interval=0.1, scaling=5.0, acceptance_threshold=0.1)
