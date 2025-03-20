import time
import h5py
import numpy as np
import robosuite as suite

def load_skill_from_h5(file_path):
    """
    Loads the skill trajectory from an HDF5 file.
    
    Returns:
        times (np.array): Array of time stamps.
        trajectory (np.array): Array of end-effector positions (x, y, z) for each time step.
    """
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
    
    # create env
    controller_config = suite.load_controller_config(default_controller="OSC_POSE")
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
    
    # move to the starting point of the trajectory.
    starting_point = trajectory[0]
    print("Moving to starting point:", starting_point)
    move_to_target(env, starting_point, control_interval, scaling, acceptance_threshold)
    
    # Iterate through the trajectory points.
    for i, point in enumerate(trajectory):
        print(f"Moving to trajectory point {i}: {point}")
        move_to_target(env, point, control_interval, scaling, acceptance_threshold)
    
    # stay at last postition of trajectory
    print("At final position.")
    hold_action = np.zeros(env.action_dim)
    for _ in range(20):
        obs, _, _, _ = env.step(hold_action)
        env.render()
        time.sleep(control_interval)
    
    time.sleep(2)
    env.close()

if __name__ == "__main__":
    skill_file_path = "skills/skill_1.h5"
    apply_skill_trajectory(skill_file_path, control_interval=0.1, scaling=5.0, acceptance_threshold=0.1)
