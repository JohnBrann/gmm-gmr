import time
import numpy as np
import robosuite as suite

def move_to_target(env, target, control_interval=0.1, scaling=1.0, threshold=0.005, max_steps=100):
    """
    Incrementally move the robot to the target using delta commands until
    the error is below a threshold or max_steps is reached.
    """
    fixed_orientation = np.zeros(3)
    
    # Initialize current observation by taking a zero-action step.
    action_zero = np.zeros(env.action_dim)
    obs, _, _, _ = env.step(action_zero)
    current = np.array(obs["robot0_eef_pos"])
    
    step = 0
    while step < max_steps:
        error = target - current
        err_norm = np.linalg.norm(error)
        print(f"Step {step}: Target: {target}, Current: {current}, Error norm: {err_norm:.5f}")
        if err_norm < threshold:
            print("Target reached with error norm below threshold.")
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

def multi_point_test(control_interval=0.1, scaling=1.0):
    # Load OSC_POSE controller configuration.
    controller_config = suite.load_controller_config(default_controller="OSC_POSE")
    # Create the environment.
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
    
    # Get the current end-effector position.
    current_pos = np.array(obs["robot0_eef_pos"])
    print("Starting position:", current_pos)
    
    fixed_orientation = np.zeros(3)
    
    # Define two target positions relative to the current position.
    target_A = current_pos + np.array([0.1, 0, 0])
    target_B = target_A + np.array([0, 0.1, 0])
    
    # Move to Target A.
    print("Moving to Target A:", target_A)
    move_to_target(env, target_A, control_interval, scaling)
    
    # Hold at Target A for 2 seconds.
    print("Holding at Target A")
    hold_action = np.zeros(env.action_dim)
    for i in range(20):
        obs, _, _, _ = env.step(hold_action)
        env.render()
        time.sleep(control_interval)
    
    # Move to Target B.
    print("Moving to Target B:", target_B)
    move_to_target(env, target_B, control_interval, scaling)
    
    # Hold at Target B for 2 seconds.
    print("Holding at Target B")
    for i in range(20):
        obs, _, _, _ = env.step(hold_action)
        env.render()
        time.sleep(control_interval)
    
    time.sleep(2)
    env.close()

if __name__ == "__main__":
    # Try with a higher scaling factor (e.g. 1.0) to command larger movements.
    multi_point_test(control_interval=0.1, scaling=1.0)
