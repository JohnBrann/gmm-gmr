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
from skill import Skill


def build_skill_library(skills_directory):
    skills = {}
    for filename in sorted(os.listdir(skills_directory)):
        if filename.endswith('.h5'):
            skill_path = os.path.join(skills_directory, filename)
            attrs = {}
            with h5py.File(skill_path, "r") as f:
                attrs.update(f.attrs)
                if "skill_name" not in attrs.keys():
                    print(f"Skipping skill from {skill_path} due to missing attributes!")
                    continue
                skill_name = attrs["skill_name"]
                if any(sk.match(skill_name) for sk in skills.values()):
                    print(f"Skipping skill from {skill_path} due to name conflict with existing skill!")
                    continue
                times = np.array(f["times"])
                trajectory = np.array(f["trajectory"])
                skills[skill_name] = Skill(times, trajectory, attrs)
                print(f"Loaded skill from: {skill_path}")
                print(f"Trajectory shape: {trajectory.shape}")
    return skills

def adjust_trajectory(trajectory, target_position):
    """
    Adjust the entire trajectory so that the final waypoint matches the block_position.
    This simply computes the difference between the block_position and the last trajectory
    point and adds that offset to the entire trajectory.
    """
    return trajectory + target_position

def move_to_target(env, target, grip_strength, control_interval=0.1, scaling=1.0, acceptance_threshold=0.02, max_steps=100):
    fixed_orientation = np.zeros(3)
    action_zero = np.zeros(env.action_dim)
    action_zero[6] = grip_strength
    obs, _, _, _ = env.step(action_zero)
    current = np.array(obs["robot0_eef_pos"])
    
    step = 0
    while step < max_steps:
        error = target - current
        err_norm = np.linalg.norm(error)
        if err_norm < acceptance_threshold:
            break
        delta = scaling * error
        action = np.concatenate((delta, fixed_orientation, np.array([grip_strength])))
        obs, _, _, _ = env.step(action)
        current = np.array(obs["robot0_eef_pos"])
        env.render()
        time.sleep(control_interval)
        step += 1
    else:
        print("Max steps reached without converging to the target.")

# Gets the block position 
def get_target_position(env, target):
    if target is None:
        return None
    candidates = [b for b in env.sim.model.body_names if b.startswith(target + "_")]
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one body for {target}, got {candidates!r}")
    body_id = env.sim.model.body_name2id(candidates[0])
    return env.sim.data.body_xpos[body_id].copy()

# Set the block position TODO: make this for every block that spawns in (semi random configuration of 3 blocks)
def set_object_position(env, object_name, new_position):
    candidates = [b for b in env.sim.model.body_names if b.startswith(object_name + "_")]
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one body for {object_name}, got {candidates!r}")
    body_id  = env.sim.model.body_name2id(candidates[0])
    jnt_addr = env.sim.model.body_jntadr[body_id]
    qpos = env.sim.data.qpos.copy()
    qpos[jnt_addr:jnt_addr+3] = new_position
    env.sim.data.qpos[:] = qpos
    env.sim.forward()
    print(f"Manually set {object_name} to {new_position}")

def apply_skill_trajectory(skill, target, control_interval=0.1, scaling=1.0, acceptance_threshold=0.02):
    # Load the learned skill
    times, trajectory = skill.trajectory_data()

    print(f"\n\nApplying skill \'{skill.name()}\' on target \'{'self' if target is None else target}\'\n\n")
    
    target_position = get_target_position(env, target)
    
    # Adjust the skill trajectory based on the updated block position.
    adjusted_trajectory = trajectory if target_position is None else adjust_trajectory(trajectory, target_position)
    
    # Move to the starting point of the adjusted trajectory
    starting_point = adjusted_trajectory[0]
    grip = 1.0 if skill.grip_initial() else -1.0  # Adjust as needed for your gripper configuration
    print("Moving to starting position:", starting_point)
    print("Starting Gripper Strength:", grip)
    move_to_target(env, starting_point, grip, control_interval, scaling, acceptance_threshold=0.01)

    time.sleep(3.0)
    
    # Iterate through the waypoints of the adjusted trajectory
    for i, point in enumerate(adjusted_trajectory):
        print(f"Moving to trajectory point {i}: {point}")
        print(f"Gripper Strength {i}: {grip}")
        move_to_target(env, point, grip, control_interval, scaling, acceptance_threshold)
    
    # Hold the last position
    print("Reached final position.")
    hold_action = np.zeros(env.action_dim)
    hold_action[6] = 1.0 if skill.grip_final() else -1.0
    for _ in range(20):
        _, _, _, _ = env.step(hold_action)
        env.render()
        time.sleep(control_interval)

if __name__ == "__main__":
    skills_dir = "skills"
    if not os.path.exists(skills_dir):
        print("No skills available...")
        sys.exit()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    solution_file = os.path.join(project_root, "tasks", "pick_and_place", "task02.pddl.soln")

    # dynamically populate the environment depending on the expected blocks in the solution file
    with open(solution_file) as f:
        commands = [l.strip() for l in f if l.strip()]
    colors = {cmd.strip("()").split()[1] for cmd in commands if len(cmd.strip("()").split()) > 1}

    RGBA = {
        "red":    [1.0, 0.0, 0.0, 1.0],
        "green":  [0.0, 1.0, 0.0, 1.0],
        "blue":   [0.0, 0.0, 1.0, 1.0],
        "purple": [0.7, 0.0, 1.0, 1.0],
        "orange": [1.0, 0.5, 0.0, 1.0],
    }

    blocks = [
        BoxObject(name=c, size=[0.02]*3, rgba=RGBA[c])
        for c in colors
    ]

    controller_config = suite.load_composite_controller_config(robot="UR5e")
    env = suite.make(
        env_name="PickPlaceCustom",
        robots="UR5e",
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
        ignore_done=True,
        controller_configs=controller_config,
        use_initializer=True,
        blocks=blocks
    )
    obs = env.reset()
    env.render()
    time.sleep(2.0)

    # Allow objects to settle before running
    for _ in range(20):
        _, _, _, _ = env.step(np.zeros(env.action_dim))
        env.render()
        time.sleep(0.1)


    skill_library = build_skill_library(skills_dir)


    # Using planner for picking block TODO: make this better pathing wise and in general all the code
    if not os.path.isfile(solution_file):
        print(f"Solution file not found: {solution_file}")
        sys.exit()

    # read the solution commands
    with open(solution_file, "r") as f:
        commands = [line.strip() for line in f if line.strip()]

    for cmd in commands:
        parts = cmd.strip("()").split()
        if parts[0] not in skill_library.keys():
            print(f"No skill found for PDDL command \'{cmd}\', skipping.")
            continue

        skill = skill_library[parts[0]]
        target = skill.get_target(parts[1:])

        # perform the pick‑and‑place skill
        apply_skill_trajectory(skill, target, control_interval=0.1, scaling=5.0, acceptance_threshold=0.02)

    # Hold for a few seconds before closing
    for _ in range(20):
        _, _, _, _ = env.step(np.zeros(env.action_dim))
        env.render()
        time.sleep(0.1)

    env.close()

