from dataclasses import dataclass

@dataclass
class TrajectoryData:
    times: list
    trajectory: list
    gr_initial: bool = False,
    gr_final: bool = False

class Skill:
    def __init__(self, timestamps, trajectory, attributes):
        self._traj_data = TrajectoryData(timestamps, trajectory, attributes["grip_initial"], attributes["grip_final"])
        self._name = attributes["skill_name"]
        self._target = attributes["target_idx"]

    def name(self):
        return self._name

    def match(self, pddl_action):
        return pddl_action == self._name

    def get_target(self, pddl_action_params):
        return None if self._target < 0 else pddl_action_params[self._target]

    def trajectory_data(self):
        return self._traj_data.times, self._traj_data.trajectory

    def grip_initial(self):
        return self._traj_data.gr_initial

    def grip_final(self):
        return self._traj_data.gr_final
