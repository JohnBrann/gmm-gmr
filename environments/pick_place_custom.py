from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena, BinsArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial # idk
from robosuite.utils.observables import Observable, sensor # probably
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import convert_quat # probably
from .arenas.sort_arena import SortArena
import numpy as np

class PickPlaceCustom(ManipulationEnv):
    def __init__(
        self,
        # ManipulationEnv Stuff
        robots,
        env_configuration="default",
        controller_configs=None,
        base_types="default",
        gripper_types="default",
        initialization_noise=None,
        use_camera_obs=True,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mjviewer",
        renderer_config=None,
        seed=None,
        # Custom Stuff
        obj_initializer=None,
        blocks=None
    ):
        # Set up default block if necessary
        if blocks is None:
            self.blocks = BoxObject(
                name="red-box",
                size=[0.02, 0.02, 0.02],
                rgba=[1, 0, 0, 1]
            )
        else:
            self.blocks = blocks
        # Set up default object initializer if necessary
        if obj_initializer is None:
            # Add blocks to sampler
            self.obj_initializer = UniformRandomSampler(
                name="CubeSampler",
                mujoco_objects=self.blocks,
                x_range=[-0.08, 0.08],
                y_range=[-0.08, 0.08],
                rotation_axis='z',
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=[0, 0, 0.8],
                z_offset=0.01
            )
        else:
            # Load configured initializer
            self.obj_initializer = obj_initializer
            # Add blocks to initializer
            self.obj_initializer.mujoco_objects.add_objects(self.blocks)
        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )
    
    def reward(self, action=None):
        # We don't currently use reward, but might in future
        return 1.0
    
    def _load_model(self):
        super()._load_model()
        
        self.robots[0].robot_model.set_base_xpos(self.robots[0].robot_model.base_xpos_offset["table"](0.8))
        
        self.model = ManipulationTask(
            mujoco_arena=SortArena(),
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.blocks
        )
    
    def _reset_internal(self):
        super()._reset_internal()
        
        if not self.deterministic_reset:
            placements = self.obj_initializer.sample()
            for pos, quat, block in placements.values():
                self.sim.data.set_joint_qpos(block.joints[0], np.concatenate([np.array(pos), np.array(quat)]))
