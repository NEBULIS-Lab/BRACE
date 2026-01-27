from typing import Any, Dict, Union

import os.path as osp
import numpy as np
import sapien
import time
import torch
import yaml
import copy
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig, Camera
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
import robofactory.utils.scenes as scene_rf
from robofactory import CONFIG_DIR
from robofactory.utils.nested_dict_utils import nested_yaml_map, replace_dir

@register_env("StackCube-rf", max_episode_steps=200)
class StackCubeEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["panda_wristcam", "panda", "fetch"]
    agent: Union[Panda, Fetch]
    cube_half_size = torch.tensor([0.02] * 3)

    def __init__(
        self, *args, robot_uids=("panda"), robot_init_qpos_noise=0.02, **kwargs
    ):
        if 'config' in kwargs:
            with open(kwargs['config'], 'r', encoding='utf-8') as f:
                cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
            del kwargs['config']
        else:
            if 'scene' in kwargs:
                scene = kwargs['scene']
                del kwargs['scene']
            else:
                scene = 'table'
            with open(osp.join(CONFIG_DIR, scene, 'stack_cube.yaml'), 'r', encoding='utf-8') as f:
                cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
        self.cfg = nested_yaml_map(replace_dir, cfg)
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        cfg = copy.deepcopy(self.cfg)
        camera_cfg = cfg.get('cameras', {})
        sensor_cfg = camera_cfg.get('sensor', [])
        all_camera_configs =[]
        for sensor in sensor_cfg:
            pose = sensor['pose']
            if pose['type'] == 'pose':
                sensor['pose'] = sapien.Pose(*pose['params'])
            elif pose['type'] == 'look_at':
                sensor['pose'] = sapien_utils.look_at(*pose['params'])
            all_camera_configs.append(CameraConfig(**sensor))
        return all_camera_configs

    @property
    def _default_human_render_camera_configs(self):
        cfg = copy.deepcopy(self.cfg)
        camera_cfg = cfg.get('cameras', {})
        render_cfg = camera_cfg.get('human_render', [])
        all_camera_configs =[]
        for render in render_cfg:
            pose = render['pose']
            if pose['type'] == 'pose':
                render['pose'] = sapien.Pose(*pose['params'])
            elif pose['type'] == 'look_at':
                render['pose'] = sapien_utils.look_at(*pose['params'])
            all_camera_configs.append(CameraConfig(**render))
        return all_camera_configs

    def _load_agent(self, options: dict):
        cfg = copy.deepcopy(self.cfg)
        init_poses = []
        for agent_cfg in cfg['agents']:
            init_poses.append(sapien.Pose(p=agent_cfg['pos']['ppos']['p']))
        super()._load_agent(options, init_poses)

    def _load_scene(self, options: dict):
        cfg = copy.deepcopy(self.cfg)
        scene_name = cfg['scene']['name']
        scene_builder = getattr(scene_rf, f'{scene_name}SceneBuilder')
        self.scene_builder = scene_builder(env=self, cfg=cfg)
        self.scene_builder.build()

    def _setup_sensors(self, options: dict):
        """Override to add wrist cameras as proper sensors during setup."""
        super()._setup_sensors(options)
        agent_count = len(self.cfg.get('agents', []))
        for agent_id in range(agent_count):
            robot = None
            if hasattr(self.agent, 'agents'):
                agents = self.agent.agents
                if isinstance(agents, list) and agent_id < len(agents):
                    robot = agents[agent_id].robot
                elif isinstance(agents, dict) and f'panda-{agent_id}' in agents:
                    robot = agents[f'panda-{agent_id}'].robot
            elif hasattr(self.agent, 'robot'):
                robot = self.agent.robot
            if robot is None:
                continue
            ee_link = None
            for link in robot.get_links():
                if link.name == 'panda_hand':
                    ee_link = link
                    break
            if ee_link is not None:
                camera_uid = f"wrist_camera_agent{agent_id}"
                cam_pose = sapien.Pose(p=[0.05, 0, 0.04], q=[0, 0.707, 0, 0.707])
                wrist_cam_config = CameraConfig(
                    uid=camera_uid, pose=cam_pose, width=320, height=240,
                    near=0.01, far=10, fov=1.5707963268, mount=ee_link,
                )
                wrist_camera = Camera(wrist_cam_config, self.scene, articulation=robot)
                self._sensors[camera_uid] = wrist_camera
        self.scene.sensors = self._sensors

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self.scene_builder.initialize(env_idx)

    def evaluate(self):
        pos_A = self.cubeA.pose.p
        pos_B = self.cubeB.pose.p
        offset = pos_A - pos_B
        xy_flag = (
            torch.linalg.norm(offset[..., :2], axis=1)
            <= torch.linalg.norm(self.cube_half_size[:2]) + 0.005
        )
        z_flag = torch.abs(offset[..., 2] - self.cube_half_size[..., 2] * 2) <= 0.005
        is_cubeA_on_cubeB = torch.logical_and(xy_flag, z_flag)
        is_cubeA_static = self.cubeA.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        is_cubeA_grasped = self.agent.is_grasping(self.cubeA)
        success = is_cubeA_on_cubeB
        return {
            "is_cubeA_grasped": is_cubeA_grasped,
            "is_cubeA_on_cubeB": is_cubeA_on_cubeB,
            "is_cubeA_static": is_cubeA_static,
            "success": success.bool(),
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if "state" in self.obs_mode:
            obs.update(
                cubeA_pose=self.cubeA.pose.raw_pose,
                cubeB_pose=self.cubeB.pose.raw_pose,
                tcp_to_cubeA_pos=self.cubeA.pose.p - self.agent.tcp.pose.p,
                tcp_to_cubeB_pos=self.cubeB.pose.p - self.agent.tcp.pose.p,
                cubeA_to_cubeB_pos=self.cubeB.pose.p - self.cubeA.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return 0

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return 0
