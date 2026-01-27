from typing import Any, Dict, Union

import os.path as osp
import numpy as np
import sapien
import torch
import json
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

@register_env("PickMeat-rf", max_episode_steps=500)
class PickMeatEnv(BaseEnv):

    SUPPORTED_ROBOTS = ["panda"]
    agent: Union[Panda, Fetch]
    goal_thresh = 0.025
    cube_color = np.concatenate((np.array([187, 116, 175]) / 255, [1]))
    cube_half_size = 0.02

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
            with open(osp.join(CONFIG_DIR, scene, 'pick_meat.yaml'), 'r', encoding='utf-8') as f:
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
        # print(self.meat.pose.p[..., 2])
        success = self.meat.pose.p[..., 2] > 0.15 + self.agent.robot.pose.p[0, 2]
        return {
            "success": success,
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return 0

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return 0

