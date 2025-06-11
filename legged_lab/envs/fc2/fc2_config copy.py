# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# Original code is licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.
#
# This file contains code derived from Isaac Lab Project (BSD-3-Clause license)
# with modifications by Legged Lab Project (BSD-3-Clause license).

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass
from legged_lab.envs.fc2 import rewards as fc2_r
import legged_lab.mdp as mdp
from legged_lab.assets.fc2.fc2 import FC2_CFG
from legged_lab.envs.base.base_env_config import (  # noqa:F401
    BaseAgentCfg,
    BaseEnvCfg,
    BaseSceneCfg,
    DomainRandCfg,
    HeightScannerCfg,
    PhysxCfg,
    RewardCfg,
    RobotCfg,
    SimCfg,
)

from legged_lab.terrains import GRAVEL_TERRAINS_CFG, ROUGH_TERRAINS_CFG


@configclass
class fc2RewardCfg(RewardCfg):
    
    track_lin_vel_xy_exp = RewTerm(func=fc2_r.track_lin_vel_xy_yaw_frame_exp, weight=2.0, params={"std": 0.5})
    track_ang_vel_z_exp = RewTerm(func=fc2_r.track_ang_vel_z_world_exp, weight=2.0, params={"std": 0.5})
    lin_vel_z_l2 = RewTerm(func=fc2_r.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=fc2_r.ang_vel_xy_l2, weight=-0.1)
    energy = RewTerm(func=fc2_r.energy, weight=-1e-3)
    dof_acc_l2 = RewTerm(func=fc2_r.joint_acc_l2, weight=-1e-7)
    action_rate_l2 = RewTerm(func=fc2_r.action_rate_l2, weight=-0.001)

    undesired_contacts = RewTerm(
        func=fc2_r.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names="(?!.*foot.*).*"), "threshold": 1.0},
    )
    fly = RewTerm(
        func=fc2_r.fly,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*foot.*"), "threshold": 1.0},
    )
    body_orientation_l2 = RewTerm(
        func=fc2_r.body_orientation_l2, params={"asset_cfg": SceneEntityCfg("robot", body_names=".*torso.*")}, weight=-1.0
    )
    # flat_orientation_l2 = RewTerm(func=fc2_r.flat_orientation_l2, weight=-1.0)
    termination_penalty = RewTerm(func=fc2_r.is_terminated, weight=-200.0)
    feet_air_time = RewTerm(
        func=fc2_r.feet_air_time_positive_biped,
        weight=1,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*foot.*"), "threshold": 2},
    )
    feet_slide = RewTerm(
        func=fc2_r.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*foot.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*foot.*"),
        },
    )
    feet_force = RewTerm(
        func=fc2_r.body_force,
        weight=-1e-2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*foot.*"),
            "threshold": 850,
            "max_reward": 400,
        },
    )

    feet_distance = RewTerm(
    func=fc2_r.feet_distance_reward,
    weight=1.0,  # 奖励越大越好
    params={
        "asset_cfg": SceneEntityCfg("robot", body_names=".*foot.*"),
        "min_dist": 0.18,   # 你实际希望的最小间距
        "max_dist": 0.5 ,   # 你实际希望的最大间距
        },
    )

    # feet_too_near = RewTerm(
    #     func=fc2_r.feet_too_near_humanoid,
    #     weight=-2.0,
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*foot.*"]), "threshold": 0.18},
    # )
    feet_stumble = RewTerm(
        func=fc2_r.feet_stumble,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*foot.*"])},
    )
    dof_pos_limits = RewTerm(func=fc2_r.joint_pos_limits, weight=-1.0)
    joint_deviation_hip = RewTerm(
        func=fc2_r.joint_deviation_l1,
        weight=-1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*hipYaw.*", ".*hipRoll.*"]
            )
        },
    )

    joint_deviation_legs = RewTerm(
        func=fc2_r.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*hipPitch.*", ".*knee.*", ".*ankle.*"])},
    )

    feet_clearance = RewTerm(
        func=fc2_r.feet_clearance_reward,
        weight=2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*foot.*"]),
            "target_height": 0.05,
            "height_tol": 0.02,
        },
    )

    feet_contact_phase = RewTerm(
        func=fc2_r.feet_contact_phase_reward,
        weight=1,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*foot.*")},
    )

    track_ref_joint_pos = RewTerm(
        func=fc2_r.track_ref_joint_pos_reward,  # 假设你已经将 track_ref_joint_pos_reward 注册在 fc2_r 模块中
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),  # 或者添加 joint_names=[".*"] 限定关节
            "scale": 2.0,
            "penalty_coef": 0.2,
            "penalty_clip": 0.5,
        },
    )
    low_speed = RewTerm(
        func=fc2_r.low_speed_reward,
        weight=1.0,
        params={
            "min_scale": 0.5,
            "max_scale": 1.2
        }
    )

    action_smoothness = RewTerm(
    func=fc2_r.action_smoothness_reward,
    weight=-2e-3,
    params={"coef": 0.05}
    )

@configclass
class fc2FlatEnvCfg(BaseEnvCfg):
    cycle_time: float = 0.7
    reward = fc2RewardCfg()
    target_joint_pos_scale: float = 0.25
    def __post_init__(self):
        super().__post_init__()
        # self.scene.height_scanner.enable_height_scan = True
        self.robot.actor_obs_history_length = 1
        self.robot.critic_obs_history_length = 1
        self.scene.height_scanner.prim_body_name = "torso"
        self.scene.robot = FC2_CFG
        self.scene.terrain_type = "generator"
        self.scene.terrain_generator = GRAVEL_TERRAINS_CFG
        self.robot.terminate_contacts_body_names = [".*torso.*",".*arm.*"]
        self.robot.feet_body_names = [".*foot.*"]
        self.domain_rand.events.add_base_mass.params["asset_cfg"].body_names = [".*torso.*"]
        # self.sim.dt = 0.005
        # self.sim.decimation = 4
        self.domain_rand.events.push_robot.params["velocity_range"]= {
            "x": (-0.3, 0.3),
            "y": (-0.3, 0.3),}
        self.normalization.clip_actions = 18
        self.normalization.clip_observations = 18


@configclass
class fc2FlatAgentCfg(BaseAgentCfg):

    experiment_name: str = "fc2_flat"
    wandb_project: str = "fc2_flat"
    max_iterations: str = 5000
    def __post_init__(self):
        super().__post_init__()
        self.policy.class_name = "ActorCriticRecurrent"
        self.policy.actor_hidden_dims = [256, 256, 128]
        self.policy.critic_hidden_dims = [256, 256, 128]
        self.policy.rnn_hidden_size = 256
        self.policy.rnn_num_layers = 1
        self.policy.rnn_type = "lstm"
        self.algorithm.entropy_coef = 0.001
    

@configclass
class fc2RoughEnvCfg(fc2FlatEnvCfg):

    def __post_init__(self):
        super().__post_init__()
        self.scene.height_scanner.enable_height_scan = True
        self.scene.terrain_generator = ROUGH_TERRAINS_CFG
        self.scene.height_scanner.prim_body_name = "torso"
        self.robot.terminate_contacts_body_names = [".*torso.*",".*arm.*"]
        self.robot.feet_body_names = [".*foot.*"]
        self.domain_rand.events.add_base_mass.params["asset_cfg"].body_names = [".*torso.*"]
        self.robot.actor_obs_history_length = 1
        self.robot.critic_obs_history_length = 1
        self.reward.feet_air_time.weight = 0.25
        self.reward.track_lin_vel_xy_exp.weight = 1.5
        self.reward.track_ang_vel_z_exp.weight = 1.5
        self.reward.lin_vel_z_l2.weight = -0.25
        self.sim.dt = 0.001
        self.sim.decimation = 10


@configclass
class fc2RoughAgentCfg(BaseAgentCfg):
    device: str = "cuda:3"
    experiment_name: str = "fc2_rough"
    wandb_project: str = "fc2_rough"
    max_iterations: str = 5000
    def __post_init__(self):
        super().__post_init__()
        self.policy.class_name = "ActorCriticRecurrent"
        self.policy.actor_hidden_dims = [256, 256, 128]
        self.policy.critic_hidden_dims = [256, 256, 128]
        self.policy.rnn_hidden_size = 256
        self.policy.rnn_num_layers = 1
        self.policy.rnn_type = "lstm"



