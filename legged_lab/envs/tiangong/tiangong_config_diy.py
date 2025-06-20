# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# Original code is licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.
#
# Copyright (c) 2025, Siyuan Wang.
# All rights reserved.
# Further modifications are licensed under BSD-3-Clause.
#
# This file contains code derived from Isaac Lab Project (BSD-3-Clause license)
# with modifications by Legged Lab Project (BSD-3-Clause license)
# and Siyuan Wang (BSD-3-Clause license).



from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass
from legged_lab.envs.tiangong import rewards as tiangong_r
import legged_lab.mdp as mdp
from legged_lab.assets.tiangong.tiangong import tiangong_CFG
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
class tiangongRewardCfg(RewardCfg):
    base_height = RewTerm(
        func=tiangong_r.base_height_reward,
        weight=0.1,  # 你可以根据任务调整权重
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*foot.*"),  # torso 是机器人躯干的 body 名称
            "target_height": 0.89,     # 你的机器人 torso 在站立时的理想高度    # 通常为 feet 几何中心到 surface 的偏移
            "scale": 100.0             # 奖励斜率，越大越惩罚偏离
        }
    )    
    track_lin_vel_xy_exp = RewTerm(func=tiangong_r.track_lin_vel_xy_yaw_frame_exp, weight=2.0, params={"std": 0.5})
    track_ang_vel_z_exp = RewTerm(func=tiangong_r.track_ang_vel_z_world_exp, weight=2.0, params={"std": 0.5})
    lin_vel_z_l2 = RewTerm(func=tiangong_r.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=tiangong_r.ang_vel_xy_l2, weight=-0.5)
    energy = RewTerm(func=tiangong_r.energy, weight=-1e-3)
    dof_acc_l2 = RewTerm(func=tiangong_r.joint_acc_l2, weight=-1e-6)
    action_rate_l2 = RewTerm(func=tiangong_r.action_rate_l2, weight=-0.001)

    undesired_contacts = RewTerm(
        func=tiangong_r.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names="(?!.*ankle_roll.*).*"), "threshold": 1.0},
    )
    fly = RewTerm(
        func=tiangong_r.fly,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"), "threshold": 1.0},
    )
    body_orientation_l2 = RewTerm(
        func=tiangong_r.body_orientation_l2, params={"asset_cfg": SceneEntityCfg("robot", body_names=".*pelvis.*")}, weight=1.5
    )
    # flat_orientation_l2 = RewTerm(func=tiangong_r.flat_orientation_l2, weight=-1.0)
    termination_penalty = RewTerm(func=tiangong_r.is_terminated, weight=-200.0)
    feet_air_time = RewTerm(
        func=tiangong_r.feet_air_time_positive_biped,
        weight=1,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"), "threshold": 2},
    )
    feet_slide = RewTerm(
        func=tiangong_r.feet_slide,
        weight=-0.8,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
        },
    )
    feet_force = RewTerm(
        func=tiangong_r.body_force,
        weight=-1e-2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
            "threshold": 700,
            "max_reward": 400,
        },
    )

    feet_parallel_ground = RewTerm(
        func=tiangong_r.feet_parallel_to_ground_reward,
        weight=0.6,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*")}
    )


    feet_distance = RewTerm(
    func=tiangong_r.feet_distance_reward,
    weight=0.3,  
    params={
        "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
        "min_dist": 0.23,   # 你实际希望的最小间距
        "max_dist": 0.6 ,   # 你实际希望的最大间距
        },
    )

    feet_stumble = RewTerm(
        func=tiangong_r.feet_stumble,
        weight=-4.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*ankle_roll.*"])},
    )
    dof_pos_limits = RewTerm(func=tiangong_r.joint_pos_limits, weight=-1.0)
    joint_deviation_hip = RewTerm(
        func=tiangong_r.joint_deviation_l1,
        weight=-1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*hip_yaw.*", ".*hip_roll.*"]
            )
        },
    )

    joint_deviation_legs = RewTerm(
        func=tiangong_r.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*hip_pitch.*", ".*knee.*", ".*ankle.*"])},
    )

    feet_clearance = RewTerm(
        func=tiangong_r.feet_clearance_reward,
        weight=1.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*foot.*"]),
            "target_height": 0.05,
            "height_tol": 0.02,
        },
    )

    feet_contact_phase = RewTerm(
        func=tiangong_r.feet_contact_phase_reward,
        weight=1,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*")},
    )

    track_ref_joint_pos = RewTerm(
        func=tiangong_r.track_ref_joint_pos_reward,  # 假设你已经将 track_ref_joint_pos_reward 注册在 tiangong_r 模块中
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),  # 或者添加 joint_names=[".*"] 限定关节
            "scale": 2.0,
            "penalty_coef": 0.2,
            "penalty_clip": 0.5,
        },
    )
    low_speed = RewTerm(
        func=tiangong_r.low_speed_reward,
        weight=0.6,
        params={
            "min_scale": 0.5,
            "max_scale": 1.2
        }
    )

    action_smoothness = RewTerm(
    func=tiangong_r.action_smoothness_reward,
    weight=-3e-3,
    params={"coef": 0.05}
    )

@configclass
class tiangongFlatEnvCfg(BaseEnvCfg):
    cycle_time: float = 1
    reward = tiangongRewardCfg()
    target_joint_pos_scale: float = 0.16
    def __post_init__(self):
        super().__post_init__()
        # self.scene.height_scanner.enable_height_scan = True
        self.robot.actor_obs_history_length = 15
        self.robot.critic_obs_history_length = 3
        self.scene.height_scanner.prim_body_name = "pelvis"
        self.scene.robot = tiangong_CFG
        self.scene.terrain_type = "generator"
        self.scene.terrain_generator = GRAVEL_TERRAINS_CFG
        self.robot.terminate_contacts_body_names = [".*pelvis.*","hip.*"]
        self.robot.feet_body_names = [".*ankle_roll.*"]
        self.domain_rand.events.add_base_mass.params["asset_cfg"].body_names = [".*pelvis.*"]
        self.domain_rand.events.push_robot.params["velocity_range"]= {
            "x": (-0.3, 0.3),
            "y": (-0.3, 0.3),}
        self.normalization.clip_actions = 18
        self.normalization.clip_observations = 30


@configclass
class tiangongFlatAgentCfg(BaseAgentCfg):

    experiment_name: str = "tiangong_flat"
    wandb_project: str = "tiangong_flat"
    max_iterations: str = 5000
    def __post_init__(self):
        super().__post_init__()
        self.policy.class_name = "ActorCritic"
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]

        self.algorithm.entropy_coef = 0.001
    

@configclass
class tiangongRoughEnvCfg(tiangongFlatEnvCfg):

    def __post_init__(self):
        super().__post_init__()
        self.scene.height_scanner.enable_height_scan = True
        self.scene.terrain_generator = ROUGH_TERRAINS_CFG
        self.scene.height_scanner.prim_body_name = "pelvis"
        self.robot.terminate_contacts_body_names = [".*pelvis.*","hip.*"]
        self.robot.feet_body_names = [".*ankle_roll.*"]
        self.domain_rand.events.add_base_mass.params["asset_cfg"].body_names = [".*pelvis.*"]
        self.robot.actor_obs_history_length = 1
        self.robot.critic_obs_history_length = 1



@configclass
class tiangongRoughAgentCfg(BaseAgentCfg):
    device: str = "cuda:0"
    experiment_name: str = "tiangong_rough"
    wandb_project: str = "tiangong_rough"
    max_iterations: str = 100000
    def __post_init__(self):
        super().__post_init__()
        self.policy.class_name = "ActorCriticRecurrent"
        self.policy.actor_hidden_dims = [256, 256, 128]
        self.policy.critic_hidden_dims = [256, 256, 128]
        self.policy.rnn_hidden_size = 256
        self.policy.rnn_num_layers = 1
        self.policy.rnn_type = "lstm"



