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
#
# Copyright (c) 2025, Siyuan Wang.
# All rights reserved.
# Further modifications are licensed under BSD-3-Clause.
#
# This file contains code derived from Isaac Lab Project (BSD-3-Clause license)
# with modifications by Legged Lab Project (BSD-3-Clause license)
# and Siyuan Wang (BSD-3-Clause license).


from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from envs.tiangong.tiangong_env import tiangongEnv as BaseEnv


def track_lin_vel_xy_yaw_frame_exp(
    env: BaseEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    vel_yaw = math_utils.quat_apply_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    lin_vel_error = torch.sum(torch.square(env.command_generator.command[:, :2] - vel_yaw[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env: BaseEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_generator.command[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def lin_vel_z_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def ang_vel_xy_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)


def energy(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.norm(torch.abs(asset.data.applied_torque * asset.data.joint_vel), dim=-1)
    return reward


def joint_acc_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def action_rate_l2(env: BaseEnv) -> torch.Tensor:
    return torch.sum(
        torch.square(
            env.action_buffer._circular_buffer.buffer[:, -1, :] - env.action_buffer._circular_buffer.buffer[:, -2, :]
        ),
        dim=1,
    )



def undesired_contacts(
    env: BaseEnv, 
    threshold: float, 
    sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # 当前帧的接触力
    net_contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    # 是否大于阈值
    is_contact = torch.norm(net_contact_forces, dim=-1) > threshold  # [N, num_bodies]
    # 每个 env，发生 contact 的 undesired body 数
    return torch.sum(is_contact.float(), dim=1)


def track_ref_joint_pos_reward(env: BaseEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    scale: float = 2.0,
    penalty_coef: float = 0.2,
    penalty_clip: float = 0.5,
) -> torch.Tensor:
    """
    奖励当前关节角度接近目标 ref_dof_pos。
    - 目标位置为 env.ref_dof_pos + default_dof_pos
    - 奖励为 Gaussian 形式 - L2 距离惩罚组合

    Args:
        env: 当前 BaseEnv 环境
        asset_cfg: 控制关节目标参考的主体配置
        scale: exp(-scale * error) 的指数幅度系数
        penalty_coef: L2误差的线性惩罚系数
        penalty_clip: 惩罚上限阈值

    Returns:
        torch.Tensor [num_envs]
    """
    asset :Articulation = env.scene[asset_cfg.name]

    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    target_joint_pos = env.ref_dof_pos[:, asset_cfg.joint_ids] + asset.data.default_joint_pos[:, asset_cfg.joint_ids]

    diff = current_joint_pos - target_joint_pos
    l2_norm = torch.norm(diff, dim=1)

    reward = torch.exp(-scale * l2_norm) - penalty_coef * l2_norm.clamp(max=penalty_clip)
    return reward


def fly(env: BaseEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=-1) < 0.5


def flat_orientation_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def is_terminated(env: BaseEnv) -> torch.Tensor:
    """Penalize terminated episodes that don't correspond to episodic timeouts."""
    return env.reset_buf * ~env.time_out_buf


def feet_air_time_positive_biped(env: BaseEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)

    return reward

from isaaclab.utils.math import quat_apply

def feet_parallel_to_ground_reward(
    env: "BaseEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    foot_quat = asset.data.body_quat_w[:, asset_cfg.body_ids, :]  # [N, num_feet, 4]
    batch, num_feet = foot_quat.shape[0], foot_quat.shape[1]
    # local z (0,0,1)
    local_z = torch.zeros((batch, num_feet, 3), dtype=foot_quat.dtype, device=foot_quat.device)
    local_z[..., 2] = 1.0
    # flatten for quat_apply
    foot_quat_flat = foot_quat.reshape(-1, 4)
    local_z_flat = local_z.reshape(-1, 3)
    foot_z_world = quat_apply(foot_quat_flat, local_z_flat).reshape(batch, num_feet, 3)
    # 世界z轴
    world_z = torch.tensor([0, 0, 1], dtype=foot_z_world.dtype, device=foot_z_world.device)
    alignment = (foot_z_world * world_z).sum(dim=-1)  # [N, num_feet]
    reward = torch.exp((alignment - 1) * 10)
    return reward.mean(dim=1)


def feet_slide(
    env: BaseEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: Articulation = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def body_force(
    env: BaseEnv, sensor_cfg: SceneEntityCfg, threshold: float = 500, max_reward: float = 400
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    reward = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2].norm(dim=-1)
    reward[reward < threshold] = 0
    reward[reward > threshold] -= threshold
    reward = reward.clamp(min=0, max=max_reward)
    return reward


def joint_deviation_l1(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(angle), dim=1)

def body_orientation_l2(
    env: BaseEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), 
    sigma: float = 0.2,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_orientation = math_utils.quat_apply_inverse(
        asset.data.body_quat_w[:, asset_cfg.body_ids[0], :], asset.data.GRAVITY_VEC_W
    )
    # 指数型奖励，偏差越小奖励越大
    l2 = torch.sum(torch.square(body_orientation[:, :2]), dim=1)
    reward = torch.exp(-l2 / sigma**2)
    return reward



def feet_stumble(env: BaseEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return torch.any(
        torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
        > 5 * torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]),
        dim=1,
    )


# def feet_too_near_humanoid(
#     env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold: float = 0.2
# ) -> torch.Tensor:
#     assert len(asset_cfg.body_ids) == 2
#     asset: Articulation = env.scene[asset_cfg.name]
#     feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
#     distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
#     return (threshold - distance).clamp(min=0)
def feet_distance_reward(
    env: BaseEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_dist: float = 0.15,
    max_dist: float = 0.28,
) -> torch.Tensor:
    """
    奖励双脚间距既不太近也不太远。
    - min_dist: 最小距离，距离小于此值会被惩罚。
    - max_dist: 最大距离，距离大于此值会被惩罚。
    - 奖励区间 [min_dist, max_dist]。
    """
    assert len(asset_cfg.body_ids) == 2, "需要正好2个脚 body id"
    asset: Articulation = env.scene[asset_cfg.name]
    foot_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :2]  # [N, 2, 2]
    foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)  # [N]
    d_min = torch.clamp(foot_dist - min_dist, -0.5, 0.0)
    d_max = torch.clamp(foot_dist - max_dist, 0, 0.5)
    reward = (
        torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)
    ) / 2
    return reward


def joint_pos_limits(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions if they cross the soft limits.

    This is computed as a sum of the absolute value of the difference between the joint position and the soft limits.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    out_of_limits = -(
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
    ).clip(max=0.0)
    out_of_limits += (
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
    ).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


def feet_clearance_reward(
    env: BaseEnv,
    sensor_cfg: SceneEntityCfg,
    target_height: float = 0.05,
    height_tol: float = 0.02,
) -> torch.Tensor:
    """
    奖励机器人在摆动相位时足部离地高度接近目标值。
    - 当脚在摆动相时（非支撑），如果当前脚高度接近目标高度，就给奖励。

    Args:
        env: 环境实例。
        sensor_cfg: 包含 feet 的 SceneEntityCfg。
        target_height: 目标足部高度。
        height_tol: 高度误差容差（奖励范围内）。

    Returns:
        torch.Tensor: 每个环境的奖励 [num_envs]
    """
    contact_forces = env.contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    contact = contact_forces > 5.0  # [N, num_feet]
    
    feet_z = env.robot.data.body_pos_w[:, sensor_cfg.body_ids, 2] - 0.078  # [N, num_feet]

    delta_z = feet_z - env.last_feet_z
    env.feet_height += delta_z
    env.last_feet_z = feet_z

    # swing phase mask
    swing_mask = 1 - env._get_gait_phase()  # [N, 2]

    match_target = torch.abs(env.feet_height - target_height) < height_tol
    reward = torch.sum(match_target * swing_mask, dim=1).float()

    # 清空支撑脚的高度累积
    env.feet_height *= ~contact
    return reward


def feet_contact_phase_reward(
    env: BaseEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """
    奖励脚接触状态是否与 gait phase（步态节奏）匹配：
    - 如果 contact == stance_mask 则奖励 +1
    - 否则惩罚 -0.8
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] > 5.0  # [N, num_feet]
    stance_mask = env._get_gait_phase()  # [N, 2]

    reward = torch.where(contact == stance_mask, 1.0, -0.8)
    return torch.mean(reward, dim=1)


def reward_feet_air_time(env: BaseEnv, max_air_time=0.5, contact_threshold=5.0, asset_cfg=None):
    feet_indices = asset_cfg.body_ids if asset_cfg else env.feet_cfg.body_ids
    contact = env.contact_sensor.data.net_forces_w_history[:, 2, feet_indices] > contact_threshold
    stance_mask = env._get_gait_phase()
    # 以下变量请确保在 env/init_buffers 中初始化为 zeros_like
    env.contact_filt = torch.logical_or(
        torch.logical_or(contact, stance_mask), getattr(env, 'last_contacts', torch.zeros_like(contact))
    )
    env.last_contacts = contact
    env.feet_air_time += env.step_dt
    first_contact = (env.feet_air_time > 0.0) * env.contact_filt
    air_time = env.feet_air_time.clamp(0, max_air_time) * first_contact
    env.feet_air_time *= ~env.contact_filt
    return air_time.sum(dim=1)


def low_speed_reward(
    env: BaseEnv, 
    min_scale: float = 0.5, 
    max_scale: float = 1.2
) -> torch.Tensor:
    """
    奖励或惩罚机器人线速度与命令速度的匹配情况。
    - 匹配范围 [min_scale, max_scale] * command，奖励高
    - 速度过低/过高/方向错，奖励低或惩罚
    """
    # 获取 base 线速度和命令（假设是 x 方向，若需全向改此处）
    base_lin_vel = env.robot.data.root_lin_vel_b[:, 0] if hasattr(env.robot.data, 'root_lin_vel_b') else env.base_lin_vel[:, 0]
    commands = env.command_generator.command[:, 0] if hasattr(env, 'command_generator') else env.commands[:, 0]

    absolute_speed = torch.abs(base_lin_vel)
    absolute_command = torch.abs(commands)

    # 匹配区间判定
    speed_too_low = absolute_speed < min_scale * absolute_command
    speed_too_high = absolute_speed > max_scale * absolute_command
    speed_desired = ~(speed_too_low | speed_too_high)

    # 方向匹配
    sign_mismatch = torch.sign(base_lin_vel) != torch.sign(commands)

    reward = torch.zeros_like(base_lin_vel)
    reward[speed_too_low] = -1.0
    reward[speed_too_high] = -0.5
    reward[speed_desired] = 1.2
    reward[sign_mismatch] = -2.0

    # 命令很小时不奖励（与原写法一致）
    reward = reward * (absolute_command > 0.1)
    return reward


def action_smoothness_reward(env: BaseEnv, coef: float = 0.05) -> torch.Tensor:
    """
    动作平滑奖励。鼓励动作连续，减少抖动/剧烈变化。
    - 需要动作历史（当前/上一步/上上步）。
    - coef 控制当前动作的 L1 项惩罚强度。
    """
    # 假设 action_buffer 是 DelayBuffer/CircularBuffer
    # buffer: [num_envs, history_len, num_actions]
    actions      = env.action_buffer._circular_buffer.buffer[:, -1, :]
    last_actions = env.action_buffer._circular_buffer.buffer[:, -2, :]
    last_last_actions = env.action_buffer._circular_buffer.buffer[:, -3, :]

    term_1 = torch.sum(torch.square(last_actions - actions), dim=1)
    term_2 = torch.sum(torch.square(actions + last_last_actions - 2 * last_actions), dim=1)
    term_3 = coef * torch.sum(torch.abs(actions), dim=1)
    return term_1 + term_2 + term_3

