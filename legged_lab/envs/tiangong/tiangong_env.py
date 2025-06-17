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


import isaaclab.sim as sim_utils
import isaacsim.core.utils.torch as torch_utils  # type: ignore
import numpy as np
import torch
from isaaclab.assets.articulation import Articulation
from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.managers import EventManager, RewardManager
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.sim import PhysxCfg, SimulationContext
from isaaclab.utils.buffers import CircularBuffer, DelayBuffer
from rsl_rl.env import VecEnv

from legged_lab.envs.base.base_env_config import BaseEnvCfg
from legged_lab.utils.env_utils.scene import SceneCfg


class tiangongEnv(VecEnv):
    def __init__(self, cfg: BaseEnvCfg, headless):
        self.cfg: BaseEnvCfg

        self.cfg = cfg
        self.headless = headless
        self.device = self.cfg.device
        self.physics_dt = self.cfg.sim.dt
        self.step_dt = self.cfg.sim.decimation * self.cfg.sim.dt
        self.num_envs = self.cfg.scene.num_envs
        self.seed(cfg.scene.seed)

        sim_cfg = sim_utils.SimulationCfg(
            device=cfg.device,
            dt=cfg.sim.dt,
            render_interval=cfg.sim.decimation,
            physx=PhysxCfg(gpu_max_rigid_patch_count=cfg.sim.physx.gpu_max_rigid_patch_count),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
        )
        self.sim = SimulationContext(sim_cfg)

        scene_cfg = SceneCfg(config=cfg.scene, physics_dt=self.physics_dt, step_dt=self.step_dt)
        self.scene = InteractiveScene(scene_cfg)
        self.sim.reset()

        self.robot: Articulation = self.scene["robot"]
        self.contact_sensor: ContactSensor = self.scene.sensors["contact_sensor"]
        if self.cfg.scene.height_scanner.enable_height_scan:
            self.height_scanner: RayCaster = self.scene.sensors["height_scanner"]

        command_cfg = UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=self.cfg.commands.resampling_time_range,
            rel_standing_envs=self.cfg.commands.rel_standing_envs,
            rel_heading_envs=self.cfg.commands.rel_heading_envs,
            heading_command=self.cfg.commands.heading_command,
            heading_control_stiffness=self.cfg.commands.heading_control_stiffness,
            debug_vis=self.cfg.commands.debug_vis,
            ranges=self.cfg.commands.ranges,
        )
        self.command_generator = UniformVelocityCommand(cfg=command_cfg, env=self)
        self.reward_manager = RewardManager(self.cfg.reward, self)

        self.init_buffers()
        print(f"[INIT] cycle_time = {self.cfg.cycle_time}, target_joint_pos_scale = {self.cfg.target_joint_pos_scale}")
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.event_manager = EventManager(self.cfg.domain_rand.events, self)
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")
        self.reset(env_ids)

    def init_buffers(self):
        self.extras = {}
        self.joint_name_to_idx = {
            name: i for i, name in enumerate(self.robot.data.joint_names)
        }

        self.max_episode_length_s = self.cfg.scene.max_episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.step_dt)
        self.num_actions = self.robot.data.default_joint_pos.shape[1]
        self.clip_actions = self.cfg.normalization.clip_actions
        self.clip_obs = self.cfg.normalization.clip_observations

        self.action_scale = self.cfg.robot.action_scale
        self.action_buffer = DelayBuffer(
            self.cfg.domain_rand.action_delay.params["max_delay"], self.num_envs, device=self.device
        )
        self.action_buffer.compute(
            torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        )
        if self.cfg.domain_rand.action_delay.enable:
            time_lags = torch.randint(
                low=self.cfg.domain_rand.action_delay.params["min_delay"],
                high=self.cfg.domain_rand.action_delay.params["max_delay"] + 1,
                size=(self.num_envs,),
                dtype=torch.int,
                device=self.device,
            )
            self.action_buffer.set_time_lag(time_lags, torch.arange(self.num_envs, device=self.device))

        self.robot_cfg = SceneEntityCfg(name="robot")
        self.robot_cfg.resolve(self.scene)
        self.termination_contact_cfg = SceneEntityCfg(
            name="contact_sensor", body_names=self.cfg.robot.terminate_contacts_body_names
        )
        self.termination_contact_cfg.resolve(self.scene)
        self.feet_cfg = SceneEntityCfg(name="contact_sensor", body_names=self.cfg.robot.feet_body_names)
        self.feet_cfg.resolve(self.scene)
        self.feet_height = torch.zeros(self.num_envs, len(self.feet_cfg.body_ids), device=self.device)
        self.last_feet_z = torch.zeros_like(self.feet_height)
        self.obs_scales = self.cfg.normalization.obs_scales
        self.add_noise = self.cfg.noise.add_noise

        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.sim_step_counter = 0
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.init_obs_buffer()


    def _get_phase(self) -> torch.Tensor:
        """
        Returns phase in [0, 1), computed from episode time and cycle period.
        """
        cycle_time = self.cfg.cycle_time  
        return (self.episode_length_buf * self.step_dt / cycle_time) % 1.0  # [num_envs]

    def _get_gait_phase(self) -> torch.Tensor:
        """
        Returns stance mask: 1 is stance, 0 is swing for [left, right] foot per env.
        Shape: [num_envs, 2]
        """
        phase = self._get_phase()  # [num_envs]
        sin_pos = torch.sin(2 * torch.pi * phase)  # [num_envs]

        stance_mask = torch.zeros((self.num_envs, 2), device=self.device)
        stance_mask[:, 0] = sin_pos >= 0  # left foot stance
        stance_mask[:, 1] = sin_pos < 0   # right foot stance

        # Optional: double support phase (e.g., near phase boundary)
        stance_mask[torch.abs(sin_pos) < 0.1] = 1
        return stance_mask

    def compute_ref_state(self):
        """
        构造 ref_dof_pos，给定 phase 控制的目标关节位置（用于节奏模仿或 reward）。
        """
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        sin_pos_l = sin_pos.clone()
        sin_pos_r = sin_pos.clone()

        self.ref_dof_pos = torch.zeros_like(self.robot.data.joint_pos)
        scale_1 = self.cfg.target_joint_pos_scale
        scale_2 = 1.5 * scale_1
        scale_3 = scale_2 - scale_1

        jn = self.joint_name_to_idx

        # 左腿 swing phase
        sin_pos_l[sin_pos_l > 0] = 0
        self.ref_dof_pos[:, jn["hip_pitch_l_joint"]] = sin_pos_l * scale_1
        self.ref_dof_pos[:, jn["knee_pitch_l_joint"]] = -sin_pos_l * scale_2
        self.ref_dof_pos[:, jn["ankle_pitch_l_joint"]] = sin_pos_l * scale_3

        # 右腿 swing phase
        sin_pos_r[sin_pos_r < 0] = 0
        self.ref_dof_pos[:, jn["hip_pitch_r_joint"]] = -sin_pos_r * scale_1
        self.ref_dof_pos[:, jn["knee_pitch_r_joint"]] = sin_pos_r * scale_2
        self.ref_dof_pos[:, jn["ankle_pitch_r_joint"]] = -sin_pos_r * scale_3

        # double support 时归零
        self.ref_dof_pos[torch.abs(sin_pos) < 0.1] = 0.0

        # 可选：构造 ref_action 作为模仿目标
        self.ref_action = 2.0 * self.ref_dof_pos

    def get_heightfield_z(env, x, y):
        """
        查找任意 (x, y) 处的地形高度，适用于 IsaacLab/LeggedGym 所有 heightfield terrain。
        支持 torch tensor 批量操作。
        """
        terrain = env.scene.terrain
        hf = terrain.height_samples             # (H, W) torch.Tensor or np.array
        H, W = hf.shape
        x_min, y_min = terrain.origin_x, terrain.origin_y
        dx = terrain.horizontal_scale           # grid resolution
        z_min = getattr(terrain, "vertical_origin", 0.0)
        vertical_scale = getattr(terrain, "vertical_scale", 1.0)

        # 确保 x, y 都是 torch 张量
        x = torch.as_tensor(x, device=hf.device if hasattr(hf, "device") else "cpu")
        y = torch.as_tensor(y, device=hf.device if hasattr(hf, "device") else "cpu")
        x_idx = ((x - x_min) / dx).long().clamp(0, W-1)
        y_idx = ((y - y_min) / dx).long().clamp(0, H-1)
        z = hf[y_idx, x_idx] * vertical_scale + z_min
        return z

    def compute_current_observations(self):
        robot = self.robot
        net_contact_forces = self.contact_sensor.data.net_forces_w_history
        phase = self._get_phase()
        phase_obs = torch.stack([torch.sin(2 * torch.pi * phase), torch.cos(2 * torch.pi * phase)], dim=1)

        ang_vel = robot.data.root_ang_vel_b
        projected_gravity = robot.data.projected_gravity_b
        command = self.command_generator.command
        joint_pos = robot.data.joint_pos - robot.data.default_joint_pos
        joint_vel = robot.data.joint_vel - robot.data.default_joint_vel
        action = self.action_buffer._circular_buffer.buffer[:, -1, :]
        current_actor_obs = torch.cat(
            [
                ang_vel * self.obs_scales.ang_vel,
                projected_gravity * self.obs_scales.projected_gravity,
                command * self.obs_scales.commands,
                phase_obs * self.obs_scales.commands,
                joint_pos * self.obs_scales.joint_pos,
                joint_vel * self.obs_scales.joint_vel,
                action * self.obs_scales.actions,
                
            ],
            dim=-1,
        )

        root_lin_vel = robot.data.root_lin_vel_b
        feet_contact = torch.max(torch.norm(net_contact_forces[:, :, self.feet_cfg.body_ids], dim=-1), dim=1)[0] > 0.5
        current_critic_obs = torch.cat(
            [current_actor_obs, root_lin_vel * self.obs_scales.lin_vel, feet_contact], dim=-1
        )

        return current_actor_obs, current_critic_obs

    def compute_observations(self):
        current_actor_obs, current_critic_obs = self.compute_current_observations()
        if self.add_noise:
            current_actor_obs += (2 * torch.rand_like(current_actor_obs) - 1) * self.noise_scale_vec

        self.actor_obs_buffer.append(current_actor_obs)
        self.critic_obs_buffer.append(current_critic_obs)

        actor_obs = self.actor_obs_buffer.buffer.reshape(self.num_envs, -1)
        critic_obs = self.critic_obs_buffer.buffer.reshape(self.num_envs, -1)
        if self.cfg.scene.height_scanner.enable_height_scan:
            height_scan = (
                self.height_scanner.data.pos_w[:, 2].unsqueeze(1)
                - self.height_scanner.data.ray_hits_w[..., 2]
                - self.cfg.normalization.height_scan_offset
            ) * self.obs_scales.height_scan
            critic_obs = torch.cat([critic_obs, height_scan], dim=-1)
            if self.add_noise:
                height_scan += (2 * torch.rand_like(height_scan) - 1) * self.height_scan_noise_vec
            actor_obs = torch.cat([actor_obs, height_scan], dim=-1)

        actor_obs = torch.clip(actor_obs, -self.clip_obs, self.clip_obs)
        critic_obs = torch.clip(critic_obs, -self.clip_obs, self.clip_obs)

        return actor_obs, critic_obs

    def reset(self, env_ids):
        if len(env_ids) == 0:
            return

        self.extras["log"] = dict()
        if self.cfg.scene.terrain_generator is not None:
            if self.cfg.scene.terrain_generator.curriculum:
                terrain_levels = self.update_terrain_levels(env_ids)
                self.extras["log"].update(terrain_levels)

        self.scene.reset(env_ids)
        if "reset" in self.event_manager.available_modes:
            self.event_manager.apply(
                mode="reset",
                env_ids=env_ids,
                dt=self.step_dt,
                global_env_step_count=self.sim_step_counter // self.cfg.sim.decimation,
            )

        reward_extras = self.reward_manager.reset(env_ids)
        self.extras["log"].update(reward_extras)
        self.extras["time_outs"] = self.time_out_buf

        self.command_generator.reset(env_ids)
        self.actor_obs_buffer.reset(env_ids)
        self.critic_obs_buffer.reset(env_ids)
        self.action_buffer.reset(env_ids)
        self.episode_length_buf[env_ids] = 0

        self.scene.write_data_to_sim()
        self.sim.forward()

    def step(self, actions: torch.Tensor):
        self.compute_ref_state()
        delayed_actions = self.action_buffer.compute(actions)

        cliped_actions = torch.clip(delayed_actions, -self.clip_actions, self.clip_actions).to(self.device)
        processed_actions = cliped_actions * self.action_scale + self.robot.data.default_joint_pos

        for _ in range(self.cfg.sim.decimation):
            self.sim_step_counter += 1
            self.robot.set_joint_position_target(processed_actions)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)

        if not self.headless:
            self.sim.render()

        self.episode_length_buf += 1
        self.command_generator.compute(self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        self.reset_buf, self.time_out_buf = self.check_reset()
        reward_buf = self.reward_manager.compute(self.step_dt)
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset(env_ids)

        actor_obs, critic_obs = self.compute_observations()
        self.extras["observations"] = {"critic": critic_obs}

        return actor_obs, reward_buf, self.reset_buf, self.extras

    def check_reset(self):
        net_contact_forces = self.contact_sensor.data.net_forces_w_history

        reset_buf = torch.any(
            torch.max(
                torch.norm(
                    net_contact_forces[:, :, self.termination_contact_cfg.body_ids],
                    dim=-1,
                ),
                dim=1,
            )[0]
            > 1.0,
            dim=1,
        )
        time_out_buf = self.episode_length_buf >= self.max_episode_length
        reset_buf |= time_out_buf
        return reset_buf, time_out_buf

    def init_obs_buffer(self):
        if self.add_noise:
            actor_obs, _ = self.compute_current_observations()
            noise_vec = torch.zeros_like(actor_obs[0])
            noise_scales = self.cfg.noise.noise_scales
            noise_vec[:3] = noise_scales.ang_vel * self.obs_scales.ang_vel
            noise_vec[3:6] = noise_scales.projected_gravity * self.obs_scales.projected_gravity
            noise_vec[6:11] = 0
            noise_vec[11 : 11 + self.num_actions] = noise_scales.joint_pos * self.obs_scales.joint_pos
            noise_vec[11 + self.num_actions : 11 + self.num_actions * 2] = (
                noise_scales.joint_vel * self.obs_scales.joint_vel
            )
            noise_vec[11 + self.num_actions * 2 : 11 + self.num_actions * 3] = 0.0
            
            self.noise_scale_vec = noise_vec

            if self.cfg.scene.height_scanner.enable_height_scan:
                height_scan = (
                    self.height_scanner.data.pos_w[:, 2].unsqueeze(1)
                    - self.height_scanner.data.ray_hits_w[..., 2]
                    - self.cfg.normalization.height_scan_offset
                )
                height_scan_noise_vec = torch.zeros_like(height_scan[0])
                height_scan_noise_vec[:] = noise_scales.height_scan * self.obs_scales.height_scan
                self.height_scan_noise_vec = height_scan_noise_vec

        self.actor_obs_buffer = CircularBuffer(
            max_len=self.cfg.robot.actor_obs_history_length, batch_size=self.num_envs, device=self.device
        )
        self.critic_obs_buffer = CircularBuffer(
            max_len=self.cfg.robot.critic_obs_history_length, batch_size=self.num_envs, device=self.device
        )

    def update_terrain_levels(self, env_ids):
        distance = torch.norm(self.robot.data.root_pos_w[env_ids, :2] - self.scene.env_origins[env_ids, :2], dim=1)
        move_up = distance > self.scene.terrain.cfg.terrain_generator.size[0] / 2
        move_down = (
            distance < torch.norm(self.command_generator.command[env_ids, :2], dim=1) * self.max_episode_length_s * 0.5
        )
        move_down *= ~move_up
        self.scene.terrain.update_env_origins(env_ids, move_up, move_down)
        extras = {"Curriculum/terrain_levels": torch.mean(self.scene.terrain.terrain_levels.float())}
        return extras

    def get_observations(self):
        actor_obs, critic_obs = self.compute_observations()
        self.extras["observations"] = {"critic": critic_obs}
        return actor_obs, self.extras

    @staticmethod
    def seed(seed: int = -1) -> int:
        try:
            import omni.replicator.core as rep  # type: ignore

            rep.set_global_seed(seed)
        except ModuleNotFoundError:
            pass
        return torch_utils.set_seed(seed)
