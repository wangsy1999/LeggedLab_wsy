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


import argparse

from isaaclab.app import AppLauncher
from rsl_rl.runners import OnPolicyRunner

from legged_lab.utils import task_registry

# local imports
import legged_lab.utils.cli_args as cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--backup_env", type=bool, default=False, help="")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
import os
from datetime import datetime

import torch
from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.utils import get_checkpoint_path

from legged_lab.envs import *  # noqa:F401, F403
from legged_lab.utils.cli_args import update_rsl_rl_cfg

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
import shutil, os, inspect

def backup_current_env(log_dir, env_cfg, env_class, extra_files=None):
    """只备份本次env相关核心文件和可选自定义文件到log_dir/env_backup/"""
    backup_path = os.path.join(log_dir, "env_backup")
    os.makedirs(backup_path, exist_ok=True)
    # 备份 config、reward、env class
    config_file = inspect.getfile(type(env_cfg)) if not isinstance(env_cfg, type) else inspect.getfile(env_cfg)
    env_file = inspect.getfile(env_class)
    files_to_backup = [config_file, env_file]

    # 如果 rewards、其它py是单独文件，也可补充（例如 rewards.py 路径可硬编码或自动推断）
    reward_file = os.path.join(os.path.dirname(config_file), "rewards.py")
    if os.path.exists(reward_file):
        files_to_backup.append(reward_file)

    if extra_files:
        files_to_backup += extra_files

    for f in set(files_to_backup):
        if os.path.exists(f):
            shutil.copy2(f, backup_path)
    print(f"[INFO] Current env backup finished at: {backup_path}")


def train():
    runner: OnPolicyRunner

    env_class_name = args_cli.task
    env_cfg, agent_cfg = task_registry.get_cfgs(env_class_name)
    env_class = task_registry.get_task_class(env_class_name)

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.seed = agent_cfg.seed
    if args_cli.device:
            env_cfg.device = args_cli.device
            if hasattr(env_cfg, "sim") and hasattr(env_cfg.sim, "device"):
                env_cfg.sim.device = args_cli.device
            
            agent_cfg.device = args_cli.device
            print(f"[INFO] Overriding device settings for play script to use: '{args_cli.device}'")
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.scene.seed = seed
        agent_cfg.seed = seed

    env = env_class(env_cfg, args_cli.headless)

    log_root_path = os.path.join("logs", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    
    if agent_cfg.resume:
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)
    if args_cli.backup_env:
        backup_current_env(log_dir, env_cfg, env_class)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    print("[INFO] Training complete. Closing simulation app.")

if __name__ == "__main__":
    train()

