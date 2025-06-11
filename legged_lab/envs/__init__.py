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

from legged_lab.envs.base.base_env import BaseEnv
from legged_lab.envs.base.base_env_config import BaseAgentCfg, BaseEnvCfg
from legged_lab.envs.g1.g1_config import (
    G1FlatAgentCfg,
    G1FlatEnvCfg,
    G1RoughAgentCfg,
    G1RoughEnvCfg,
)
from legged_lab.envs.gr2.gr2_config import (
    GR2FlatAgentCfg,
    GR2FlatEnvCfg,
    GR2RoughAgentCfg,
    GR2RoughEnvCfg,
)
from legged_lab.envs.h1.h1_config import (
    H1FlatAgentCfg,
    H1FlatEnvCfg,
    H1RoughAgentCfg,
    H1RoughEnvCfg,
)

from legged_lab.envs.fc2.fc2_config import(
    fc2FlatAgentCfg,
    fc2FlatEnvCfg,
    fc2RoughAgentCfg,
    fc2RoughEnvCfg,

)

from legged_lab.envs.fc2.fc2_env import fc2Env
from legged_lab.utils.task_registry import task_registry
from legged_lab.envs.fc2.fc2_config_diy import(
    bhr8FlatEnvCfg,
    bhr8FlatAgentCfg,
    bhr8RoughEnvCfg,
    bhr8RoughAgentCfg,
)
task_registry.register("h1_flat", BaseEnv, H1FlatEnvCfg(), H1FlatAgentCfg())
task_registry.register("h1_rough", BaseEnv, H1RoughEnvCfg(), H1RoughAgentCfg())
task_registry.register("g1_flat", BaseEnv, G1FlatEnvCfg(), G1FlatAgentCfg())
task_registry.register("g1_rough", BaseEnv, G1RoughEnvCfg(), G1RoughAgentCfg())
task_registry.register("gr2_flat", BaseEnv, GR2FlatEnvCfg(), GR2FlatAgentCfg())
task_registry.register("gr2_rough", BaseEnv, GR2RoughEnvCfg(), GR2RoughAgentCfg())
task_registry.register("fc2_flat", BaseEnv, fc2FlatEnvCfg(), fc2FlatAgentCfg())
task_registry.register("fc2_rough", BaseEnv, fc2RoughEnvCfg(), fc2RoughAgentCfg())
task_registry.register("bhr8_flat", fc2Env, bhr8FlatEnvCfg(), bhr8FlatAgentCfg())
task_registry.register("bhr8_rough", fc2Env, bhr8RoughEnvCfg(), bhr8RoughAgentCfg())