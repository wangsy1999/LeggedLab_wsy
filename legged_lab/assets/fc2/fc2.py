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


import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from legged_lab.assets import ISAAC_ASSET_DIR

FC2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/fc2/bhr8_fc2_noarm.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=1
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.88),
        joint_pos={
            ".*hipPitch": -0.30,
            ".*knee": 0.6,
            ".*ankle1": -0.3,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.90,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*hipYaw",
                ".*hipRoll",
                ".*hipPitch",
                ".*knee",
            ],
            effort_limit_sim={
                ".*hipYaw": 88.0,
                ".*hipRoll": 139.0,
                ".*hipPitch": 88.0,
                ".*knee": 139.0,

            },
            velocity_limit_sim={
                ".*hipYaw": 32.0,
                ".*hipRoll": 32.0,
                ".*hipPitch": 32.0,
                ".*knee": 20.0,
            },
            stiffness={
                ".*hipYaw": 150.0,
                ".*hipRoll": 150.0,
                ".*hipPitch": 200.0,
                ".*knee": 200.0,
            },
            damping={
                ".*hipYaw": 2.5,
                ".*hipPitch": 2.5,
                ".*hipPitch": 2.5,
                ".*knee": 3.0,
            },
            armature=0.01,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*ankle1", ".*ankle2"],
            effort_limit_sim={
                ".*ankle1": 50.0,
                ".*ankle2": 50.0,
            },
            velocity_limit_sim={
                ".*ankle1": 37.0,
                ".*ankle2": 37.0,
            },
            stiffness=50.0,
            damping=2.0,
            armature=0.01,
        ),
    },
)
