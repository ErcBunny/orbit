# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

import omni.isaac.orbit.sim as sim_utils
import torch
from omni.isaac.orbit.assets import RigidQuadCfg
from omni.isaac.orbit.sensors import CameraCfg
from omni.isaac.orbit.uav import (
    SimpleBfRateCtrlCfg,
    RotorPolyLagCfg,
    PropellerPolyCfg,
    BodyDragQuadraticCfg,
    WrenchCompositorCfg,
)
from omni.isaac.orbit.utils.math import quat_from_euler_xyz

# rotor geometry: compressed X
R = 0.125
THETA = math.radians(102)
X = R * math.cos(THETA / 2)
Y = R * math.sin(THETA / 2)

# camera tilt in degrees
CAM_ANGLE = math.radians(20)
CAM_QUAT = quat_from_euler_xyz(
    torch.tensor(0), torch.tensor(-CAM_ANGLE), torch.tensor(0)
)

KINGFISHER_CFG = RigidQuadCfg(
    spawn=sim_utils.RaceQuadcopterCfg(
        arm_length_front=R,
        arm_length_rear=R,
        arm_front_angle=THETA,
        motor_diameter=0.023,
        motor_height=0.006,
        propeller_diameter=0.12954,
        center_of_mass=(0.0, 0.0, 0.0),
        diagonal_inertia=(0.0025, 0.0021, 0.0043),
        principal_axes_rotation=(1.0, 0.0, 0.0, 0.0),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.752),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    ),
    uav_cfgs={
        "SimpleBfRateCtrlCfg": SimpleBfRateCtrlCfg(
            # rates roll
            center_sensitivity_roll=100.0,
            max_rate_roll=670.0,
            rate_expo_roll=0.0,
            # rates pitch
            center_sensitivity_pitch=100.0,
            max_rate_pitch=670.0,
            rate_expo_pitch=0.0,
            # rates yaw
            center_sensitivity_yaw=100.0,
            max_rate_yaw=670.0,
            rate_expo_yaw=0.0,
            # pid dterm lpf
            dterm_lpf_cutoff=200,
            # pid roll
            kp_roll=150.0,
            ki_roll=2.0,
            kd_roll=2.0,
            kff_roll=0.0,
            iterm_lim_roll=10.0,
            pid_sum_lim_roll=1000,
            # pid pitch
            kp_pitch=150.0,
            ki_pitch=2.0,
            kd_pitch=2.0,
            kff_pitch=0.0,
            iterm_lim_pitch=10.0,
            pid_sum_lim_pitch=1000,
            # pid yaw
            kp_yaw=100.0,
            ki_yaw=15.0,
            kd_yaw=0.0,
            kff_yaw=0.0,
            iterm_lim_yaw=10.0,
            pid_sum_lim_yaw=1000,
            # mixer
            rotors_x=[-X, X, -X, X],
            rotors_y=[Y, Y, -Y, -Y],
            # output idle
            output_idle=0.05,
            # throttle boost
            throttle_boost_gain=10.0,
            throttle_boost_freq=50.0,
            # thrust linearization
            thrust_linearization_gain=0.4,
        ),
        "RotorPolyLagCfg": RotorPolyLagCfg(
            # https://store.tmotor.com/goods.php?id=1106
            # V2306 V2 KV2400 T5147
            spinup_time_constant=0.033,
            slowdown_time_constant=0.033,
            k_rpm_quadratic=-13421.95,
            k_rpm_linear=37877.42,
            rotor_diagonal_inertia_x=0.0,
            rotor_diagonal_inertia_y=0.0,
            rotor_diagonal_inertia_z=9.3575e-6,
            rotor_principal_axes_rotation_qw=1.0,
            rotor_principal_axes_rotation_qx=0.0,
            rotor_principal_axes_rotation_qy=0.0,
            rotor_principal_axes_rotation_qz=0.0,
        ),
        "PropellerPolyCfg": PropellerPolyCfg(
            # https://store.tmotor.com/goods.php?id=1106
            # V2306 V2 KV2400 T5147
            k_force_quadratic=2.1549e-08,
            k_force_linear=-4.5101e-05,
            k_torque_quadratic=2.1549e-08 * 0.022,
            k_torque_linear=-4.5101e-05 * 0.022,
        ),
        "WrenchCompositorCfg": WrenchCompositorCfg(
            pos_x=[-X, X, -X, X],
            pos_y=[Y, Y, -Y, -Y],
        ),
        "BodyDragQuadraticCfg": BodyDragQuadraticCfg(),
    },
    camera_cfgs={
        "FPV": CameraCfg(
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=8.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 1.0e5),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.08, 0.0, 0.015),
                rot=(CAM_QUAT[0], CAM_QUAT[1], CAM_QUAT[2], CAM_QUAT[3]),
                convention="world",
            ),
        )
    },
)
