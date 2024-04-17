# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from omni.isaac.orbit.utils import configclass


@configclass
class RotorBaseCfg:
    num_rotors: int = 4
    """Number of rotors per environment."""

    rotor_dir: list[int] = [1, -1, -1, 1]
    """Direction of angular velocity vector relative to body-z (down) axis."""


@configclass
class RotorPolyLagCfg(RotorBaseCfg):
    """Configuration for polynomial RPM with first-order lag model.

    Source of defaults:
    - time constant: PX4-SITL IRIS model
    - polynomial: T-MOTOR AIR GEAR 450II benchtest
    - rotor inertia: Pegasus Simulator IRIS model
    """

    update_dt: float = 0.001
    """Update time interval."""

    spinup_time_constant: float = 0.0125
    """Time constant for rotor acceleration for first-order lag."""

    slowdown_time_constant: float = 0.025
    """Time constant for rotor deceleration for first-order lag."""

    k_rpm_quadratic: float = -3562.1
    """Quadratic term of the polynomial model."""

    k_rpm_linear: float = 12412.4
    """Linear term of the polynomial model."""

    rotor_diagonal_inertia_x: float = 7.46e-05
    """The (0, 0) element of the 3x3 diagonal inertia matrix."""

    rotor_diagonal_inertia_y: float = 1.39e-06
    """The (1, 1) element of the 3x3 diagonal inertia matrix."""

    rotor_diagonal_inertia_z: float = 7.57e-05
    """The (2, 2) element of the 3x3 diagonal inertia matrix."""

    rotor_principal_axes_rotation_qw: float = 1.0
    """The scalar component of the quaternion representing the same rotation as the principal axes matrix."""

    rotor_principal_axes_rotation_qx: float = 0.0
    """The x component of the quaternion representing the same rotation as the principal axes matrix."""

    rotor_principal_axes_rotation_qy: float = 0.0
    """The y component of the quaternion representing the same rotation as the principal axes matrix."""

    rotor_principal_axes_rotation_qz: float = 0.0
    """The z component of the quaternion representing the same rotation as the principal axes matrix."""
