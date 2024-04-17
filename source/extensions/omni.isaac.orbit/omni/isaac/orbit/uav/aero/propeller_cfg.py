# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from omni.isaac.orbit.utils import configclass


@configclass
class PropellerBaseCfg:
    num_props: int = 4
    """Number of propellers per environment."""

    propeller_dir: list[int] = [1, -1, -1, 1]
    """Direction of angular velocity vector relative to body-z (down) axis."""


@configclass
class PropellerPolyCfg(PropellerBaseCfg):
    """Configuration for propeller polynomial wrench model.

    Source of defaults:
    - T-MOTOR AIR GEAR 450II benchtest
    """

    k_force_quadratic: float = 1.6141e-07
    """Quadratic term of the force polynomial model."""

    k_force_linear: float = -3.9834e-05
    """Linear term of the force polynomial model."""

    k_torque_quadratic: float = 2.0150e-09
    """Quadratic term of the torque polynomial model."""

    k_torque_linear: float = 2.9231e-07
    """Linear term of the torque polynomial model."""
