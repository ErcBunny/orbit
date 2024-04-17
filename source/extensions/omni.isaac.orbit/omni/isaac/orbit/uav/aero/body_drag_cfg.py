# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from omni.isaac.orbit.utils import configclass


@configclass
class BodyDragBaseCfg:
    pass


@configclass
class BodyDragQuadraticCfg(BodyDragBaseCfg):
    air_density: float = 1.204
    """ISA air density [kg / m^3]."""

    k_xy: float = 1.04
    """Drag coefficient in body x/y plane."""

    k_z: float = 1.04
    """Drag coefficient in body z plane."""

    a_x: float = 1.5e-2
    """Area pushing against air in body x direction."""

    a_y: float = 1.5e-2
    """Area pushing against air in body y direction."""

    a_z: float = 1.5e-2
    """Area pushing against air in body z direction."""
