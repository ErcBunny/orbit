# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.sensors import CameraCfg
from typing import Any

from ..rigid_object import RigidObjectCfg
from .rigid_quad import RigidQuad


@configclass
class RigidQuadCfg(RigidObjectCfg):
    class_type: type = RigidQuad

    uav_cfgs: dict[str, Any] = MISSING

    camera_cfgs: dict[str, CameraCfg] | None = None
