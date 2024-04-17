# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
from typing import TYPE_CHECKING

from ..rigid_object import RigidObject

if TYPE_CHECKING:
    from .rigid_quad_cfg import RigidQuadCfg


class RigidQuad(RigidObject):
    cfg: RigidQuadCfg

    def __init__(self, cfg: RigidQuadCfg):
        super().__init__(cfg)
