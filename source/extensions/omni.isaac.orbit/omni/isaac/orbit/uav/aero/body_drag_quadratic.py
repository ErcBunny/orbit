# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from .body_drag_base import BodyDragBase

if TYPE_CHECKING:
    from .body_drag_cfg import BodyDragQuadraticCfg


class BodyDragQuadratic(BodyDragBase):
    # basic params
    cfg: BodyDragQuadraticCfg
    num_envs: int
    device: str

    # coefficient
    k: torch.Tensor

    # pre-allocated tensor
    torque: torch.Tensor

    def __init__(self, cfg: BodyDragQuadraticCfg, num_envs: int, device: str):
        # init base class
        super().__init__()

        # save cfg
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device

        # coefficient
        self.k = torch.tensor(
            [
                0.5 * cfg.air_density * cfg.k_xy * cfg.a_x,
                0.5 * cfg.air_density * cfg.k_xy * cfg.a_y,
                0.5 * cfg.air_density * cfg.k_z * cfg.a_z,
            ],
            device=device,
        )

        # torque
        self.torque = torch.zeros(num_envs, 3, device=device)

    def compute(self, lin_vel: torch.Tensor, ang_vel: torch.Tensor) -> torch.Tensor:
        force = -self.k * lin_vel * torch.abs(lin_vel)

        return force, self.torque
