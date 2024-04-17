# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from .propeller_base import PropellerBase

if TYPE_CHECKING:
    from .propeller_cfg import PropellerPolyCfg


class PropellerPoly(PropellerBase):
    # basic params
    cfg: PropellerPolyCfg
    num_envs: int
    device: str

    wrench: torch.Tensor  # (num_envs, num_props, 6)

    torque_direction: torch.Tensor

    def __init__(self, cfg: PropellerPolyCfg, num_envs: int, device: str):
        # init base class
        super().__init__()

        # save cfg
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device

        # force and torque direction
        self.torque_direction = -torch.tensor(cfg.propeller_dir, device=device)

        # zero-init wrench tensor
        self.wrench = torch.zeros(num_envs, cfg.num_props, 6, device=device)

    def compute(self, rpm: torch.Tensor) -> torch.Tensor:
        # compute force and torque
        force = (-1) * (
            self.cfg.k_force_quadratic * torch.pow(rpm, 2)
            + self.cfg.k_force_linear * rpm
        )
        torque = (
            self.cfg.k_torque_quadratic * torch.pow(rpm, 2)
            + self.cfg.k_torque_linear * rpm
        ) * self.torque_direction

        # update wrench
        self.wrench[:, :, 2] = force
        self.wrench[:, :, 5] = torque

        return self.wrench
