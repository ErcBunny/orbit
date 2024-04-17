# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from omni.isaac.orbit.utils import configclass


@configclass
class WrenchCompositorCfg:
    # wrench application positions in body FRD frame
    # the order should match that of the wrench
    # default values mean there are 4 positions of application
    # the n-th position is (pos_x[n], pos_y[n], pos_z[n]) w.r.t. body FRD frame
    pos_x: list[float] = [-0.099, 0.099, -0.099, 0.099]
    pos_y: list[float] = [0.099, 0.099, -0.099, -0.099]
    pos_z: list[float] = [0.0, 0.0, 0.0, 0.0]


class WrenchCompositor:
    cfg: WrenchCompositorCfg
    num_envs: int
    device: str

    num_positions: int

    r: torch.Tensor

    def __init__(self, cfg: WrenchCompositorCfg, num_envs, device):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device

        if not (len(cfg.pos_x) == len(cfg.pos_y) and len(cfg.pos_y) == len(cfg.pos_z)):
            raise ValueError("Configuration error.")
        num_positions = len(cfg.pos_x)
        self.num_positions = num_positions

        # from body origin to position (num_envs, num_positions, 3)
        self.r = torch.zeros(num_envs, num_positions, 3, device=device)
        self.r[:] = torch.tensor(
            [
                cfg.pos_x,
                cfg.pos_y,
                cfg.pos_z,
            ],
            device=device,
        ).T

    def compute(self, wrench: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """Compute total force and torque from scattered wrench.

        The total force is the sum of all scattered forces.
        The total torque is the sum of all scattered torques plus force-induced torques.

        Args:
            wrench: wrench applied to scattered positions, (num_envs, num_positions, 6).

        Returns:
            - Total force tensor (num_envs, 3) to be applied to the body frame origin.
            - Total torque tensor (num_envs, 3) to be applied to the body frame origin.
        """
        force = wrench[:, :, :3].sum(dim=1)
        torque = wrench[:, :, 3:].sum(dim=1) + torch.cross(
            self.r, wrench[:, :, :3], 2
        ).sum(dim=1)

        return force, torque
