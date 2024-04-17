# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

import torch
from omni.isaac.orbit.utils.math import matrix_from_quat
from typing import TYPE_CHECKING
from .rotor_base import RotorBase

if TYPE_CHECKING:
    from .rotor_cfg import RotorPolyLagCfg


class RotorPolyLag(RotorBase):
    # basic params
    cfg: RotorPolyLagCfg
    num_envs: int
    device: str

    # first-order lag
    alpha_spinup: float
    alpha_slowdown: float

    # rotor inertia matrix
    rotor_inertia: torch.Tensor  # (num_envs, num_rotors, 3, 3)

    # multi-iter variable
    rpm: torch.Tensor  # (num_envs, num_rotors)
    rpm_ss: torch.Tensor  # (num_envs, num_rotors)
    omega_dot: torch.Tensor  # (num_envs, num_rotors, 3)
    wrench: torch.Tensor  # (num_envs, num_rotors, 6)

    # rotor direction tensor
    rotor_dir: torch.Tensor

    def __init__(self, cfg: RotorPolyLagCfg, num_envs: int, device: str):
        # init base class
        super().__init__()

        # basic params
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device

        # first-order lag param
        self.alpha_spinup = math.exp(-cfg.update_dt / cfg.spinup_time_constant)
        self.alpha_slowdown = math.exp(-cfg.update_dt / cfg.slowdown_time_constant)

        # rotor inertia matrix
        principal_axes_quat = torch.tensor(
            [
                cfg.rotor_principal_axes_rotation_qw,
                cfg.rotor_principal_axes_rotation_qx,
                cfg.rotor_principal_axes_rotation_qy,
                cfg.rotor_principal_axes_rotation_qz,
            ],
            device=device,
        )
        principal_axes_matrix = matrix_from_quat(principal_axes_quat)
        diag_inertia_matrix = torch.diag(
            torch.tensor(
                [
                    cfg.rotor_diagonal_inertia_x,
                    cfg.rotor_diagonal_inertia_y,
                    cfg.rotor_diagonal_inertia_z,
                ],
                device=device,
            )
        )
        inertia_matrix = (
            principal_axes_matrix @ diag_inertia_matrix @ principal_axes_matrix.T
        )
        self.rotor_inertia = torch.zeros(num_envs, cfg.num_rotors, 3, 3, device=device)
        self.rotor_inertia[:] = inertia_matrix

        # initialize zeros
        self.rpm = torch.zeros(num_envs, cfg.num_rotors, device=device)
        self.rpm_ss = torch.zeros(num_envs, cfg.num_rotors, device=device)
        self.omega_dot = torch.zeros(num_envs, cfg.num_rotors, 3, device=device)
        self.wrench = torch.zeros(num_envs, cfg.num_rotors, 6, device=device)

        # rotor direction tensor
        self.rotor_dir = torch.tensor(cfg.rotor_dir, device=device)

    def compute(self, command: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        # update rpm from first order lag
        rpm_to_ss = self.rpm_ss - self.rpm
        d_rpm = (
            (rpm_to_ss >= 0) * (1 - self.alpha_spinup)
            + (rpm_to_ss < 0) * (1 - self.alpha_slowdown)
        ) * rpm_to_ss
        self.rpm += d_rpm

        # wrench due to rotor acceleration
        self.omega_dot[:, :, -1] = (  # (num_envs, num_rotors)
            -d_rpm * 2 * torch.pi / 60 / self.cfg.update_dt * self.rotor_dir
        )
        self.wrench[:, :, 3:] = torch.matmul(
            self.rotor_inertia,  # (num_envs, num_rotors, 3, 3)
            self.omega_dot.unsqueeze(-1),  # (num_envs, num_rotors, 3, 1)
        ).squeeze(3)

        # target RPM, as input to the first order lag system
        self.rpm_ss = (
            self.cfg.k_rpm_quadratic * torch.pow(command, 2)
            + self.cfg.k_rpm_linear * command
        )

        return self.rpm, self.wrench
