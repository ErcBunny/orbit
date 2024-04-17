# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from abc import ABC, abstractmethod


class BodyDragBase(ABC):

    def __init__(self):
        """Constructor."""

        pass

    @abstractmethod
    def compute(
        self, lin_vel: torch.Tensor, ang_vel: torch.Tensor
    ) -> (torch.Tensor, torch.Tensor):
        """Main processing logic.

        Args:
            lin_vel (torch.Tensor): Linear velocity in body frame (num_envs, 3).
            ang_vel (torch.Tensor): Angular velocity in body frame (num_envs, 3).

        Returns:
            - Drag force (num_envs, 3).
            - Drag torque (num_envs, 3).

        """

        raise NotImplementedError
