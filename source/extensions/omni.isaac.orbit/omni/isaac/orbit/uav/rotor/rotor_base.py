# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from abc import ABC, abstractmethod


class RotorBase(ABC):

    def __init__(self):
        """Constructor."""

        pass

    @abstractmethod
    def compute(self, command: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """Main processing logic.

        Args:
            command: normalized rotor command in (num_envs, num_rotors).

        Returns:
            - Rotor RPM in (num_envs, num_rotors).
            - 6-DOF wrench caused by motor dynamics in (num_envs, num_rotors, 6).
        """

        raise NotImplementedError
