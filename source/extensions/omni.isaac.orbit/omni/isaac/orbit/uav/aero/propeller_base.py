# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from abc import ABC, abstractmethod


class PropellerBase(ABC):

    def __init__(self):
        """Constructor."""

        pass

    @abstractmethod
    def compute(self, rpm: torch.Tensor) -> torch.Tensor:
        """Main processing logic.

        Args:
            rpm: propeller RPM tensor in (num_envs, num_propellers).

        Returns:
            - 6-DOF wrench caused by propeller in (num_envs, num_propellers, 6).
        """

        raise NotImplementedError
