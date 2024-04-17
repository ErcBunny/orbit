# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from omni.isaac.orbit.utils import configclass


@configclass
class FirstOrderLowPassFilterCfg:

    update_dt: float = 0.001

    cutoff_freq: float = 100.0

    init_value: float = 0.0


class FirstOrderLowPassFilter:

    cfg: FirstOrderLowPassFilterCfg
    data_size: torch.Size
    device: str

    alpha: float

    filter_output: torch.Tensor

    def __init__(self, cfg: FirstOrderLowPassFilterCfg, data_size: torch.Size, device):
        self.cfg = cfg
        self.data_size = data_size
        self.device = device

        self.alpha = 1 / (1 + 1 / (2 * torch.pi * cfg.cutoff_freq * cfg.update_dt))

        self.filter_output = self.cfg.init_value * torch.ones(data_size, device=device)

    def get_output(self) -> torch.Tensor:
        return self.filter_output

    def update(self, filter_input: torch.Tensor):
        self.filter_output += self.alpha * (filter_input - self.filter_output)
