# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from omni.isaac.orbit.utils import configclass
from ..utils import FirstOrderLowPassFilterCfg, FirstOrderLowPassFilter


@configclass
class SimpleBfRateCtrlCfg:
    """Configuration for simplified betaflight rate PID controller."""

    # control period
    update_dt: float = 0.002
    """Controller update time interval in seconds.
    """

    # d-term low pass filter
    dterm_lpf_cutoff: float = 200.0
    """The d-term LPF cutoff frequency in Hz."""

    # stick position to angular rates (deg/s)
    # The pilot can adjust their Rates to suit their flying style. Typically:
    # racers prefer a more linear curve with a maximum turn rate of around 550-650 deg/s
    # freestyle typically uses a combination of a soft center region with high maximum turn rates (850-1200 deg/s)
    # cinematic flying will be smoother with a flatter center region.
    center_sensitivity_roll: float = 100.0
    max_rate_roll: float = 670.0
    rate_expo_roll: float = 0.6

    center_sensitivity_pitch: float = 100.0
    max_rate_pitch: float = 670.0
    rate_expo_pitch: float = 0.6

    center_sensitivity_yaw: float = 100.0
    max_rate_yaw: float = 670.0
    rate_expo_yaw: float = 0.6

    # PID (rad)
    kp_roll: float = 150.0
    ki_roll: float = 2.0
    kd_roll: float = 2.0
    kff_roll: float = 0.0
    iterm_lim_roll: float = 10.0
    pid_sum_lim_roll: float = 1000

    kp_pitch: float = 150.0
    ki_pitch: float = 2.0
    kd_pitch: float = 2.0
    kff_pitch: float = 0.0
    iterm_lim_pitch: float = 10.0
    pid_sum_lim_pitch: float = 1000

    kp_yaw: float = 100.0
    ki_yaw: float = 15.0
    kd_yaw: float = 0.0
    kff_yaw: float = 0.0
    iterm_lim_yaw: float = 10.0
    pid_sum_lim_yaw: float = 1000

    # rotor positions in body FRD frame
    # all rotors are assumed to only produce thrust along the body-z axis
    # so z component does not matter anyway
    # rotor indexing: https://betaflight.com/docs/wiki/configurator/motors-tab
    rotors_x: list[float] = [-0.099, 0.099, -0.099, 0.099]
    rotors_y: list[float] = [0.099, 0.099, -0.099, -0.099]
    rotors_dir: list[int] = [1, -1, -1, 1]
    pid_sum_mixer_scale: float = 1000.0

    # output idle
    output_idle: float = 0.05

    # throttle boost
    throttle_boost_gain: float = 10.0
    throttle_boost_freq: float = 50.0

    # thrust linearization
    thrust_linearization_gain: float = 0.4


class SimpleBfRateCtrl:
    """Simplified Betaflight rate PID control.

    I/O:
        - In: normalized stick positions in AETR channels, from -1 to 1.
        - Out: normalized rotor command of the rotors (u from 0 to 1).

    Implemented:
        - Actual rates mapping from AETR to angular velocity.
        - Angular rate PID with error-derivative I-term, D-term LPF, and FF based on setpoint value.
        - Mixing supporting customizable airframe.
        - AirMode using betaflight default (LEGACY) mixer adjustment.
        - Throttle Boost: throttle command is boosted by high-frequency component of itself.
        - Thrust Linearization: boosting output at low throttle, and lowering it at high throttle.

    Not implemented:
        - Antigravity: boosting PI during fast throttle movement.
        - Throttle PID Attenuation: reducing PID at high throttle to cope with motor noise.
        - I-term relax: disabling I-term calculation during fast maneuvers.
        - Dynamic damping: higher D-term coefficient during fast maneuvers.
        - Integrated yaw: integrating PID sum about z-axis before putting it into the mixer.
        - Absolute control: for better tracking to sticks, particularly during rotations involving fast yaw movement.
        - Sensor noise (gyro noise) and additional filtering (gyro filters, notch filters).
        - Dynamic Idle: controlling the minimum command level using PID to prevent motor-ESC de-synchronization.
        - Battery voltage compensation: for consistent response throughout a battery run.

    Reference:
        [1] https://betaflight.com/docs/wiki
        [2] https://www.desmos.com/calculator/r5pkxlxhtb
        [3] https://en.wikipedia.org/wiki/Low-pass_filter
    """

    # basic settings
    cfg: SimpleBfRateCtrlCfg
    num_envs: int
    device: str

    # input
    command: torch.Tensor

    # rates profile
    center_sensitivity: torch.Tensor
    max_rate: torch.Tensor
    rate_expo: torch.Tensor

    # PID
    kp: torch.Tensor
    ki: torch.Tensor
    kd: torch.Tensor
    kff: torch.Tensor
    iterm_lim: torch.Tensor
    pid_sum_lim: torch.Tensor
    int_err_ang_vel: torch.Tensor
    last_ang_vel: torch.Tensor
    dterm_lpf: FirstOrderLowPassFilter

    # mixing
    num_rotors: int
    mix_tab: torch.Tensor

    # throttle boost
    throttle_boost_lpf: FirstOrderLowPassFilter

    # thrust linearization
    thrust_lin_throttle_compensation: float

    def __init__(self, cfg: SimpleBfRateCtrlCfg, num_envs: int, device: str):
        """Constructor."""

        # basic params
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device

        # rate mapping
        self.center_sensitivity = torch.tensor(
            [
                cfg.center_sensitivity_roll,
                cfg.center_sensitivity_pitch,
                cfg.center_sensitivity_yaw,
            ],
            device=device,
        )
        self.max_rate = torch.tensor(
            [cfg.max_rate_roll, cfg.max_rate_pitch, cfg.max_rate_yaw], device=device
        )
        self.rate_expo = torch.tensor(
            [cfg.rate_expo_roll, cfg.rate_expo_pitch, cfg.rate_expo_yaw], device=device
        )

        # pid
        self.kp = torch.tensor([cfg.kp_roll, cfg.kp_pitch, cfg.kp_yaw], device=device)
        self.ki = torch.tensor([cfg.ki_roll, cfg.ki_pitch, cfg.ki_yaw], device=device)
        self.kd = torch.tensor([cfg.kd_roll, cfg.kd_pitch, cfg.kd_yaw], device=device)
        self.kff = torch.tensor(
            [cfg.kff_roll, cfg.kff_pitch, cfg.kff_yaw], device=device
        )
        self.iterm_lim = torch.tensor(
            [cfg.iterm_lim_roll, cfg.iterm_lim_pitch, cfg.iterm_lim_yaw], device=device
        )
        self.pid_sum_lim = torch.tensor(
            [
                cfg.pid_sum_lim_roll,
                cfg.pid_sum_lim_pitch,
                cfg.pid_sum_lim_yaw,
            ],
            device=device,
        )
        self.int_err_ang_vel = torch.zeros(num_envs, 3, device=device)
        self.last_ang_vel = torch.zeros(num_envs, 3, device=device)
        dterm_lpf_cfg = FirstOrderLowPassFilterCfg(
            update_dt=cfg.update_dt,
            cutoff_freq=cfg.dterm_lpf_cutoff,
            init_value=0.0,
        )
        self.dterm_lpf = FirstOrderLowPassFilter(
            cfg=dterm_lpf_cfg, data_size=self.last_ang_vel.size(), device=device
        )

        # mixing table
        if not (
            len(cfg.rotors_x) == len(cfg.rotors_y)
            and len(cfg.rotors_y) == len(cfg.rotors_dir)
        ):
            raise ValueError("Rotors configuration error.")
        self.num_rotors = len(cfg.rotors_x)
        rotors_x_abs = [abs(item) for item in cfg.rotors_x]
        rotors_y_abs = [abs(item) for item in cfg.rotors_y]
        scale = max(max(rotors_x_abs), max(rotors_y_abs))
        table_data = []
        for i in range(self.num_rotors):
            table_data.append(
                [
                    1,  # throttle
                    -cfg.rotors_y[i] / scale,  # roll
                    cfg.rotors_x[i] / scale,  # pitch
                    -cfg.rotors_dir[i],  # yaw
                ]
            )
        self.mix_tab = torch.tensor(table_data, device=device)

        # throttle boost
        throttle_boost_lpf_cfg = FirstOrderLowPassFilterCfg(
            update_dt=cfg.update_dt,
            cutoff_freq=cfg.throttle_boost_freq,
            init_value=0.0,
        )
        self.throttle_boost_lpf = FirstOrderLowPassFilter(
            cfg=throttle_boost_lpf_cfg, data_size=torch.Size([num_envs]), device=device
        )

        # thrust linearization
        # betaflight pid_init.c
        """
        #ifdef USE_THRUST_LINEARIZATION
            pidRuntime.thrustLinearization = pidProfile->thrustLinearization / 100.0f;
            pidRuntime.throttleCompensateAmount = pidRuntime.thrustLinearization - 0.5f * sq(pidRuntime.thrustLinearization);
        #endif
        """
        self.thrust_lin_throttle_compensation = (
            cfg.thrust_linearization_gain - 0.5 * cfg.thrust_linearization_gain**2
        )

    def set_command(self, command: torch.Tensor):
        """Set the command (stick positions).

        Args:
            command: normalized stick positions in tensor shaped (num_envs, 4).
        """

        # set command
        self.command = command

    def compute(self, ang_vel: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """Run main controller logic.

        Args:
            ang_vel: the current sensed angular velocity in rad/s, shaped (num_envs, 3).

        Returns:
            - Desired angular velocity in rad/s, shaped (num_envs, 3).
            - Normalized rotor command u[0, 1], shaped (num_envs, num_rotors).
        """

        # map stick positions to desired body angular velocity
        # https://betaflight.com/docs/wiki/guides/current/Rate-Calculator
        # assuming FRD body frame
        # channel A -> roll (body x)
        # channel E -> pitch (body y)
        # channel R -> yaw (body z)
        # let x[-1, 1] be the stick position, d the center sensitivity, f the max rate, g the expo
        # desired body rate = sgn(x) * ( d|x| + (f-d) ( (1-g)x^2 + gx^6 ) )
        cmd_aer = self.command[:, [0, 1, 3]]
        des_body_rates = torch.sgn(cmd_aer) * (
            self.center_sensitivity * torch.abs(cmd_aer)
            + (self.max_rate - self.center_sensitivity)
            * (
                (1 - self.rate_expo) * torch.pow(cmd_aer, 2)
                + self.rate_expo * torch.pow(cmd_aer, 6)
            )
        )
        des_ang_vel = torch.deg2rad(des_body_rates)

        # angular velocity error
        err_ang_vel = des_ang_vel - ang_vel

        # integral of error rate, and limit the integral amount
        self.int_err_ang_vel += err_ang_vel
        torch.clamp(
            input=self.int_err_ang_vel,
            min=-self.iterm_lim,
            max=self.iterm_lim,
            out=self.int_err_ang_vel,
        )

        # PID sum and clamp
        pid_sum = (
            self.kp * err_ang_vel
            + self.ki * self.int_err_ang_vel
            - self.kd * self.dterm_lpf.get_output()
            + self.kff * des_ang_vel
        )
        torch.clamp(
            input=pid_sum, min=-self.pid_sum_lim, max=self.pid_sum_lim, out=pid_sum
        )

        # update dterm low-pass filter
        self.dterm_lpf.update((ang_vel - self.last_ang_vel) / self.cfg.update_dt)
        self.last_ang_vel = ang_vel

        # scale the PID sum before mixing
        pid_sum /= self.cfg.pid_sum_mixer_scale

        # find desired motor command from RPY PID, shape (num_envs, num_rotors)
        rpy_u = torch.matmul(self.mix_tab[:, 1:], pid_sum.T).T

        # u range for each environment, shape (num_envs, )
        rpy_u_max = torch.max(rpy_u, 1).values
        rpy_u_min = torch.min(rpy_u, 1).values
        rpy_u_range = rpy_u_max - rpy_u_min

        # normalization factor
        norm_factor = 1 / rpy_u_range  # (num_envs, )
        torch.clamp(input=norm_factor, max=1.0, out=norm_factor)

        # mixer adjustment
        rpy_u_normalized = norm_factor.view(-1, 1) * rpy_u
        rpy_u_normalized_max = norm_factor * rpy_u_max
        rpy_u_normalized_min = norm_factor * rpy_u_min

        # throttle boost
        # betaflight mixer.c
        """
        #if defined(USE_THROTTLE_BOOST)
            if (throttleBoost > 0.0f) {
                const float throttleHpf = throttle - pt1FilterApply(&throttleLpf, throttle);
                throttle = constrainf(throttle + throttleBoost * throttleHpf, 0.0f, 1.0f);
            }
        #endif
        """
        cmd_t = (self.command[:, 2] + 1) / 2  # (num_envs, )
        throttle_low_freq_component = self.throttle_boost_lpf.get_output()
        throttle_high_freq_component = cmd_t - throttle_low_freq_component
        throttle = cmd_t + self.cfg.throttle_boost_gain * throttle_high_freq_component
        torch.clamp(input=throttle, min=0.0, max=1.0, out=throttle)
        self.throttle_boost_lpf.update(cmd_t)

        # thrust linearization step 1
        # betaflight mixer.c
        """
        #ifdef USE_THRUST_LINEARIZATION
            // reduce throttle to offset additional motor output
            throttle = pidCompensateThrustLinearization(throttle);
        #endif
        """
        # betaflight pid.c
        """
        #ifdef USE_THRUST_LINEARIZATION
        float pidCompensateThrustLinearization(float throttle)
        {
            if (pidRuntime.thrustLinearization != 0.0f) {
                // for whoops where a lot of TL is needed, allow more throttle boost
                const float throttleReversed = (1.0f - throttle);
                throttle /= 1.0f + pidRuntime.throttleCompensateAmount * sq(throttleReversed);
            }
            return throttle;
        }
        """
        throttle /= 1 + self.thrust_lin_throttle_compensation * torch.pow(
            1 - throttle, 2
        )

        # constrain throttle so it won't clip any outputs
        torch.clamp(
            input=throttle,
            min=-rpy_u_normalized_min,
            max=(1 - rpy_u_normalized_max),
            out=throttle,
        )

        # synthesize output
        u_rpy_t = rpy_u_normalized + throttle.view(-1, 1)

        # thrust linearization step 2
        # betaflight mixer.c
        """
        #ifdef USE_THRUST_LINEARIZATION
                motorOutput = pidApplyThrustLinearization(motorOutput);
        #endif
        """
        # betaflight pid.c
        """
        float pidApplyThrustLinearization(float motorOutput)
        {
            motorOutput *= 1.0f + pidRuntime.thrustLinearization * sq(1.0f - motorOutput);
            return motorOutput;
        }
        """
        u_rpy_t *= 1 + self.cfg.thrust_linearization_gain * torch.pow(1 - u_rpy_t, 2)

        # calculate final u based on idle
        u = self.cfg.output_idle + (1 - self.cfg.output_idle) * u_rpy_t

        # return results
        return des_ang_vel, u
