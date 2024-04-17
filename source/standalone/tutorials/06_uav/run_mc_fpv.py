# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import time

import argparse
import json
import pygame
import torch
import zmq

"""Arguments."""

parser = argparse.ArgumentParser(description="Tutorial on using the UAV utilities.")
parser.add_argument(
    "--phy_dt", type=float, default=0.002, help="Time size of a physics step."
)
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to spawn."
)
parser.add_argument(
    "--device", type=str, default="cuda", help="Either ``cpu`` or ``cuda``."
)
parser.add_argument("--fps", type=float, default=50, help="Desired FPS of rendering.")
parser.add_argument(
    "--debug",
    type=bool,
    default=False,
    help="If True, data will be sent to PlotJuggler and will slow down simulation.",
    # TODO: how to plot data without slowing down the sim?
)
parser.add_argument(
    "--env_spacing",
    type=float,
    default=1.0,
    help="Visual spacing between each environment.",
)
parser.add_argument(
    "--joystick_deadzone",
    type=float,
    default=0.01,
    help="Values below the deadzone will be regarded as 0.",
)
parser.add_argument(
    "--aileron_channel",
    type=int,
    default=1,
    help="The channel number for aileron on the RC.",
)
parser.add_argument(
    "--elevator_channel",
    type=int,
    default=2,
    help="The channel number for elevator on the RC.",
)
parser.add_argument(
    "--throttle_channel",
    type=int,
    default=0,
    help="The channel number for throttle on the RC.",
)
parser.add_argument(
    "--rudder_channel",
    type=int,
    default=3,
    help="The channel number for rudder on the RC.",
)
parser.add_argument(
    "--aileron_dir", type=int, default=1, help="``1`` or ``-1`` (inverted)."
)
parser.add_argument(
    "--elevator_dir", type=int, default=-1, help="``1`` or ``-1`` (inverted)."
)
parser.add_argument(
    "--throttle_dir", type=int, default=1, help="``1`` or ``-1`` (inverted)."
)
parser.add_argument(
    "--rudder_dir", type=int, default=1, help="``1`` or ``-1`` (inverted)."
)
args_cli = parser.parse_args()

"""Utils."""

# pygame
pygame.init()
pygame.joystick.init()
joystick = pygame.joystick.Joystick(0)
joystick.init()

# zmq for PlotJuggler
zmq_context = zmq.Context()
zmq_socket = zmq_context.socket(zmq.PUB)
port = 9872
zmq_socket.bind("tcp://*:" + str(port))

"""Launch Isaac Sim Simulator."""

from omni.isaac.orbit.app import AppLauncher

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything else in Orbit follows."""

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.assets import AssetBaseCfg, RigidQuad, RigidQuadCfg
from omni.isaac.orbit.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.uav import (
    SimpleBfRateCtrl,
    RotorPolyLag,
    PropellerPoly,
    BodyDragQuadratic,
    WrenchCompositor,
)
from omni.isaac.orbit.sensors.camera.utils import convert_orientation_convention
from omni.isaac.orbit_assets import KINGFISHER_CFG


@configclass
class TestSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(size=(100, 100)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    quad = KINGFISHER_CFG
    quad.prim_path = "{ENV_REGEX_NS}/Quad"
    quad.init_state.pos = (0, 0, 0.2)


def run_simulator(
    sim_context: sim_utils.SimulationContext, interactive_scene: InteractiveScene
):
    """Setup and run the simulation loop."""

    # simulation stepping
    sim_dt = sim_context.get_physics_dt()
    framerate = args_cli.fps
    steps_per_render = int(1 / framerate / sim_dt)
    clk = pygame.time.Clock()
    print(f"Target FPS: {framerate}")
    print(f"Target Physics dt: {sim_dt}")
    print(f"Target Steps per render: {steps_per_render}")

    # init UAV pipeline
    quad: RigidQuad = interactive_scene["quad"]
    quad.cfg.uav_cfgs["SimpleBfRateCtrlCfg"].update_dt = sim_dt
    quad.cfg.uav_cfgs["RotorPolyLagCfg"].update_dt = sim_dt
    bf_rate_ctrl = SimpleBfRateCtrl(
        quad.cfg.uav_cfgs["SimpleBfRateCtrlCfg"], args_cli.num_envs, args_cli.device
    )
    rotor_model = RotorPolyLag(
        quad.cfg.uav_cfgs["RotorPolyLagCfg"], args_cli.num_envs, args_cli.device
    )
    prop_model = PropellerPoly(
        quad.cfg.uav_cfgs["PropellerPolyCfg"], args_cli.num_envs, args_cli.device
    )
    body_drag_model = BodyDragQuadratic(
        quad.cfg.uav_cfgs["BodyDragQuadraticCfg"], args_cli.num_envs, args_cli.device
    )
    wrench_compo = WrenchCompositor(
        quad.cfg.uav_cfgs["WrenchCompositorCfg"], args_cli.num_envs, args_cli.device
    )

    # FLU <-> FRD
    flu_frd = torch.tensor([[1.0, -1.0, -1.0]], device=args_cli.device)

    rc_command = torch.zeros(args_cli.num_envs, 4, device=args_cli.device)
    rc_command[:, 2] = -1

    # allocate tensor for data to plot
    plot_data = torch.zeros(steps_per_render, 2, 3, device=args_cli.device)

    # simulation loop
    while simulation_app.is_running():

        # get joysticks and set command
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break

        rc_aileron = joystick.get_axis(args_cli.aileron_channel) * args_cli.aileron_dir
        rc_elevator = (
            joystick.get_axis(args_cli.elevator_channel) * args_cli.elevator_dir
        )
        rc_throttle = (
            joystick.get_axis(args_cli.throttle_channel) * args_cli.throttle_dir
        )
        rc_rudder = joystick.get_axis(args_cli.rudder_channel) * args_cli.rudder_dir

        rc_command[:, 0] = rc_aileron
        rc_command[:, 1] = rc_elevator
        rc_command[:, 2] = rc_throttle
        rc_command[:, 3] = rc_rudder

        rc_command[:, [0, 1, 3]] *= (
            torch.abs(rc_command[:, [0, 1, 3]]) > args_cli.joystick_deadzone
        )

        # feed command into rate ctrl
        bf_rate_ctrl.set_command(rc_command)

        # physics
        for i in range(steps_per_render):
            # get angular velocity feedback
            ang_vel = quad.data.root_ang_vel_b * flu_frd  # FLU

            # run uav pipeline
            des_ang_vel, rotor_cmd = bf_rate_ctrl.compute(ang_vel)
            rpm, rotor_wrench = rotor_model.compute(rotor_cmd)
            prop_wrench = prop_model.compute(rpm)
            if args_cli.debug:
                plot_data[i, 0, :] = des_ang_vel[0]
                plot_data[i, 1, :] = ang_vel[0]

            # body drag
            lin_vel = quad.data.root_lin_vel_b * flu_frd
            drag_force, drag_torque = body_drag_model.compute(lin_vel, ang_vel)

            # wrench composition
            total_wrench = prop_wrench + rotor_wrench
            force, torque = wrench_compo.compute(total_wrench)
            force = flu_frd * (force + drag_force)
            torque = flu_frd * (torque + drag_torque)

            # apply wrench
            quad.set_external_force_and_torque(
                force.view(args_cli.num_envs, 1, 3),
                torque.view(args_cli.num_envs, 1, 3),
            )
            quad.write_data_to_sim()

            # perform step
            sim_context.step(False)

            # update buffers
            interactive_scene.update(sim_dt)

        # render frame
        sim_context.render()

        # send data to PlotJuggler via zmq
        if args_cli.debug:
            data_np = plot_data.cpu().numpy()
            for i in range(data_np.shape[0]):
                msg = {
                    "t": time.time() - args_cli.phy_dt * (data_np.shape[0] - i),
                    "des_ang_vel": {
                        "x": float(data_np[i, 0, 0]),
                        "y": float(data_np[i, 0, 1]),
                        "z": float(data_np[i, 0, 2]),
                    },
                    "ang_vel": {
                        "x": float(data_np[i, 1, 0]),
                        "y": float(data_np[i, 1, 1]),
                        "z": float(data_np[i, 1, 2]),
                    },
                }
                zmq_socket.send_string(json.dumps(msg))

        # limit render framerate
        clk.tick(framerate)


if __name__ == "__main__":
    # load kit helper
    sim_cfg = sim_utils.SimulationCfg(
        dt=args_cli.phy_dt,
        device=args_cli.device,
        use_gpu_pipeline=(args_cli.device == "cuda"),
    )
    sim = sim_utils.SimulationContext(sim_cfg)

    # set main camera
    sim.set_camera_view((1.5, 1.5, 1.5), (0.0, 0.0, 0.0))

    # design scene
    scene_cfg = TestSceneCfg(
        num_envs=args_cli.num_envs, env_spacing=args_cli.env_spacing
    )
    scene = InteractiveScene(scene_cfg)

    # add a camera prims but do not register sensors
    # this avoids slowing down the simulation
    cfg: RigidQuadCfg = scene["quad"].cfg
    if cfg.camera_cfgs is not None:
        for key, value in cfg.camera_cfgs.items():
            rot = torch.tensor(value.offset.rot, dtype=torch.float32).unsqueeze(0)
            rot_offset = convert_orientation_convention(
                rot, origin=value.offset.convention, target="opengl"
            )
            rot_offset = rot_offset.squeeze(0).numpy()
            sim_utils.spawn_camera(
                prim_path=cfg.prim_path + "/" + key,
                cfg=value.spawn,
                translation=value.offset.pos,
                orientation=rot_offset,
            )

    # play the simulator
    sim.reset()

    # now we are ready!
    print("[INFO]: Setup complete...")

    # run the simulator
    run_simulator(sim, scene)

    # close sim app
    simulation_app.close()

    # close zmq
    zmq_socket.close()
    zmq_context.term()
