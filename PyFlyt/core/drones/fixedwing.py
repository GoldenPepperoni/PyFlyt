from __future__ import annotations

import numpy as np
import yaml
from pybullet_utils import bullet_client

from ..abstractions.base_drone import DroneClass
from ..abstractions.camera import Camera
from ..abstractions.lifting_surfaces import LiftingSurface, LiftingSurfaces
from ..abstractions.motors import Motors


class FixedWing(DroneClass):
    """FixedWing instance that handles everything about a FixedWing."""

    def __init__(
        self,
        p: bullet_client.BulletClient,
        start_pos: np.ndarray,
        start_orn: np.ndarray,
        ctrl_hz: int,
        physics_hz: int,
        drone_model: str = "fixedwing",
        model_dir: None | str = None,
        use_camera: bool = False,
        use_gimbal: bool = False,
        camera_angle_degrees: int = 0,
        camera_FOV_degrees: int = 90,
        camera_resolution: tuple[int, int] = (128, 128),
        np_random: None | np.random.RandomState = None,
    ):
        """Creates a fixed wing UAV and handles all relevant control and physics.

        Args:
            p (bullet_client.BulletClient): p
            start_pos (np.ndarray): start_pos
            start_orn (np.ndarray): start_orn
            ctrl_hz (int): ctrl_hz
            physics_hz (int): physics_hz
            model_dir (None | str): model_dir
            drone_model (str): drone_model
            use_camera (bool): use_camera
            use_gimbal (bool): use_gimbal
            camera_angle_degrees (int): camera_angle_degrees
            camera_FOV_degrees (int): camera_FOV_degrees
            camera_resolution (tuple[int, int]): camera_resolution
            np_random (None | np.random.RandomState): np_random
        """
        super().__init__(
            p=p,
            start_pos=start_pos,
            start_orn=start_orn,
            ctrl_hz=ctrl_hz,
            physics_hz=physics_hz,
            model_dir=model_dir,
            drone_model=drone_model,
            np_random=np_random,
        )

        """Reads fixedwing.yaml file and load UAV parameters"""
        with open(self.param_path, "rb") as f:
            # load all params from yaml
            all_params = yaml.safe_load(f)

            # all lifting surfaces
            surfaces = list()
            surfaces.append(
                LiftingSurface(
                    p=self.p,
                    physics_period=self.physics_period,
                    np_random=self.np_random,
                    uav_id=self.Id,
                    surface_id=3,
                    command_id=1,
                    command_sign=+1.0,
                    lifting_vector=np.array([0.0, 0.0, 1.0]),
                    forward_vector=np.array([0.0, 1.0, 0.0]),
                    aerofoil_params=all_params["left_wing_flapped_params"],
                )
            )
            surfaces.append(
                LiftingSurface(
                    p=self.p,
                    physics_period=self.physics_period,
                    np_random=self.np_random,
                    uav_id=self.Id,
                    surface_id=4,
                    command_id=1,
                    command_sign=-1.0,
                    lifting_vector=np.array([0.0, 0.0, 1.0]),
                    forward_vector=np.array([0.0, 1.0, 0.0]),
                    aerofoil_params=all_params["right_wing_flapped_params"],
                )
            )
            surfaces.append(
                LiftingSurface(
                    p=self.p,
                    physics_period=self.physics_period,
                    np_random=self.np_random,
                    uav_id=self.Id,
                    surface_id=1,
                    command_id=0,
                    command_sign=-1.0,
                    lifting_vector=np.array([0.0, 0.0, 1.0]),
                    forward_vector=np.array([0.0, 1.0, 0.0]),
                    aerofoil_params=all_params["horizontal_tail_params"],
                )
            )
            surfaces.append(
                LiftingSurface(
                    p=self.p,
                    physics_period=self.physics_period,
                    np_random=self.np_random,
                    uav_id=self.Id,
                    surface_id=5,
                    command_id=None,
                    command_sign=+1.0,
                    lifting_vector=np.array([0.0, 0.0, 1.0]),
                    forward_vector=np.array([0.0, 1.0, 0.0]),
                    aerofoil_params=all_params["main_wing_params"],
                )
            )
            surfaces.append(
                LiftingSurface(
                    p=self.p,
                    physics_period=self.physics_period,
                    np_random=self.np_random,
                    uav_id=self.Id,
                    surface_id=2,
                    command_id=2,
                    command_sign=-1.0,
                    lifting_vector=np.array([1.0, 0.0, 0.0]),
                    forward_vector=np.array([0.0, 1.0, 0.0]),
                    aerofoil_params=all_params["vertical_tail_params"],
                )
            )
            self.lifting_surfaces = LiftingSurfaces(lifting_surfaces=surfaces)

            # motor
            motor_params = all_params["motor_params"]
            tau = np.array([motor_params["tau"]])
            max_rpm = np.array([1.0]) * np.sqrt(
                (motor_params["total_thrust"]) / motor_params["thrust_coef"]
            )
            thrust_coef = np.array([[0.0, 1.0, 0.0]]) * motor_params["thrust_coef"]
            torque_coef = np.array([[0.0, 1.0, 0.0]]) * motor_params["torque_coef"]
            noise_ratio = np.array([motor_params["noise_ratio"]])
            self.motors = Motors(
                p=self.p,
                physics_period=self.physics_period,
                np_random=self.np_random,
                uav_id=self.Id,
                motor_ids=[0],
                tau=tau,
                max_rpm=max_rpm,
                thrust_coef=thrust_coef,
                torque_coef=torque_coef,
                noise_ratio=noise_ratio,
            )

        """ CAMERA """
        self.use_camera = use_camera
        if self.use_camera:
            self.camera = Camera(
                p=self.p,
                uav_id=self.Id,
                camera_id=0,
                use_gimbal=use_gimbal,
                camera_FOV_degrees=camera_FOV_degrees,
                camera_angle_degrees=camera_angle_degrees,
                camera_resolution=camera_resolution,
                camera_position_offset=np.array([0.0, -3.0, 1.0]),
                is_tracking_camera=True,
            )

        self.reset()

    def reset(self):
        self.set_mode(0)
        self.setpoint = np.zeros((4))
        self.cmd = np.zeros((4))

        self.p.resetBasePositionAndOrientation(self.Id, self.start_pos, self.start_orn)
        self.p.resetBaseVelocity(self.Id, [0, 20, 0], [0, 0, 0])
        self.disable_artificial_damping()
        self.lifting_surfaces.reset()
        self.motors.reset()
        self.update_state()

        if self.use_camera:
            self.rgbaImg, self.depthImg, self.segImg = self.camera.capture_image()

    def update_state(self):
        """ang_vel, ang_pos, lin_vel, lin_pos"""
        lin_pos, ang_pos = self.p.getBasePositionAndOrientation(self.Id)
        lin_vel, ang_vel = self.p.getBaseVelocity(self.Id)

        # express vels in local frame
        rotation = np.array(self.p.getMatrixFromQuaternion(ang_pos)).reshape(3, 3).T
        lin_vel = np.matmul(rotation, lin_vel)
        ang_vel = np.matmul(rotation, ang_vel)

        # ang_pos in euler form
        ang_pos = self.p.getEulerFromQuaternion(ang_pos)

        # create the state
        self.state = np.stack([ang_vel, ang_pos, lin_vel, lin_pos], axis=0)

        # update all lifting surface velocities
        self.lifting_surfaces.update_local_surface_velocities(rotation)

        # update auxiliary information
        self.aux_state = np.concatenate(
            (self.lifting_surfaces.get_states(), self.motors.get_states())
        )

    def update_control(self):
        """runs through controllers"""
        # the default mode
        if self.mode == 0:
            self.cmd = self.setpoint
            return

        # otherwise, check that we have a custom controller
        if self.mode not in self.registered_controllers.keys():
            raise ValueError(
                f"Don't have other modes aside from 0, received {self.mode}."
            )

        # custom controllers run if any
        self.cmd = self.instanced_controllers[self.mode].step(self.state, self.setpoint)

    def update_forces(self):
        """Calculates and applies forces acting on UAV"""
        assert self.cmd[3] >= 0.0, f"thrust `{self.cmd[3]}` must be more than 0.0."

        self.lifting_surfaces.cmd2forces(self.cmd)
        self.motors.pwm2forces(self.cmd[[3]])

    def update_physics(self):
        """update_physics."""
        self.update_state()
        self.update_forces()

    def update_avionics(self):
        """
        updates state and control
        """
        self.update_control()

        if self.use_camera:
            self.rgbaImg, self.depthImg, self.segImg = self.camera.capture_image()
