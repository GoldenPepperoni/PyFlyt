from __future__ import annotations

import math
import os

import numpy as np
import pybullet as p

from .dubins_path_planner import plan_dubins_path


class DubinsPathHandler:
    def __init__(
        self,
        enable_render: bool,
        num_targets: int,
        use_yaw_targets: bool,
        goal_reach_distance: float,
        goal_reach_angle: float,
        flight_dome_size: float,
        turning_radius: float,
        path_step_size: float,
        start_pos: np.ndarray,
        np_random: np.random.Generator,
    ):
        # constants
        self.enable_render = enable_render
        self.num_targets = num_targets
        self.use_yaw_targets = use_yaw_targets
        self.goal_reach_distance = goal_reach_distance
        self.goal_reach_angle = goal_reach_angle
        self.flight_dome_size = flight_dome_size
        self.turning_radius = turning_radius
        self.path_step_size = path_step_size
        self.start_pos = start_pos
        self.np_random = np_random

        # the target visual
        file_dir = os.path.dirname(os.path.realpath(__file__))
        self.targ_obj_dir = os.path.join(file_dir, f"../models/target.urdf")

    def reset(self):
        """TARGET GENERATION"""
        # reset carrot position
        self.carrot = [0, 0, 0]

        # reset the error
        self.new_distance = 0.0
        self.old_distance = 0.0

        # we sample from polar coordinates to generate linear targets
        self.targets = np.zeros(shape=(self.num_targets, 3))
        thts = self.np_random.uniform(0.0, 2.0 * math.pi, size=(self.num_targets,))
        phis = self.np_random.uniform(0.0, 2.0 * math.pi, size=(self.num_targets,))
        for i, tht, phi in zip(range(self.num_targets), thts, phis):
            dist = self.np_random.uniform(low=1.0, high=self.flight_dome_size * 0.9)
            x = dist * math.sin(phi) * math.cos(tht)
            y = dist * math.sin(phi) * math.sin(tht)
            z = self.np_random.uniform(10, 30)

            # check for floor of z
            self.targets[i] = np.array([x, y, z if z > 0.1 else 0.1])

        # yaw targets (Must be true for Dubins path)
        if self.use_yaw_targets:
            self.yaw_targets = self.np_random.uniform(
                low=-math.pi, high=math.pi, size=(self.num_targets,)
            )

        curvature = 1 / self.turning_radius
        self.path = []
        self.closest_idx = 0

        # Handle first section (from spawn to target 1)
        # Start and Goal
        s_x = self.start_pos[0][0]
        g_x = self.targets[0][0]

        s_y = self.start_pos[0][1]
        g_y = self.targets[0][1]

        s_yaw = np.pi/2
        g_yaw = self.yaw_targets[0]

        # Get Z path coordinates
        s_z = self.start_pos[0][2]
        g_z = self.targets[0][2]

        # Get X and Y path coordinates (20 cm step)
        x, y, yaw, mode, path_lens = plan_dubins_path(s_x, s_y, s_yaw, g_x, g_y, g_yaw, curvature, step_size=self.path_step_size/self.turning_radius)
        z = np.linspace(s_z, g_z, num=len(x))

        # Combine into 3D coordinates
        for i in range(0, len(x)):
            self.path.append([x[i], y[i], z[i]])

        # Generate path for subsequent sections
        for sect in range(0, self.num_targets-1):
            # Start and Goal
            s_x = self.targets[sect][0]
            g_x = self.targets[sect+1][0]

            s_y = self.targets[sect][1]
            g_y = self.targets[sect+1][1]

            s_yaw = self.yaw_targets[sect]
            g_yaw = self.yaw_targets[sect+1]


            # Get Z path coordinates
            s_z = self.targets[sect][2]
            g_z = self.targets[sect+1][2]

            # Get X and Y path coordinates
            x, y, yaw, mode, path_lens = plan_dubins_path(s_x, s_y, s_yaw, g_x, g_y, g_yaw, curvature, step_size=self.path_step_size/self.turning_radius)
            z = np.linspace(s_z, g_z, num=len(x))

            # Combine into 3D coordinates
            for i in range(0, len(x)):
                self.path.append([x[i], y[i], z[i]])

        # if we are rendering, laod in the targets
        if self.enable_render:
            self.target_visual = []
            for target in self.targets:
                self.target_visual.append(
                    p.loadURDF(
                        self.targ_obj_dir,
                        basePosition=target,
                        useFixedBase=True,
                        globalScaling=self.goal_reach_distance / 4.0,
                    )
                )

            for i, visual in enumerate(self.target_visual):
                p.changeVisualShape(
                    visual,
                    linkIndex=-1,
                    rgbaColor=(0, 1 - (i / len(self.target_visual)), 0, 1),
                )
        
            # Render carrot
            self.carrotid = p.loadURDF(
                        self.targ_obj_dir,
                        basePosition=self.carrot,
                        useFixedBase=True,
                        globalScaling=self.goal_reach_distance / 4.0,
                    )
            p.changeVisualShape(self.carrotid, linkIndex=-1, rgbaColor=(1, 0, 0, 1))

            # Cast initial ray for cross track distance
            self.rayid = p.addUserDebugLine([0, 0, 0], [0, 0, 0])

            # Draw path
            for i in range(0, len(self.path)-1):
                p.addUserDebugLine(self.path[i], self.path[i+1], lifeTime=0)

        

    def get_carrot(
        self,
        lin_pos: np.ndarray,
        lookahead: int
    ):

        lookahead = lookahead / self.path_step_size # convert into idx (0.5m step)
        self._get_closest_point(lin_pos)

        try:
            self.carrot = self.path[int(self.closest_idx+lookahead)]
        except:
            self.carrot = self.path[-1]

        if self.enable_render:
            p.resetBasePositionAndOrientation(self.carrotid, self.carrot, [1, 1, 1, 1])
        
        return self.carrot


    def _get_closest_point(
        self,
        lin_pos: np.ndarray
    ):
        cross_track_error = []
        idx_list = []
        # Calculate distance to 10m behind and ahead of UAV, and choose shortest distance
        for test_idx in range(self.closest_idx-20, self.closest_idx+20):
            try:
                if test_idx > 0: # Don't jump to check final coords (negative idx)
                    error = np.linalg.norm(self.path[test_idx]-lin_pos)
                    cross_track_error.append(error)
                    idx_list.append(test_idx)
            except:
                pass

        self.closest_idx = idx_list[np.argmin(cross_track_error)]

        if self.enable_render:
            # Red line
            self.rayid = p.addUserDebugLine(lin_pos, self.path[self.closest_idx], lineColorRGB=[1, 0, 0], replaceItemUniqueId=self.rayid)
        


    def distance_to_target(
        self,
        ang_pos: np.ndarray,
        lin_pos: np.ndarray,
        quarternion: np.ndarray,
    ):
        # rotation matrix
        rotation = np.array(p.getMatrixFromQuaternion(quarternion)).reshape(3, 3).T

        # drone to target
        target_deltas = np.matmul(rotation, (self.targets - lin_pos).T).T

        # record distance to the next target
        self.old_distance = self.new_distance
        self.new_distance = float(np.linalg.norm(target_deltas[0]))

        if self.use_yaw_targets:
            yaw_errors = self.yaw_targets - ang_pos[-1]

            # rollover yaw
            yaw_errors[yaw_errors > math.pi] -= 2.0 * math.pi
            yaw_errors[yaw_errors < -math.pi] += 2.0 * math.pi
            yaw_errors=yaw_errors[:, None]

            # add the yaw delta to the target deltas
            target_deltas = np.concatenate([target_deltas, yaw_errors], axis=-1)
            # compute the yaw error scalar
            self.yaw_error_scalar = np.abs(yaw_errors[0])

        return target_deltas

    def progress_to_target(self):
        return self.old_distance - self.new_distance

    def target_reached(self):
        """target_reached."""
        if not self.new_distance < self.goal_reach_distance:
            return False

        if not self.use_yaw_targets:
            return True

        if self.yaw_error_scalar < self.goal_reach_angle:
            return True

        return False

    def advance_targets(self):
        if len(self.targets) > 1:
            # still have targets to go
            self.targets = self.targets[1:]
            if self.use_yaw_targets:
                self.yaw_targets = self.yaw_targets[1:]
        else:
            self.targets = []
            self.yaw_targets = []

        # delete the reached target and recolour the others
        if self.enable_render and len(self.target_visual) > 0:
            p.removeBody(self.target_visual[0])
            self.target_visual = self.target_visual[1:]

            # recolour
            for i, visual in enumerate(self.target_visual):
                p.changeVisualShape(
                    visual,
                    linkIndex=-1,
                    rgbaColor=(0, 1 - (i / len(self.target_visual)), 0, 1),
                )

    def num_targets_reached(self):
        return self.num_targets - len(self.targets)

    def all_targets_reached(self):
        return len(self.targets) == 0
