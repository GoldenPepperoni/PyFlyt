from __future__ import annotations

import numpy as np
from gymnasium import spaces

from ..dubins_path_handler import DubinsPathHandler
from .fixedwing_base_env import FixedwingBaseEnv
import pybullet as p


class FixedwingCCDubinsPathEnv(FixedwingBaseEnv):
    """
    Generates a Dubins path for Carrot Chasing Path Following Algorithm

    Actions are vp, vq, vr, T, ie: angular rates and thrust

    The target is a set of `[x, y, z, yaw]` targets in space

    Reward:
        200.0 for reaching target,
        -100 for collisions or out of bounds,
        -0.1 otherwise
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        sparse_reward: bool = False,
        num_targets: int = 2,
        goal_reach_distance: float = 2.0,
        flight_dome_size: float = 200.0,
        turning_radius : float = 40,
        path_step_size : float = 0.5, 
        CC_lookahead : float = 11.5,
        max_duration_seconds: float = 120.0,
        angle_representation: str = "quaternion",
        custom_targets: np.ndarray = None,
        custom_yaw_targets: np.ndarray = None,
        agent_hz: int = 30,
        render_mode: None | str = None,
        spawn_pos: np.ndarray = np.array([[0.0, 0.0, 10.0]]),
        spawn_orn: np.ndarray = np.array([[0.0, 0.0, 0.0]]),
        ):
        """__init__.

        Args:
            num_targets (int): num_targets
            goal_reach_distance (float): goal_reach_distance
            flight_dome_size (float): size of the allowable flying area
            max_duration_seconds (float): maximum simulatiaon time of the environment
            angle_representation (str): can be "euler" or "quaternion"
            agent_hz (int): looprate of the agent to environment interaction
            render_mode (None | str): can be "human" or None
        """

        super().__init__(
            start_pos=spawn_pos,
            start_orn=spawn_orn,
            drone_type="fixedwing",
            drone_model="fixedwing",
            flight_dome_size=flight_dome_size,
            max_duration_seconds=max_duration_seconds,
            angle_representation=angle_representation,
            agent_hz=agent_hz,
            render_mode=render_mode,
        )

        # define Dubins path
        self.dubinspath = DubinsPathHandler(
            enable_render=self.enable_render,
            num_targets=num_targets,
            use_yaw_targets=True,
            goal_reach_distance=goal_reach_distance,
            goal_reach_angle=np.inf,
            flight_dome_size=flight_dome_size,
            turning_radius=turning_radius,
            path_step_size=path_step_size,
            custom_targets=custom_targets,
            custom_yaw_targets=custom_yaw_targets,
            start_pos=np.array([[0.0, 0.0, 10.0]]), # Hardcoded to start from origin
            np_random=self.np_random,
        )

        # Define observation space
        self.observation_space = spaces.Dict(
            {
                "attitude": self.attitude_space,
                "target_deltas": spaces.Sequence(
                    space=spaces.Box(
                        low=-2 * flight_dome_size,
                        high=2 * flight_dome_size,
                        shape=(3,),
                        dtype=np.float64,
                    )
                ),
                "carrot_pos": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64
                ),
            }
        )

        """ ENVIRONMENT CONSTANTS """
        self.sparse_reward = sparse_reward
        self.spawn_pos = spawn_pos
        self.spawn_orn = spawn_orn


        """Carrot Chasing parameters"""
        self.CC_lookahead = CC_lookahead

    def reset(self, seed=None, options=None, aviary_options=dict()):
        """reset.

        Args:
            seed: seed to pass to the base environment.
            options:
        """
        super().begin_reset(seed, options, aviary_options)
        self.path = self.dubinspath.reset()
        self.info["path"] = self.path
        self.info["num_targets_reached"] = 0
        self.distance_to_immediate = np.inf
        super().end_reset()

        return self.state, self.info

    def compute_state(self):
        """state.

        This returns the observation as well as the distances to target.
        - "attitude" (Box)
            - ang_vel (vector of 3 values)
            - ang_pos (vector of 3/4 values)
            - lin_vel (vector of 3 values)
            - lin_pos (vector of 3 values)
            - previous_action (vector of 4 values)
        - "target_deltas" (Sequence)
            - list of body_frame distances to target (vector of 3/4 values)
        - "carrot_pos" (Box)
            - Cartesian coordinates of carrot linear position (vector of 3 values)
        """
        ang_vel, ang_pos, lin_vel, lin_pos, quarternion = super().compute_attitude()

        # combine everything
        new_state = dict()
        if self.angle_representation == 0:
            new_state["attitude"] = np.array(
                [*ang_vel, *ang_pos, *lin_vel, *lin_pos, *self.action]
            )
        elif self.angle_representation == 1:
            new_state["attitude"] = np.array(
                [*ang_vel, *quarternion, *lin_vel, *lin_pos, *self.action]
            )

        new_state["target_deltas"] = self.dubinspath.distance_to_target(
            ang_pos, lin_pos, quarternion
        )
        self.distance_to_immediate = float(
            np.linalg.norm(new_state["target_deltas"][0])
        )

        new_state["carrot_pos"] = self.dubinspath.get_CC_carrot(lin_pos, self.CC_lookahead)

        self.state = new_state

    def compute_term_trunc_reward(self):
        """compute_term_trunc."""
        super().compute_base_term_trunc_reward()

        # bonus reward if we are not sparse
        if not self.sparse_reward:
            self.reward += max(3.0 * self.dubinspath.progress_to_target(), 0.0)
            self.reward += 1.0 / self.distance_to_immediate

        # target reached
        # if self.dubinspath.target_reached():
        #     # ONLY FOR RL USE, UNCOMMENT THIS BLOCK FOR RL USE
        #     self.reward = 100.0

        #     # advance the targets
        #     self.dubinspath.advance_targets()

        #     # update infos and dones
        #     self.truncation |= self.dubinspath.all_targets_reached()
        #     self.info["env_complete"] = self.dubinspath.all_targets_reached()
        #     self.info["num_targets_reached"] = self.dubinspath.num_targets_reached()

        if self.dubinspath.path_end_reached():
            # NOT FOR RL USE, COMMENT THIS BLOCK FOR RL USAGE
            # update infos and dones
            self.truncation |= True
            self.info["env_complete"] = True
            self.info["num_targets_reached"] = self.dubinspath.num_targets_reached()