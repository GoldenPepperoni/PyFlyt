from __future__ import annotations

import numpy as np
from gymnasium import spaces

from ..waypoint_handler import WaypointHandler
from .quadx_base_env import QuadXBaseEnv


class QuadXWaypointsEnv(QuadXBaseEnv):
    """
    Simple Waypoint Environment

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
        num_targets: int = 4,
        use_yaw_targets: bool = False,
        goal_reach_distance: float = 0.2,
        goal_reach_angle: float = 0.1,
        flight_dome_size: float = 5.0,
        max_duration_seconds: float = 10.0,
        angle_representation: str = "quaternion",
        agent_hz: int = 30,
        render_mode: None | str = None,
    ):
        """__init__.

        Args:
            num_targets (int): num_targets
            use_yaw_targets (bool): use_yaw_targets
            goal_reach_distance (float): goal_reach_distance
            goal_reach_angle (float): goal_reach_angle
            flight_dome_size (float): size of the allowable flying area
            max_duration_seconds (float): maximum simulatiaon time of the environment
            angle_representation (str): can be "euler" or "quaternion"
            agent_hz (int): looprate of the agent to environment interaction
            render_mode (None | str): can be "human" or None
        """

        super().__init__(
            start_pos=np.array([[0.0, 0.0, 1.0]]),
            drone_type="quadx",
            drone_model="cf2x",
            flight_dome_size=flight_dome_size,
            max_duration_seconds=max_duration_seconds,
            angle_representation=angle_representation,
            agent_hz=agent_hz,
            render_mode=render_mode,
        )

        # define waypoints
        self.waypoints = WaypointHandler(
            enable_render=self.enable_render,
            num_targets=num_targets,
            use_yaw_targets=use_yaw_targets,
            goal_reach_distance=goal_reach_distance,
            goal_reach_angle=goal_reach_angle,
            flight_dome_size=flight_dome_size,
            np_random=self.np_random,
        )

        # Define observation space
        self.observation_space = spaces.Dict(
            {
                "attitude": self.combined_space,
                "target_deltas": spaces.Sequence(
                    space=spaces.Box(
                        low=-2 * flight_dome_size,
                        high=2 * flight_dome_size,
                        shape=(4,) if use_yaw_targets else (3,),
                        dtype=np.float64,
                    )
                ),
            }
        )

        """ ENVIRONMENT CONSTANTS """
        self.sparse_reward = sparse_reward

    def reset(self, seed=None, options=None):
        """reset.

        Args:
            seed: seed to pass to the base environment.
            options:
        """
        super().begin_reset(seed, options)
        self.waypoints.reset()
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
            - auxiliary information (vector of 4 values)
        - "target_deltas" (Sequence)
            - list of body_frame distances to target (vector of 3/4 values)
        """
        ang_vel, ang_pos, lin_vel, lin_pos, quarternion = super().compute_attitude()
        aux_state = super().compute_auxiliary()

        # combine everything
        new_state = dict()
        if self.angle_representation == 0:
            new_state["attitude"] = np.array(
                [*ang_vel, *ang_pos, *lin_vel, *lin_pos, *self.action, *aux_state]
            )
        elif self.angle_representation == 1:
            new_state["attitude"] = np.array(
                [*ang_vel, *quarternion, *lin_vel, *lin_pos, *self.action, *aux_state]
            )

        new_state["target_deltas"] = self.waypoints.distance_to_target(
            ang_pos, lin_pos, quarternion
        )
        self.distance_to_immediate = float(
            np.linalg.norm(new_state["target_deltas"][0])
        )

        self.state = new_state

    def compute_term_trunc_reward(self):
        """compute_term_trunc."""
        super().compute_base_term_trunc_reward()

        # bonus reward if we are not sparse
        if not self.sparse_reward:
            self.reward += max(2.0 * self.waypoints.progress_to_target(), 0.0)
            self.reward += 1.0 / self.distance_to_immediate

        # target reached
        if self.waypoints.target_reached():
            self.reward = 100.0

            # advance the targets
            self.waypoints.advance_targets()

            # update infos and dones
            self.truncation |= self.waypoints.all_targets_reached()
            self.info["env_complete"] = self.waypoints.all_targets_reached()
            self.info["num_targets_reached"] = self.waypoints.num_targets_reached()
