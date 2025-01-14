import os

from gymnasium.envs.registration import register

# QuadX Envs
register(
    id="PyFlyt/QuadX-Hover-v0",
    entry_point="PyFlyt.gym_envs.quadx_envs.quadx_hover_env:QuadXHoverEnv",
)
register(
    id="PyFlyt/QuadX-Waypoints-v0",
    entry_point="PyFlyt.gym_envs.quadx_envs.quadx_waypoints_env:QuadXWaypointsEnv",
)
register(
    id="PyFlyt/QuadX-Gates-v0",
    entry_point="PyFlyt.gym_envs.quadx_envs.quadx_gates_env:QuadXGatesEnv",
)

# Fixedwing Envs
register(
    id="PyFlyt/Fixedwing-Waypoints-v0",
    entry_point="PyFlyt.gym_envs.fixedwing_envs.fixedwing_waypoints_env:FixedwingWaypointsEnv",
)

register(
    id="PyFlyt/Fixedwing-CCDubinsPath-v0",
    entry_point="PyFlyt.gym_envs.fixedwing_envs.fixedwing_CC_env:FixedwingCCDubinsPathEnv",
)

register(
    id="PyFlyt/Fixedwing-NLGLDubinsPath-v0",
    entry_point="PyFlyt.gym_envs.fixedwing_envs.fixedwing_NLGL_env:FixedwingNLGLDubinsPathEnv",
)

# Rocket Envs
register(
    id="PyFlyt/Rocket-Landing-v0",
    entry_point="PyFlyt.gym_envs.rocket_envs.rocket_landing_env:RocketLandingEnv",
)

