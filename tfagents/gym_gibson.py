# coding=utf-8
# Copyright 2018 The TF-Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import gin

from tf_agents.environments import gym_wrapper
from tf_agents.environments import wrappers
from envs.dynamic_env import DynamicEnv
import igibson


@gin.configurable
def load(config_file,
         model_id=None,
         env_mode='headless',
         action_timestep=1.0 / 10.0,
         physics_timestep=1.0 / 40.0,
         device_idx=0,
         gym_env_wrappers=(),
         env_wrappers=(),
         spec_dtype_map=None):
    env = DynamicEnv(config_file=config_file,
                     scene_id=model_id,
                     mode=env_mode,
                     action_timestep=action_timestep,
                     physics_timestep=physics_timestep,
                     device_idx=device_idx)

    discount = env.config.get('discount_factor', 0.99)
    max_episode_steps = env.config.get('max_step', 500)

    return wrap_env(
        env,
        discount=discount,
        max_episode_steps=max_episode_steps,
        gym_env_wrappers=gym_env_wrappers,
        time_limit_wrapper=wrappers.TimeLimit,
        env_wrappers=env_wrappers,
        spec_dtype_map=spec_dtype_map,
        auto_reset=True
    )


@gin.configurable
def wrap_env(env,
             discount=1.0,
             max_episode_steps=0,
             gym_env_wrappers=(),
             time_limit_wrapper=wrappers.TimeLimit,
             env_wrappers=(),
             spec_dtype_map=None,
             auto_reset=True):
    for wrapper in gym_env_wrappers:
        env = wrapper(env)
    env = gym_wrapper.GymWrapper(
        env,
        discount=discount,
        spec_dtype_map=spec_dtype_map,
        match_obs_space_dtype=True,
        auto_reset=auto_reset,
        simplify_box_bounds=True
    )

    if max_episode_steps > 0:
        env = time_limit_wrapper(env, max_episode_steps)

    for wrapper in env_wrappers:
        env = wrapper(env)

    return env
