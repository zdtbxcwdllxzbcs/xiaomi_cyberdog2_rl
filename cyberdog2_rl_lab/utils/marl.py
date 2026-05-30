"""Compatibility helpers for using DirectMARLEnv with the local RSL-RL scripts."""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from isaaclab.envs import DirectMARLEnv, DirectRLEnv


def multi_agent_to_single_agent(env: DirectMARLEnv, state_as_observation: bool = False) -> DirectRLEnv:
    """Convert a multi-agent env into a single-agent env with ``_get_observations`` support.

    IsaacLab's stock converter is sufficient for normal stepping, but the local
    RSL-RL wrapper calls ``_get_observations()`` on the unwrapped env during
    initialization. This adapter mirrors the upstream behavior and additionally
    implements that method so DirectMARLEnv tasks can train through the existing
    scripts without further changes.
    """

    class Env(DirectRLEnv):
        def __init__(self, wrapped_env: DirectMARLEnv) -> None:
            self.env: DirectMARLEnv = wrapped_env.unwrapped
            self._state_as_observation = state_as_observation
            if self._state_as_observation:
                assert self.env.cfg.state_space != 0, (
                    "The environment state cannot be used as observation since it was explicitly defined as"
                    " unconstructed"
                )

            self.cfg = self.env.cfg
            self.sim = self.env.sim
            self.scene = self.env.scene
            self.render_mode = self.env.render_mode

            self.single_observation_space = gym.spaces.Dict()
            if self._state_as_observation:
                self.single_observation_space["policy"] = self.env.state_space
            else:
                self.single_observation_space["policy"] = gym.spaces.flatten_space(
                    gym.spaces.Tuple([self.env.observation_spaces[agent] for agent in self.env.possible_agents])
                )
            self.single_action_space = gym.spaces.flatten_space(
                gym.spaces.Tuple([self.env.action_spaces[agent] for agent in self.env.possible_agents])
            )
            self.observation_space = gym.vector.utils.batch_space(
                self.single_observation_space["policy"], self.num_envs
            )
            self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)

        @property
        def unwrapped(self) -> "Env":
            return self

        @property
        def num_envs(self) -> int:
            return self.env.num_envs

        @property
        def device(self):
            return self.env.device

        @property
        def max_episode_length(self):
            return self.env.max_episode_length

        @property
        def episode_length_buf(self) -> torch.Tensor:
            return self.env.episode_length_buf

        @episode_length_buf.setter
        def episode_length_buf(self, value: torch.Tensor):
            self.env.episode_length_buf = value

        def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
            obs, extras = self.env.reset(seed, options)
            return self._convert_obs(obs), extras

        def step(self, action: torch.Tensor):
            index = 0
            multi_actions = {}
            for agent in self.env.possible_agents:
                delta = gym.spaces.flatdim(self.env.action_spaces[agent])
                multi_actions[agent] = action[:, index : index + delta]
                index += delta
            obs, rewards, terminated, time_outs, extras = self.env.step(multi_actions)
            obs = self._convert_obs(obs)
            rewards = sum(rewards.values())
            terminated = math.prod(terminated.values()).to(dtype=torch.bool)
            time_outs = math.prod(time_outs.values()).to(dtype=torch.bool)
            return obs, rewards, terminated, time_outs, extras

        def _get_observations(self):
            return self._convert_obs(self.env._get_observations())

        def _convert_obs(self, obs: dict[str, torch.Tensor]):
            if self._state_as_observation:
                return {"policy": self.env.state()}
            return {
                "policy": torch.cat(
                    [obs[agent].reshape(self.num_envs, -1) for agent in self.env.possible_agents], dim=-1
                )
            }

        def render(self, recompute: bool = False) -> np.ndarray | None:
            return self.env.render(recompute)

        def close(self) -> None:
            self.env.close()

    return Env(env)
