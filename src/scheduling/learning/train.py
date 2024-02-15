""""""

from typing import Any

from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.env_checker import check_env

import gymnasium as gym


def train_for_settings(settings: list[dict[str, Any]]):
    for i, setting in enumerate(settings):
        env = gym.make("Scheduling-v0", **setting)
        check_env(env)
        if i == 0:
            model = PPO2(
                MlpPolicy, env, verbose=1
            )  # Create a single PPO model instance
        else:
            model.set_env(env)
        model.learn(total_timesteps=10000)
        env.close()
    model.save("ppo_scheduling")
