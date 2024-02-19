""""""

from typing import Any

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

import gymnasium as gym


def train_for_settings(settings: list[dict[str, Any]]):
    for i, setting in enumerate(settings):
        env = gym.make("Scheduling-v0", **setting)
        check_env(env)
        if i == 0:
            model = PPO(
                "MlpPolicy", env, verbose=1
            )  # Create a single PPO model instance
        else:
            model.set_env(env)
        model.learn(total_timesteps=10000)
        env.close()
    model.save("ppo_scheduling")
