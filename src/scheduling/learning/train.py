"""Train a PPO model for the scheduling environment."""

from typing import Any
import logging

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy

from gymnasium.wrappers import FlattenObservation
import gymnasium as gym


def train_for_settings(settings: list[dict[str, Any]]):
    for i, setting in enumerate(settings):
        logging.info("Training model for setting %d", i)
        env = gym.make("Scheduling-v0", **setting)
        env = FlattenObservation(env)
        check_env(env)
        logging.info("Environment checked successfully. Training model...")
        if i == 0:
            model = PPO(
                "MlpPolicy", env, verbose=1
            )  # Create a single PPO model instance
        else:
            model.set_env(env)
        model.learn(total_timesteps=1000000)
        logging.info("Setting completed.")
        env.close()
    model.save("ppo_scheduling")


def run_model(setting: dict[str, Any]) -> None:
    env = gym.make("Scheduling-v0", **setting)
    env = FlattenObservation(env)
    print("loading")
    model = PPO.load("ppo_scheduling", env)
    mean_reward, std_reward = evaluate_policy(
        model, model.get_env(), n_eval_episodes=10
    )
    print(f"Mean reward: {mean_reward} +/- {std_reward}")
    obs, _ = env.reset()
    actions = []
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        actions.append(action)
        obs, _, done, *_ = env.step(action)
        if done:
            break
    print(actions)
