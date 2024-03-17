"""Train a PPO model for the scheduling environment."""

from typing import Any
import logging

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy

from gymnasium.wrappers import FlattenObservation
import gymnasium as gym

from src.scheduling.common import Schedule


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


def run_model(setting: dict[str, Any]) -> Schedule:
    env = gym.make("Scheduling-v0", **setting)
    env = FlattenObservation(env)
    logging.debug("loading rl model")
    model = PPO.load("ppo_scheduling", env)
    mean_reward, std_reward = evaluate_policy(
        model, model.get_env(), n_eval_episodes=10
    )
    logging.debug("Mean reward: %d +/- %d", mean_reward, std_reward)
    obs, _ = env.reset()
    actions = []
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        actions.append(action)
        obs, _, done, _, info = env.step(action)
        if done:
            break
    assert "schedule" in info
    return info["schedule"]
