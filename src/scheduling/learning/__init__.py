from gymnasium.envs.registration import register


from src.scheduling.learning.environment import SchedulingEnv


register(
    id="Scheduling-v0",
    entry_point="src.scheduling.learning.environment:SchedulingEnv",
    max_episode_steps=1000,
)


from .train import train_for_settings
