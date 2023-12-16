import gymnasium as gym
# You can choose to use gym library instead of gymnasium
# import gym 

import warnings
from collections import namedtuple
warnings.filterwarnings("ignore", category=DeprecationWarning)
partition = namedtuple("partition", ["type", "subtype"])

env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()