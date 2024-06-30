# Setup Game
import gym_super_mario_bros
import stable_baselines3
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

# Preprocess Environment && Import Frame Stacker Wrapper and GrayScaling Wrapper
from gym.wrappers import GrayScaleObservation
# Import Vectorization Wrappers
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
# Import matplotlib
from matplotlib import pyplot as plt

# 1. Create Base Environment
env = gym_super_mario_bros.make('SuperMarioBros-v0')
# 2. Simplify Controls
env = JoypadSpace(env, SIMPLE_MOVEMENT)
# 3. Grayscale
env = GrayScaleObservation(env, keep_dim=True)
# 4. Wrap inside the Dummy Environment
env = DummyVecEnv([lambda: env])
# 5. Stack the Frames.
env = VecFrameStack(env, 4, channels_order='last')

# RL Algorithm
# Import os for file path management, PPO for algos, Base Callback for saving
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback