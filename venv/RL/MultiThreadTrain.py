# Setup Game
import gym_super_mario_bros
# Preprocess Environment && Import Frame Stacker Wrapper and GrayScaling Wrapper
from gym.wrappers import GrayScaleObservation
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from stable_baselines3 import PPO
# Import Vectorization Wrappers
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv

from TrainAndLoggingCallback import TrainAndLoggingCallback
from TrainingWrapper import TrainingWrapper


def create_env():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = GrayScaleObservation(env, keep_dim=True)
    env = TrainingWrapper(env)
    return env


if __name__ == '__main__':
    num_envs = 8  # Number of parallel environments
    envs = [lambda: create_env() for _ in range(num_envs)]
    envs = SubprocVecEnv(envs)
    envs = VecFrameStack(envs, 4, channels_order='last')

    # RL Algorithm

    CHECKPOINT_DIR = './train/'
    LOG_DIR = './logs/'

    callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)

    model = PPO('CnnPolicy', envs, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.000001, n_steps=512)

    model.learn(total_timesteps=1000000, callback=callback)
