# Setup Game
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

# Preprocess Environment && Import Frame Stacker Wrapper and GrayScaling Wrapper
from gym.wrappers import GrayScaleObservation
# Import Vectorization Wrappers
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

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

class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.save_path = save_path
        self.check_freq = check_freq

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True


CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'

# Model Saving Callback
callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)

# Start the Model
model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.000001, n_steps=512)

# Train the Model.
model.learn(total_timesteps=1000000, callback=callback)

# Load Model
MODEL_DIR = './train/best_model_100000.zip' # You have to have run the model before this.
model = PPO.load(MODEL_DIR)

# Start the game
state = env.reset()
# Loop through frames
while True:

    action, _ = model.predict(state)
    state, reward, done, info = env.step(action)
    env.render()