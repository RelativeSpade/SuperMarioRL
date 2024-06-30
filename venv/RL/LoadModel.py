# Setup Game
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from stable_baselines3 import PPO

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

# Load Model
MODEL_DIR = './train/best_model_100000.zip'  # You have to have run the model before this.
model = PPO.load(MODEL_DIR)

# Start the game
state = env.reset()
# Loop through frames
while True:
    action, _ = model.predict(state)
    state, reward, done, info = env.step(action)
    env.render()
