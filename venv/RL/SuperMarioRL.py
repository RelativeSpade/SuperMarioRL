# Setup Game
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

#env = gym_super_mario_bros.make('SuperMarioBros-v0')
#env = JoypadSpace(env, SIMPLE_MOVEMENT)

# Create a flag whether done
#done = True

# Loop through each frame in the game
#for step in range(10000):
    # Start the game to begin
    #if done:
        #env.reset()
    # Random Move
    #state, reward, done, info = env.step(env.action_space.sample())
    # Update Screen
    #env.render()

# Close when finished.
#env.close()

# Preprocess Enviorment

# Import Frame Stacker Wrapper and GrayScaleing Wrapper
from gym.wrappers import GrayScaleObservation
# Import Vectorization Wrappers
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
# Import matplotlib
from matplotlib import pyplot as plt

# 1. Create Base Enviorment
env = gym_super_mario_bros.make('SuperMarioBros-v0')
# 2. Simplify Controls
env = JoypadSpace(env, SIMPLE_MOVEMENT)
# 3. Grayscale
env = GrayScaleObservation(env, keep_dim=True)
# 4. Wrap inside the Dummy Enviorment

# 5. Stack the Frames.
