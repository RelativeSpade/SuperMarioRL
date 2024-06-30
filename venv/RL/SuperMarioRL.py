# Setup Game
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

# Setup game
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# Create a flag whether done
done = True

# Loop through each frame in the game
for step in range(10000):
    # Start the game to begin
    if done:
        state = env.reset()
    # Random Move
    state, reward, done, info = env.step(env.action_space.sample())
    # Update Screen
    env.render()

# Close when finished.
env.close()
