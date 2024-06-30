import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from gym.wrappers import GrayScaleObservation


# Create a custom reward wrapper
class TrainingWrapper(gym.wrappers):
    def __init__(self, env):
        super(TrainingWrapper, self).__init__(env)
        self.prev_x_pos = 0

    def reset(self):
        self.prev_x_pos = 0
        return self.env.reset()

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        x_pos = info['x_pos']
        # Reward for moving to the right
        reward += (x_pos - self.prev_x_pos) * 0.1
        self.prev_x_pos = x_pos
        # Penalty for standing still
        if action == SIMPLE_MOVEMENT.index('NOOP'):
            reward -= 0.1
        return state, reward, done, info

# Setup the environment
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = GrayScaleObservation(env, keep_dim=True)
env = TrainingWrapper(env)  # Add custom reward wrapper
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, 4, channels_order='last')

# RL Algorithm
model = PPO('CnnPolicy', env, verbose=1, tensorboard_log='./logs/', learning_rate=0.000001, n_steps=512)

# Train the Model
model.learn(total_timesteps=1000000)

# Load Model
model = PPO.load('./train/best_model_100000.zip')

# Start the game
state = env.reset()
while True:
    action, _ = model.predict(state)
    state, reward, done, info = env.step(action)
    env.render()
