import gym
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

class TrainingWrapper(gym.Wrapper):
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
        if action == 0:
            reward -= 0.1
        return state, reward, done, info
