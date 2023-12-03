import gymnasium as gym
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
from tqdm import tqdm

# Rules from Sutton & Barto
env = gym.make("Blackjack-v1", sab=True)

done = False
# obs =  (15, 2, 0)
# player current sum, dealer value face up card, player usable ace or not
obs, info = env.reset()
print('obs = ', obs)
print('info = ', info)

action = env.action_space.sample()
# action =  0
print('action = ', action)

# obs =  (18, 10, 0)
# reward =  0.0
# terminated =  False
# truncated =  False
# info =  {}
obs, reward, terminated, truncated, info = env.step(action)
print('obs = ', obs)
print('reward = ', reward)
print('terminated = ', terminated)
print('truncated = ', truncated)
print('info = ', info)

# class BlackjackAgent:
#     def __init__(self, learning_rate: float, initial_epsilon: float, epsilon_decay: float, final_epsilon: float, discount_factor: float = 0.95):
#         self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

#         self.lr = learning_rate
#         self.discount_factor = discount_factor

#         self.epsilon = initial_epsilon
#         self.epsilon_decay = epsilon_decay
#         self.final_epsilon = final_epsilon

#         self.training_error = []
    
#     def get_action(self, obs: tuple[int, int, bool]) -> int:
#         # explore the env first
#         if np.random.random() < self.epsilon:
#             return env.action_space.sample()
#         # greedy exploit the env
#         else:
#             return int(np.argmax(self.q_values[obs]))
    
#     def update(self, obs: tuple[int, int, bool], action: int, reward: float, terminated: bool, next_obs: tuple[int, int, bool]):
#         future_q_value = (not terminated) * np.max(self.q_values[next_obs])
#         temporal_difference = (reward + self.discount_factor * future_q_value - self.q_values[obs][action])

#         self.q_values[obs][action] = (self.q_values[obs][action] + self.lr * temporal_difference)
#         self.training_error.append(temporal_difference)
    
#     def decay_epsilon(self):
#         self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)