import gymnasium as gym
from stable_baselines3 import DQN

env=gym.make("CartPole-v1")

model=DQN("MlpPolicy",env,exploration_initial_eps=0.8,verbose=1)

model.learn(total_timesteps=100000)