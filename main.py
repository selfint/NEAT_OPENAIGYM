import gym
from neat import NEAT

env = gym.make("CartPole-v1")
POPULATION_SIZE = 10
INPUTS = env.observation_space
agents = NEAT(POPULATION_SIZE, 10, 10)

"""
observation = env.reset()
for _ in range(1000):
  env.render()
  action = env.action_space.sample() # your agent here (this takes random actions)
  observation, reward, done, info = env.step(action)

  if done:
    observation = env.reset()
env.close()
"""