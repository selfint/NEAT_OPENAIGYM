import gym
import numpy as np
from neat import NEAT
import matplotlib.pyplot as plt


env = gym.make("CartPole-v1")
POPULATION_SIZE = 10
INPUTS = env.observation_space.shape[0]
OUTPUTS = env.action_space.n
neat_manager = NEAT(POPULATION_SIZE, INPUTS, OUTPUTS)
ENV_STEPS = 1000
GENERATIONS = 10

avg_scores = []
for gen in range(GENERATIONS):
    agent_scores = []
    for agent in neat_manager.iter_agents():
        observation = env.reset()
        score = 0
        for _ in range(ENV_STEPS):
            env.render()
            action = agent.predict(np.array(observation)).argmax() # your agent here (this takes random actions)
            observation, reward, done, info = env.step(action)
            score += reward
            if done:
                observation = env.reset()
                break
        agent_scores.append(score)
    
    # get a new generation
    avg_scores.append(np.average(agent_scores))
    neat_manager.new_generation(agent_scores)
env.close()

plt.plot(range(1, GENERATIONS+1), avg_scores)
plt.show()
