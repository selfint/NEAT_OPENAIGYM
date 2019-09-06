import gym
import numpy as np
import matplotlib.pyplot as plt
from neat import NEAT


ENV = gym.make("CartPole-v1")
POPULATION_SIZE = 20
INPUTS = ENV.observation_space.shape[0]
OUTPUTS = ENV.action_space.n
SPECIATION_CONSTS = {'c1': 1, 'c2': 1, 'c3': 1, 't': 0.001}
NEAT_MANAGER = NEAT(POPULATION_SIZE, INPUTS, OUTPUTS, SPECIATION_CONSTS)
ENV_STEPS = 1000
GENERATIONS = 10

AVG_SCORES = []
for gen in range(GENERATIONS):
    agent_scores = []
    score_hist = []
    for agent in NEAT_MANAGER.iter_agents():
        observation = ENV.reset()
        score = 0
        for _ in range(ENV_STEPS):
            # ENV.render()
            action = agent.predict(np.array(observation)).argmax()
            observation, reward, done, info = ENV.step(action)
            score += reward
            if done:
                observation = ENV.reset()
                break
        agent_scores.append((agent.genome, score))
        score_hist.append(score)

    # get a new generation
    AVG_SCORES.append(np.average(score_hist))
    NEAT_MANAGER.new_generation(agent_scores)
    
ENV.close()

plt.plot(range(1, GENERATIONS+1), AVG_SCORES)
plt.show()
