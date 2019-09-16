from genome_renderer import GenomeRenderer
import gym
import numpy as np
import matplotlib.pyplot as plt
from neat import NEAT


ENV = gym.make("MountainCar-v0")
POPULATION_SIZE = 150
INPUTS = ENV.observation_space.shape[0]
OUTPUTS = ENV.action_space.n
SPECIATION_CONSTS = {'c1': 1.0, 'c2': 1.0, 'c3': 0.4, 't': 3.0}
MUTATION_CONSTS = {'weight': 0.8, 'weight_pert': 0.9,
                   'connection': 0.05, 'node': 0.03,
                   'enable_disabled_connection': 0.25, 'no_crossover': 0.25,
                   'interspecies_mating': 0.001}
GENERATION_CONSTS = {'large_species': 5}
ENV_STEPS = 500
GENERATIONS = 20
RUNS = 1
RUN_HISTORY = []
for run in range(RUNS):
    AVG_SCORES = []
    NEAT_MANAGER = NEAT(POPULATION_SIZE, INPUTS, OUTPUTS, SPECIATION_CONSTS)
    best_score = -1
    for gen in range(GENERATIONS):
        agent_scores = []
        score_hist = []
        for agent in NEAT_MANAGER.iter_agents():
            observation = ENV.reset()
            score = 0
            for _ in range(ENV_STEPS):
                if score > best_score or gen == GENERATIONS - 1:
                    ENV.render()
                    best_score = score
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
        NEAT_MANAGER.new_generation(agent_scores, MUTATION_CONSTS, GENERATION_CONSTS)
        print(f"---{run}:{gen}---\nMax Score: {max(score_hist)}\nAvg Score: {np.mean(score_hist)}")
        NEAT_MANAGER.print_status()
        for i, (genome, score) in enumerate(agent_scores):
            if score == max(score_hist):
                render = GenomeRenderer(
                    genome, f"genome_{run}_{gen}_{i}_render.png", 'renders')
                break
    RUN_HISTORY.append(AVG_SCORES)
ENV.close()


for run in RUN_HISTORY:
    plt.plot(range(1, GENERATIONS+1), run)
plt.show()
