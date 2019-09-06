
# Objective

Implement the NEAT algorithm as elegantly as possible, and use it on
OpenAI-gym environments.
Based on this paper: <http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf>.

# Algorithm Outline

## Testing the agents

1. Each agent is represented as a genome, which contains the nodes and connections
of that genome.

2. The simulation will generate an agent using each genome in its population list, let
it run in the environment (in turn) and return its accumulated reward.

## Generating a new generation1

1. Split the genomes into species based on their 'genetic distance' (formula 1 page 13).

2. Adjust each genomes fitness based its species (formula 2 page 13).

3. Generate a new generation using the rules specified in pages 11 - 13.
