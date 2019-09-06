from typing import List, Iterator
from itertools import product
import numpy as np
from data_types import Innovation, Node, Genome, NodeType
from neural_network import NeuralNetwork


class NEAT:
    """Manages genomes using the NEAT algorithm"""

    def __init__(self, population_size: int, inputs: int, outputs: int, connected: bool = True):
        self.population_size: int = population_size
        self.inputs = inputs
        self.outputs = outputs

        # nodes and innovations will be managed here, and assigned to genomes at each new generation
        self.nodes = {i: Node(i, NodeType.INPUT if i < inputs else NodeType.OUTPUT, 0.0)
                      for i in range(inputs + outputs)}
        self.innovations = {idx: Innovation(idx, i,
                                            j+inputs, np.random.random_sample() * 2 - 1, True)
                            for idx, (i, j) in enumerate(product(range(inputs), range(outputs)))}

        # generate initial population
        self.population: List[Genome] = [Genome(self.nodes.values(), self.innovations.values()
                                                if connected else [])
                                         for _ in range(population_size)]

    def new_generation(self, genome_scores: List[float]) -> None:
        """updates the genome population using speciation, crossover and mutations
        based on the scores of each genome

        Arguments:
            genome_scores {List[float]} -- scores from environments

        """

    def iter_agents(self) -> Iterator[NeuralNetwork]:
        """Returns an iterator that goes through all genomes in the population
        and generates neural networks for them.

        Returns:
            Iterator[NeuralNetwork] -- neural networks built from genomes
        """
        for genome in self.population:
            yield NeuralNetwork(genome)


if __name__ == "__main__":
    TEST = NEAT(2, 2, 1)
    for agent in TEST.iter_agents():
        print(agent)
