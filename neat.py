import numpy as np
from data_types import Innovation, Node, Genome, NodeType
from typing import List
from itertools import product
from neural_network import NeuralNetwork


class NEAT:
    """manages genomes that will be run by the main simulation"""

    def __init__(self, population_size: int, inputs: int, outputs: int, connected: bool = True):
        self.population_size: int = population_size
        self.inputs = inputs
        self.outputs = outputs
        self.population: List[Genome] = []

        # generate initial nodes for genomes here since they are the same for all genomes
        _initial_nodes: List[Node] = [Node(i, NodeType.INPUT if i < inputs else NodeType.OUTPUT, 0.0)
                                      for i in range(inputs + outputs)]

        # generate initial population genomes
        def initial_innovations(inputs: int, outputs: int) -> List[Innovation]:
            """generate innovations with random weights between -1.0 and 1.0 for each genome

            Arguments:
                inputs {int} -- input node count
                outputs {int} -- output node count

            Returns:
                List[Innovation] -- innovations connecting all input nodes to all output nodes
                random weights
            """
            return [Innovation(idx, i, j+inputs, np.random.random_sample() * 2 - 1, True)
                    for idx, (i, j) in enumerate(product(range(inputs), range(outputs)))]

        self.population: List[Genome] = [Genome(_initial_nodes, initial_innovations(inputs, outputs)
                                                if connected else [])
                                         for _ in range(population_size)]
        self.innovation_counter: int = 0
        self.node_counter: int = 0

    def new_generation(self, genome_scores: List[float]) -> None:
        """updates the genome population using speciation, crossover and mutations
        based on the scores of each genome

        Arguments:
            genome_scores {List[float]} -- scores from environments

        """

    def generate_agent(self, genome: Genome) -> NeuralNetwork:
        """read the a genome and generates a neural network for it

        Arguments:
            genome {Genome} -- contains the nodes and connection of the genome

        Returns:
            NeuralNetwork -- the network built from that genome
        """
        return NeuralNetwork(genome)


if __name__ == "__main__":
    test = NEAT(2, 2, 1)
    test_agent = test.population[0]
    n_test = NeuralNetwork(test_agent.nodes, test_agent.innovations)
    print(n_test.predict(np.array([1.0, 1.0])))
