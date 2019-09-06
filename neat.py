from typing import List, Iterator, Dict, Tuple
from itertools import product
import numpy as np
from data_types import Innovation, Node, Genome, NodeType
from neural_network import NeuralNetwork

# TODO: implement mutation and crossover


class NEAT:
    """Manages genomes using the NEAT algorithm"""

    def __init__(self, population_size: int, inputs: int, outputs: int,
                 speciation_consts: Dict[str, float]):
        self.population_size = population_size
        self.inputs = inputs
        self.outputs = outputs

        # nodes and innovations will be managed here, and assigned to genomes at each new generation
        self.nodes = {idx: Node(idx, NodeType.INPUT if idx < inputs else NodeType.OUTPUT, 0.0)
                      for idx in range(inputs + outputs)}
        self.innovations = {idx: Innovation(idx, i, j+inputs, 0.0, True)
                            for idx, (i, j) in enumerate(product(range(inputs), range(outputs)))}

        # generate initial population
        self.population: List[Genome] = []
        for temp in range(population_size):
            nodes = [Node(idx, NodeType.INPUT if idx < inputs else NodeType.OUTPUT,
                          np.random.random_sample() * 2 - 1)
                     for idx in range(inputs + outputs)]
            innovations = [Innovation(idx, i, j+inputs, np.random.random_sample() * 2 - 1, True)
                           for idx, (i, j) in enumerate(product(range(inputs), range(outputs)))]
            self.population.append(Genome(tuple(nodes), tuple(innovations)))

        # initial population is just one species
        self.speciation_consts = speciation_consts
        self.species: Dict[Genome, List[Genome]] = self.split_population()

    def split_population(self) -> Dict[Genome, List[Genome]]:
        """Splits the population into species based on genetic distance from
        each species representative

        Returns:
            Dict[Genome, List[Genome]] -- species representative and its species
        """
        species = dict()

        for genome in self.population:

            # add genome to species that has a rep with similiar genes to it
            for species_rep in species:
                gen_distance = self.genetic_distance(genome, species_rep)
                if gen_distance < self.speciation_consts['t']:
                    species[species_rep].append(genome)
                    break

            # if no such species is found, create a new species with only the genome in it
            else:
                species[genome] = [genome]

        return species

    def genetic_distance(self, a: Genome, b: Genome) -> float:
        """Computes the genetic distance between two genomes based on speciation params

        Arguments:
            a {Genome} -- first genome
            b {Genome} -- second genome

        Returns:
            float -- genetic distance of the two genomes
        """

        # get constants
        c1 = self.speciation_consts['c1']
        c2 = self.speciation_consts['c2']
        c3 = self.speciation_consts['c3']

        # get max nodes
        max_nodes = float(max(len(a.nodes), len(b.nodes)))

        # split innovation into matching, disjoint and excess genes
        matching, disjoint, excess = self.get_diff(a, b)

        # get delta weights between matching innovations
        a_innovations_dict = {inn.idx: inn for inn in a.innovations}
        b_innovations_dict = {inn.idx: inn for inn in b.innovations}
        delta_weights = [a_innovations_dict[idx].weight - b_innovations_dict[idx].weight
                         for idx in matching]
        delta_weights = abs(np.average(delta_weights))
        # calculate genetic distance
        return (c1 * len(excess)) / max_nodes + \
            (c2 * len(disjoint)) / max_nodes + \
            c3 * delta_weights

    def get_diff(self, a: Genome, b: Genome) -> Tuple[List[Innovation],
                                                      List[Innovation], List[Innovation]]:
        """Returns the matching, disjoint and excess genes between two genomes

        Arguments:
            a {Genome} -- [description]
            b {Genome} -- [description]

        Returns:
            Tuple[List[Innovation], List[Innovation], List[Innovation]] -- matching, disjoint,
            excess genes in that order
        """
        matching, disjoint, excess = [], [], []
        a_innovations, b_innovations = \
            [innovation.idx for innovation in a.innovations], \
            [innovation.idx for innovation in b.innovations]
        for idx in range(max(len(a_innovations),
                             len(b_innovations))):
            if idx in a_innovations and idx in b_innovations:
                matching.append(idx)
            elif idx in a_innovations:
                if idx > max(b_innovations):
                    excess.append(idx)
                else:
                    disjoint.append(idx)
            else:
                if idx > max(a_innovations):
                    excess.append(idx)
                else:
                    disjoint.append(idx)
        return matching, disjoint, excess

    def new_generation(self, genome_scores: List[Tuple[Genome, float]]) -> None:

        # assign genome fitness based on fitness sharing within species
        genome_fitness = {genome: score for (genome, score) in genome_scores}
        for genome in genome_fitness:
            genome_fitness[genome] /= len(self.get_genome_species(genome))

        # generate new genome species reps from previous generation
        self.species = self.split_population()

        print('new generation---------------')
        print(len(self.species))

    def get_genome_species(self, genome: Genome) -> List[Genome]:
        """Returns all the genomes in the species of a given genom

        Arguments:
            genome {Genome} -- genome to search

        Returns:
            List[Genome] -- all genomes in the genome's species
        """
        for species in self.species.values():
            if genome in species:
                return species
        raise Exception('Genome without a species is not allowed')

    def iter_agents(self) -> Iterator[NeuralNetwork]:
        """Returns an iterator that goes through all genomes in the population
        and generates neural networks for them.

        Returns:
            Iterator[NeuralNetwork] -- neural networks built from genomes
        """
        for genome in self.population:
            yield NeuralNetwork(genome)


if __name__ == "__main__":
    TEST = NEAT(2, 5, 5, {'c1': 2.0, 'c2': 2.0, 'c3': 2.0, 't': 15.0})
