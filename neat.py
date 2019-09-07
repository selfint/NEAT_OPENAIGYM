from typing import List, Iterator, Dict, Tuple
from itertools import product
import numpy as np
from data_types import Innovation, Node, Genome, NodeType
from neural_network import NeuralNetwork


class NEAT:
    """Manages genomes using the NEAT algorithm"""

    # pylint: disable=too-many-instance-attributes

    def __init__(self, population_size: int, inputs: int, outputs: int,
                 speciation_consts: Dict[str, float]):
        self.population_size = population_size
        self.inputs = inputs
        self.outputs = outputs

        # nodes and innovations metadata will be saved here
        # and the actual values stored inside each genome
        self.nodes: Dict[int, NodeType] = {idx: NodeType.INPUT if idx < inputs else NodeType.OUTPUT
                                           for idx in range(inputs + outputs)}
        self.innovations: Dict[int, Tuple[int, int]] = {idx: (i, j+inputs)
                                                        for idx, (i, j)
                                                        in enumerate(product(range(inputs),
                                                                             range(outputs)))}

        # generate initial population
        self.population: List[Genome] = []
        for _ in range(population_size):
            nodes = [Node(idx, node_role)
                     for idx, node_role in self.nodes.items()]
            innovations = [Innovation(idx, src, dst,
                                      np.random.random_sample(), True)
                           for idx, (src, dst) in self.innovations.items()]
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

    def get_diff(self, a: Genome, b: Genome) -> Tuple[List[int], List[int], List[int]]:
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

        # generate new creatures for each species based on that species fitness
        species_fitness = dict()
        for species in self.species:
            fitness = 0
            for genome in self.species[species]:
                fitness += genome_fitness[genome]
            species_fitness[species] = fitness

        # assign new children amount to each species
        total_fitness = sum(species_fitness.values())
        species_children = dict()
        for species in species_fitness:
            new_children = species_fitness[species] / total_fitness
            species_children[species] = round(
                new_children * self.population_size)

        # limit new children to population size
        ignore = []
        while sum(species_children.values()) != self.population_size:
            if sum(species_children.values()) < self.population_size:
                options = [species for species in species_children
                           if species not in ignore]
                chosen = np.random.choice(list(species_children.keys()))
                species_children[chosen] += 1
            else:
                options = [species for species in species_children
                           if species_children[species] > 0 and species not in ignore]
                chosen = np.random.choice(options)
                species_children[chosen] -= 1
            ignore.append(chosen)

        # generate new children for each species
        children = []
        for species in species_children:
            for _ in range(species_children[species]):
                child = self.get_new_child(species, genome_fitness)
                children.append(child)
        
        # re-assign population to children
        self.population = children

    def get_new_child(self, species: Genome, genome_fitness: Dict[Genome, float]) -> Genome:
        """Generates a new child for a given species

        Arguments:
            species {Genome} -- species rep of the given species

        Returns:
            Genome -- genome of the new child
        """

        # select two parents from that species
        # TODO add interspecies crossover
        parent_a: Genome = np.random.choice(self.species[species])
        parent_b: Genome = np.random.choice(
            [genome for genome in self.species[species] if genome is not parent_a
             or len(self.species[species]) == 1])

        # perform crossover
        matching, disjoint, excess = self.get_diff(parent_a, parent_b)
        child_innovations = []
        a_innovations = {inn.idx: inn for inn in parent_a.innovations}
        b_innovations = {inn.idx: inn for inn in parent_b.innovations}

        # crossover matching innovations
        for idx in matching:
            if np.random.random_sample() < 0.5:
                weight = a_innovations[idx].weight
                enabled = a_innovations[idx].enabled
            else:
                weight = b_innovations[idx].weight
                enabled = b_innovations[idx].enabled
            src, dst = self.innovations[idx]
            innovation = Innovation(idx, src, dst, weight, enabled)
            child_innovations.append(innovation)

        # crossover disjoint and excess genes
        for idx in disjoint + excess:
            print(idx)
            # TODO implement disjint and excess gene crossover

        # get necassary nodes from resulting innovations
        child_node_idxs = []
        for innovation in child_innovations:
            child_node_idxs.append(innovation.src)
            child_node_idxs.append(innovation.dst)
        child_nodes = []
        for idx in set(child_node_idxs):
            child_nodes.append(Node(idx, self.nodes[idx]))

        # generate child genome
        child = Genome(tuple(child_nodes), tuple(child_innovations))
        return child

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
