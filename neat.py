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
        # self.innovations: Dict[int, Tuple[int, int]] = dict()

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

        # keep track of all innovation splitting
        self.innovation_splits = dict()

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
        if max_nodes < 20:
            max_nodes = 1.0

        # split innovation into matching, disjoint and excess genes
        matching, disjoint, excess = self.get_diff(a, b)

        # get delta weights between matching innovations
        a_innovations_dict = {inn.idx: inn for inn in a.innovations}
        b_innovations_dict = {inn.idx: inn for inn in b.innovations}
        delta_weights = [a_innovations_dict[idx].weight - b_innovations_dict[idx].weight
                         for idx in matching]
        if delta_weights:
            delta_weights = abs(np.average(delta_weights))
        else:
            delta_weights = 0

        # calculate genetic distance
        genetic_distance = (c1 * len(excess)) / max_nodes + (c2 * len(disjoint)) / max_nodes + \
            c3 * delta_weights
        return genetic_distance

    @staticmethod
    def get_diff(a: Genome, b: Genome) -> Tuple[List[int], List[int], List[int]]:
        """Returns the matching, disjoint and excess genes between two genomes

        Arguments:
            a {Genome} -- [description]
            b {Genome} -- [description]

        Returns:
            Tuple[List[Innovation], List[Innovation], List[Innovation]] -- matching, disjoint,
            excess genes in that order
        """
        matching, disjoint, excess = [], [], []
        a_innovations = {inn.idx: inn for inn in a.innovations}
        b_innovations = {inn.idx: inn for inn in b.innovations}
        all_innovations = set(list(a_innovations.keys()) +
                              list(b_innovations.keys()))

        for idx in all_innovations:
            if idx in a_innovations and idx in b_innovations:
                matching.append(idx)
            else:
                if idx in a_innovations:
                    if b_innovations and idx > max(b_innovations):
                        excess.append(idx)
                    else:
                        disjoint.append(idx)
                else:  # idx in b innovations and not in a innovations
                    if a_innovations and idx > max(a_innovations):
                        excess.append(idx)
                    else:
                        disjoint.append(idx)

        return matching, disjoint, excess

    def calc_fitness(self, genome_scores: List[Tuple[Genome, float]]) -> Dict[Genome, float]:
        """Calculates the fitness of each genome using fitness-sharing in each species

        Arguments:
            genome_scores {List[Tuple[Genome, float]]} -- the score each genome 
            recieved from the environment

        Returns:
            Dict[genome, float] -- each genome and its adjusted score
        """
        genome_fitness = {genome: score for (genome, score) in genome_scores}
        for genome in genome_fitness:
            genome_fitness[genome] /= len(self.get_genome_species(genome))
        return genome_fitness

    def calc_species_fitness(self, genome_fitness: Dict[Genome, float]) -> Dict[Genome, float]:
        """Calculates the total fitness of each species

        Arguments:
            genome_fitness {Dict[genome, float]} -- each genome and its fitness

        Returns:
            Dict[genome, float] -- each species rep and its species fitness
        """
        species_fitness = dict()
        for species in self.species:
            fitness = 0
            for genome in self.species[species]:
                fitness += genome_fitness[genome]
            species_fitness[species] = fitness
        return species_fitness

    def calc_child_amounts(self, species_fitness: Dict[Genome, float]) -> Dict[Genome, int]:
        # assign new children amount to each species
        total_fitness = sum(species_fitness.values())
        species_children = dict()
        for species in species_fitness:
            new_children = species_fitness[species] / total_fitness
            species_children[species] = int(round(
                new_children * self.population_size))

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

        return species_children

    def new_generation(self, genome_scores: List[Tuple[Genome, float]],
                       mutation_consts: Dict[str, float]) -> None:
        """Generates a new generation 

        Arguments:
            genome_scores {List[Tuple[Genome, float]]} -- the score each genome 
            recieved from the environment

        Returns:
            None -- sets the population and species dicts of self
        """

        # assign genome fitness based on fitness sharing within species
        genome_fitness = self.calc_fitness(genome_scores)

        # generate new genome species reps from current generation
        self.species = self.split_population()

        # generate new creatures for each species based on that species fitness
        species_fitness = self.calc_species_fitness(genome_fitness)
        species_children = self.calc_child_amounts(species_fitness)

        # generate new children for each species
        children = []
        children_species = {species: [] for species in species_children}
        for species in species_children:
            for _ in range(species_children[species]):
                child = self.get_new_child(
                    species, genome_fitness, mutation_consts)
                children.append(child)
                children_species[species].append(child)

        # re-assign population to children
        self.population = children

        # re-assign species so that each species is represented by a genome from the
        # previous generation
        self.species = children_species

    def get_parent(self, species: Genome, genome_fitness: Dict[Genome, float],
                   ignore: Genome = None) -> Genome:
        """Returns a parent based on the species and the genome fitness levels

        Arguments:
            species {Genome} -- species to choose from (unless interspecies mating happened)
            genome_fitness {Dict[Genome, float]} -- each genome and its fitness

        Returns:
            Genome -- parent chosen
        """
        options = [genome for genome in self.species[species] if genome is not ignore
                   or len(self.species[species]) == 1]
        probs = np.array([genome_fitness[genome] for genome in options])
        probs /= sum(probs)
        return np.random.choice(options, p=probs)

    def get_new_child(self, species: Genome, genome_fitness: Dict[Genome, float],
                      mutation_consts: Dict[str, float]) -> Genome:
        """Generates a new child and manages any new innovations and/or nodes
        that were created during its mutations

        Arguments:
            species {Genome} -- desired species rep of the child genome
            genome_fitness {Dict[Genome, float]} -- each genome and its fitness
            mutation_consts {Dict[str, float]} -- mutation probabilities

        Returns:
            Genome -- the child genome
        """

        # select two parents from that species
        # TODO add interspecies crossover
        parent_a: Genome = self.get_parent(species, genome_fitness)
        parent_b: Genome = self.get_parent(species, genome_fitness, parent_a)

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

        # get more fit parent
        fit_parent = max([parent_a, parent_b], key=lambda x: genome_fitness[x])
        fit_innovations = {inn.idx: inn for inn in fit_parent.innovations}

        # crossover disjoint and excess genes
        for idx in disjoint + excess:
            if idx in fit_innovations:
                parent_inn = fit_innovations[idx]
                innovation = Innovation(idx, parent_inn.src, parent_inn.dst,
                                        parent_inn.weight, parent_inn.enabled)
                child_innovations.append(innovation)

        # get necassary nodes from resulting innovations
        child_node_idxs = [idx for idx in range(self.inputs + self.outputs)]
        for innovation in child_innovations:
            child_node_idxs.append(innovation.src)
            child_node_idxs.append(innovation.dst)
        child_nodes = []
        for idx in set(child_node_idxs):
            child_nodes.append(Node(idx, self.nodes[idx]))

        # generate child genome
        child = Genome(tuple(child_nodes), tuple(child_innovations))

        # mutate child
        child = self.mutate_genome(child, mutation_consts)
        return child

    def mutate_genome(self, genome: Genome, mutation_consts: Dict[str, float]) -> Genome:
        """Mutates a genome with any of the following mutations:
        1. Mutate a weight value
        2. Split a connection with a node
        3. Add a connection between two nodes
        Then adds any mutations to the global node and innovation dicts

        Arguments:
            genome {Genome} -- genome

        Returns:
            Genome -- the mutated genome
        """
        new_genome = genome

        # 1. change weight mutation
        if np.random.random_sample() < mutation_consts['weight'] and new_genome.innovations:
            chosen = np.random.choice(new_genome.innovations)
            new_innovation = Innovation(chosen.idx, chosen.src, chosen.dst,
                                        np.random.random_sample() * 2 - 1, chosen.enabled)
            new_innovations = [new_innovation if inn.idx == new_innovation.idx else inn
                               for inn in new_genome.innovations]
            new_genome = Genome(new_genome.nodes, tuple(new_innovations))

        # 2. add connection mutation
        if np.random.random_sample() < mutation_consts['connection']:
            # get all connections outputing from the chosen source node
            connections = []
            for innovation in new_genome.innovations:
                connections.append((innovation.src, innovation.dst))

            # get all available new connections in genome
            options = []
            for src_node in new_genome.nodes:
                for dst_node in new_genome.nodes:
                    if dst_node.role is NodeType.INPUT or (src_node.role is NodeType.OUTPUT and dst_node.role is NodeType.OUTPUT):
                        continue
                    if (src_node.idx, dst_node.idx) not in connections:
                        options.append((src_node, dst_node))

            # if a new connection can be made, choose one
            if options:
                chosen_src, chosen_dst = options[np.random.randint(
                    0, len(options))]

                # check if innovation already happened
                for idx, (src, dst) in self.innovations.items():
                    if src == chosen_src.idx and dst == chosen_dst.idx:
                        new_innovation = Innovation(idx, src, dst,
                                                    np.random.random_sample(), True)
                        break

                # if this is the first time, add it to the innovations dict
                else:
                    innovation_idx = 0 if not self.innovations else max(self.innovations) + 1
                    new_innovation = Innovation(innovation_idx, chosen_src.idx, chosen_dst.idx,
                                                np.random.random_sample(), True)
                    self.innovations[new_innovation.idx] = (
                        chosen_src.idx, chosen_dst.idx)

                # return mutated genome
                new_innovations = [inn for inn in new_genome.innovations]
                new_innovations.append(new_innovation)
                new_genome = Genome(new_genome.nodes, tuple(new_innovations))
                return new_genome

        # 3. split connection mutation
        if np.random.random_sample() < mutation_consts['node'] and new_genome.innovations:

            # choose a connection
            chosen = np.random.choice(new_genome.innovations)

            # check if connection was split before
            if chosen in self.innovation_splits:
                src_innovation_id, split_node_id, dst_innovation_id = self.innovation_splits[
                    chosen]

            # if not, generate new ids for mutations and log mutation in innovation splits dict
            else:
                src_innovation_id = max(self.innovations) + 1
                dst_innovation_id = max(self.innovations) + 2
                split_node_id = max(self.nodes) + 1
                self.innovation_splits[chosen] = (src_innovation_id,
                                                  split_node_id, dst_innovation_id)
                self.nodes[split_node_id] = NodeType.HIDDEN
                self.innovations[src_innovation_id] = (
                    chosen.src, split_node_id)
                self.innovations[dst_innovation_id] = (
                    split_node_id, chosen.dst)

            new_node = Node(split_node_id, NodeType.HIDDEN)
            new_nodes = new_genome.nodes + (new_node, )
            src_innovation = Innovation(src_innovation_id,
                                        chosen.src, split_node_id, 1.0, True)
            dst_innovation = Innovation(dst_innovation_id,
                                        split_node_id, chosen.dst, chosen.weight, True)
            new_innovations = []
            for innovation in (new_genome.innovations + (src_innovation, dst_innovation)):
                if innovation is chosen:
                    new_innovations.append(Innovation(chosen.idx, chosen.src,
                                                      chosen.dst, chosen.weight, False))
                else:
                    new_innovations.append(innovation)
            new_genome = Genome(new_nodes, tuple(new_innovations))
            return new_genome
        return new_genome

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

    def print_status(self, verbose: int = 1):
        """Prints the status of the algorithm

        Keyword Arguments:
            verbose {int} -- verbosity_level (default: {1})
        """
        if verbose == 1:
            print(
                f"Total Species: {len(self.species)}\nTotal Nodes: {len(self.nodes)}\nTotal Innovations: {len(self.innovations)}")


if __name__ == "__main__":
    TEST = NEAT(150, 5, 5, {'c1': 1.0, 'c2': 1.0, 'c3': 1.0, 't': 3.0})
    TEST.new_generation([(gen, np.random.random_sample() * 100) for gen in TEST.population],
                        {'weight': 0.4, 'connection': 1, 'node': 0.0})
    TEST.new_generation([(gen, np.random.random_sample() * 100) for gen in TEST.population],
                        {'weight': 0.4, 'connection': 1, 'node': 0.0})
    print(TEST.get_diff(TEST.population[0], TEST.population[1]))
    print(TEST.genetic_distance(TEST.population[0], TEST.population[1]))
    print([inn for inn in TEST.population[0].innovations])
    print([inn for inn in TEST.population[1].innovations])
