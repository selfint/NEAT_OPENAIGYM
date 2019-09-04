import numpy as np
from genome import Genome


class NEAT:

    def __init__(self, population_size, inputs, outputs):
        self.population_size = population_size
        self.population = [Genome(inputs, outputs) for _ in range(population_size)]
        


