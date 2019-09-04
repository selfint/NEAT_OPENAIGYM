import numpy as np


class Genome:

    def __init__(self, inputs, outputs, connected=True):
        self.inputs = inputs
        self.outputs = outputs
