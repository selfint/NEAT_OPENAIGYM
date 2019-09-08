from typing import List, Callable, Tuple
import numpy as np
from data_types import Node, Innovation, NodeType, Genome
from activations import steep_sigmoid


class NeuralNetwork:
    """Custom network class for NEAT implementation

    Raises:
        ValueError: inputs not in right dimensions
        ValueError: node not in network nodes

    """

    def __init__(self, genome: Genome, activation: Callable[[float], float] = steep_sigmoid):
        """Creates a neural network object from Node and Innovation objects.
        Output is calculated recursively from the output nodes.

        Arguments:
            genome {Genome} -- genome to generate network for

        Keyword Arguments:
            activation {function} -- node activation function (default: {steep_sigmoid})
        """
        # TODO add bias node
        self.genome = genome

        # copy nodes and innovations so the genome remains unchanged
        self.nodes = self.genome.nodes
        self.innovations = self.genome.innovations
        self.input_nodes = [
            node for node in self.nodes if node.role is NodeType.INPUT]
        self.output_nodes = [
            node for node in self.nodes if node.role is NodeType.OUTPUT]
        self.activation = activation
        self.inputs = dict()

    def predict(self, inputs: np.array) -> np.array:
        """Calculates the output of the network given an input

        Arguments:
            inputs {np.array} -- input array

        Returns:
            np.array -- output array
        """

        # check input length
        if len(inputs) != len(self.input_nodes):
            raise ValueError(
                'length of inputs did not match length of network inputs')

        # assign input to input nodes
        self.inputs = dict()
        for i, node in enumerate(self.input_nodes):
            self.inputs[node] = inputs[i]

        # get output of output layer
        return np.array([self.get_node_output(node, []) for node in self.output_nodes])

    def get_node_output(self, node: Node, exclude: List[Innovation]) -> float:
        """Calculates the output of a single node, recursively

        Arguments:
            node {Node} -- the node to get output from
            exclude {List[Innovation]} -- innovations to ignore

        Returns:
            float -- the output of the node
        """
        if node.role is NodeType.INPUT:
            return self.inputs[node]

        return self.activation(sum(self.get_node_output(input_node,
                                                        [input_innovation] + exclude) *
                                   input_innovation.weight *
                                   int(input_innovation.enabled)
                                   for input_innovation, input_node
                                   in self.get_node_inputs(node, exclude)))

    def get_node_inputs(self, node: Node, exclude: List[Innovation]) -> List[Tuple[Innovation, Node]]:
        """Returns all inovations that OUTPUT into the node

        Arguments:
            node {Node} -- node to find inputs for
            exclude {List[Innovation]} -- innovations to ignore

        Returns:
            List[Innovation] -- all innovations outputing into node
        """
        return [(innovation, self.get_node(innovation.src)) for innovation in self.innovations
                if innovation.dst == node.idx and innovation not in exclude]

    def get_node(self, node_idx: int) -> Node:
        """Returns node by its idx

        Arguments:
            node_idx {int} -- idx of node

        Returns:
            Node -- node with that idx
        """
        for node in self.nodes:
            if node.idx == node_idx:
                return node
        raise ValueError(f'Node with id {node_idx} not in available nodes')


if __name__ == "__main__":
    from itertools import product
    inputs = 5
    outputs = 5
    nodes = [Node(idx, NodeType.INPUT if idx < inputs else NodeType.OUTPUT)
             for idx in range(inputs + outputs)]
    innovations = [Innovation(idx, i,
                              j+inputs, np.random.random_sample() * 2 - 1, True)
                   for idx, (i, j) in enumerate(product(range(inputs), range(outputs)))]
    genome = Genome(tuple(nodes), tuple(innovations))
    network = NeuralNetwork(genome)
    print(network.predict(np.random.random(inputs)))
    print(genome.nodes)
