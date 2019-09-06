from typing import List
import numpy as np
from data_types import Node, Innovation, NodeType
from activations import sigmoid


class NeuralNetwork:
    """Custom network class for NEAT implementation

    Raises:
        ValueError: inputs not in right dimensions
        ValueError: node not in network nodes

    """

    def __init__(self, nodes: List[Node], innovations: List[Innovation], activation=sigmoid):
        """Creates a neural network object from Node and Innovation objects.
        Output is calculated recursively from the output nodes.

        Arguments:
            nodes {List[Node]} -- list of nodes
            innovations {List[Innovation]} -- list of innovations
        """
        self.nodes = nodes
        self.innovations = innovations
        self.input_nodes = [
            node for node in self.nodes if node.role is NodeType.INPUT]
        self.output_nodes = [
            node for node in self.nodes if node.role is NodeType.OUTPUT]
        self.activation = activation

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

        # assign input values to input nodes, use the node bias for this
        for i in range(len(inputs)):
            self.input_nodes[i].bias = inputs[i]

        # get output of output layer
        return [self.get_node_output(node) for node in self.output_nodes]

    def get_node_output(self, node: Node) -> float:
        """Calculates the output of a single node, recursively

        Arguments:
            node {Node} -- the node to get output from

        Returns:
            float -- the output of the node
        """

        return node.bias + self.activation(sum(self.get_node_output(input_node) *
                                               input_innovation.weight *
                                               input_innovation.enabled
                                               for input_innovation, input_node
                                               in self.get_node_inputs(node)))

    def get_node_inputs(self, node: Node) -> List[Innovation]:
        """Returns all inovations that OUTPUT into the node

        Arguments:
            node {Node} -- node to find inputs for

        Returns:
            List[Innovation] -- all innovations outputing into node
        """
        return [(innovation, self.get_node(innovation.src)) for innovation in self.innovations
                if innovation.dst == node.idx]

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
