from math import exp


def sigmoid(x):
    return 1.0 / (1.0 + exp(-x))

def relu(x):
    return max(0, x)
