from math import exp


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + exp(-x))

def relu(x: float) -> float:
    return max(0.0, x)
