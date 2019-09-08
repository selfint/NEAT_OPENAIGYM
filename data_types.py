from typing import Tuple
from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class Innovation():
    idx: int
    src: int
    dst: int
    weight: float
    enabled: bool


class NodeType(Enum):
    INPUT = 'input'
    HIDDEN = 'hidden'
    OUTPUT = 'output'
    BIAS = 'bias'


@dataclass(frozen=True)
class Node:
    idx: int
    role: NodeType


@dataclass(frozen=True)
class Genome:
    nodes: Tuple[Node, ...]
    innovations: Tuple[Innovation, ...]

    # TODO add __getitem__ method hook that returns nodes and innovations as dicts
