from typing import List
from dataclasses import dataclass
from enum import Enum


@dataclass
class Innovation:
    idx: int
    src: int
    dst: int
    weight: float
    enabled: bool


class NodeType(Enum):
    INPUT = 'input'
    HIDDEN = 'hidden'
    OUTPUT = 'output'


@dataclass
class Node:
    idx: int
    role: NodeType


@dataclass
class Genome:
    nodes: List[Node]
    innovations: List[Innovation]
