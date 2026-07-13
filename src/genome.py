"""
Genome representation for NEAT-style algorithms.

A Genome is a genetic encoding of a neural network:
  - Node genes: each node has an id, type (input/bias/hidden/output), and activation
  - Connection genes: each connection has innovation number, in/out node ids, weight, enabled flag

Innovation numbers are global so that the same structural mutation gets the same innovation
number across the population, enabling meaningful crossover (historical markings).
"""
from __future__ import annotations
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple


class NodeType(Enum):
    INPUT = "input"
    BIAS = "bias"
    HIDDEN = "hidden"
    OUTPUT = "output"


class Activation:
    """Activation functions as simple callables (kept lightweight for speed)."""
    @staticmethod
    def tanh(x: float) -> float:
        import math
        return math.tanh(x)
    @staticmethod
    def relu(x: float) -> float:
        return x if x > 0.0 else 0.0
    @staticmethod
    def sigmoid(x: float) -> float:
        import math
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        z = math.exp(x)
        return z / (1.0 + z)
    @staticmethod
    def identity(x: float) -> float:
        return x
    @staticmethod
    def gaussian(x: float) -> float:
        import math
        return math.exp(-(x * x) / 2.0)


ACTIVATIONS = {
    "tanh": Activation.tanh,
    "relu": Activation.relu,
    "sigmoid": Activation.sigmoid,
    "identity": Activation.identity,
    "gaussian": Activation.gaussian,
}


@dataclass
class NodeGene:
    id: int
    type: NodeType
    activation: str = "tanh"
    # Aggregate state used by GATE's saliency mechanism; not part of canonical NEAT.
    saliency: float = 0.0
    age: int = 0  # how many generations since this node was created


@dataclass
class ConnectionGene:
    innovation: int
    in_node: int
    out_node: int
    weight: float
    enabled: bool = True
    # GATE-specific: tracks how much this connection "wants" to change.
    saliency: float = 0.0
    age: int = 0


@dataclass
class Genome:
    nodes: Dict[int, NodeGene] = field(default_factory=dict)
    connections: Dict[int, ConnectionGene] = field(default_factory=dict)
    fitness: float = 0.0
    adjusted_fitness: float = 0.0
    # behavioral signature (for GATE speciation); filled by evaluator.
    behavior: Optional[Tuple[float, ...]] = None
    # species id assigned by the speciator
    species_id: int = -1

    # ---- structural queries ----
    def input_ids(self) -> List[int]:
        return sorted([n.id for n in self.nodes.values() if n.type == NodeType.INPUT])

    def output_ids(self) -> List[int]:
        return sorted([n.id for n in self.nodes.values() if n.type == NodeType.OUTPUT])

    def bias_ids(self) -> List[int]:
        return sorted([n.id for n in self.nodes.values() if n.type == NodeType.BIAS])

    def hidden_ids(self) -> List[int]:
        return sorted([n.id for n in self.nodes.values() if n.type == NodeType.HIDDEN])

    def is_feedforward(self) -> bool:
        """Sanity check: ensure no cycles. NEAT-Cartpole uses feedforward only."""
        # Topological sort check
        incoming = {nid: [] for nid in self.nodes}
        outgoing = {nid: [] for nid in self.nodes}
        for c in self.connections.values():
            if not c.enabled:
                continue
            if c.in_node not in self.nodes or c.out_node not in self.nodes:
                return False
            outgoing[c.in_node].append(c.out_node)
            incoming[c.out_node].append(c.in_node)
        # Kahn's algorithm
        from collections import deque
        queue = deque([n for n in self.nodes if len(incoming[n]) == 0])
        visited = 0
        in_deg = {n: len(incoming[n]) for n in self.nodes}
        while queue:
            n = queue.popleft()
            visited += 1
            for m in outgoing[n]:
                in_deg[m] -= 1
                if in_deg[m] == 0:
                    queue.append(m)
        return visited == len(self.nodes)

    def copy(self) -> "Genome":
        g = Genome()
        for nid, node in self.nodes.items():
            g.nodes[nid] = NodeGene(
                id=node.id, type=node.type, activation=node.activation,
                saliency=node.saliency, age=node.age,
            )
        for inv, conn in self.connections.items():
            g.connections[inv] = ConnectionGene(
                innovation=conn.innovation, in_node=conn.in_node, out_node=conn.out_node,
                weight=conn.weight, enabled=conn.enabled,
                saliency=conn.saliency, age=conn.age,
            )
        g.fitness = self.fitness
        g.adjusted_fitness = self.adjusted_fitness
        g.behavior = self.behavior
        g.species_id = self.species_id
        return g

    def num_enabled_connections(self) -> int:
        return sum(1 for c in self.connections.values() if c.enabled)

    def num_hidden(self) -> int:
        return sum(1 for n in self.nodes.values() if n.type == NodeType.HIDDEN)


class InnovationTracker:
    """Hands out globally-unique innovation numbers and node ids for structural mutations.

    Two genomes that perform the same structural mutation (same in/out pair, or same
    connection being split) get the same innovation number, which lets crossover line
    up their genes by historical marking (the core NEAT trick)."""

    def __init__(self):
        self._next_innovation = 0
        self._next_node_id = 0
        # maps (in_node, out_node) -> innovation number for "add connection" mutations
        self._conn_history: Dict[Tuple[int, int], int] = {}
        # maps innovation number being split -> (new_node_id, in_innov, out_innov)
        self._node_history: Dict[int, Tuple[int, int, int]] = {}

    def new_node_id(self) -> int:
        nid = self._next_node_id
        self._next_node_id += 1
        return nid

    def new_innovation(self) -> int:
        inv = self._next_innovation
        self._next_innovation += 1
        return inv

    def get_add_connection(self, in_node: int, out_node: int) -> int:
        """Return innovation number for adding a connection between in_node and out_node.
        Reuses the same number if this exact addition has been seen before."""
        key = (in_node, out_node)
        if key not in self._conn_history:
            self._conn_history[key] = self.new_innovation()
        return self._conn_history[key]

    def get_add_node(self, conn_innov: int) -> Tuple[int, int, int]:
        """Return (new_node_id, in_conn_innov, out_conn_innov) for splitting the connection
        with innovation number conn_innov. Reuses history so the same split yields the same
        node id and innovation numbers."""
        if conn_innov not in self._node_history:
            new_node = self.new_node_id()
            in_inv = self.new_innovation()
            out_inv = self.new_innovation()
            self._node_history[conn_innov] = (new_node, in_inv, out_inv)
        return self._node_history[conn_innov]


def make_initial_genome(
    n_inputs: int,
    n_outputs: int,
    tracker: InnovationTracker,
    output_activation: str = "tanh",
    connect_input_output: bool = True,
    weight_init_std: float = 1.0,
    rng: random.Random | None = None,
) -> Genome:
    """Create a minimal starting genome: inputs + bias + outputs, fully connected (or unconnected)."""
    rng = rng or random
    g = Genome()
    input_ids = []
    for i in range(n_inputs):
        nid = tracker.new_node_id()
        g.nodes[nid] = NodeGene(id=nid, type=NodeType.INPUT, activation="identity")
        input_ids.append(nid)
    bias_id = tracker.new_node_id()
    g.nodes[bias_id] = NodeGene(id=bias_id, type=NodeType.BIAS, activation="identity")
    output_ids = []
    for _ in range(n_outputs):
        nid = tracker.new_node_id()
        g.nodes[nid] = NodeGene(id=nid, type=NodeType.OUTPUT, activation=output_activation)
        output_ids.append(nid)

    if connect_input_output:
        for in_id in input_ids + [bias_id]:
            for out_id in output_ids:
                inv = tracker.get_add_connection(in_id, out_id)
                g.connections[inv] = ConnectionGene(
                    innovation=inv, in_node=in_id, out_node=out_id,
                    weight=rng.gauss(0.0, weight_init_std),
                )
    return g
