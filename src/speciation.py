"""
Crossover and speciation for NEAT.
"""
from __future__ import annotations
import math
import random
from typing import Dict, List, Tuple
from .genome import Genome, NodeGene, ConnectionGene, NodeType


def crossover(parent_a: Genome, parent_b: Genome, rng: random.Random, saliency_aware: bool = False) -> Genome:
    """Crossover two genomes by aligning connection genes via innovation number.
    - Matching genes: inherited from either parent (fitter one more often).
    - Disjoint/excess genes: inherited from the fitter parent only.
    - Nodes: inherited from the union of matching connection genes' endpoints.

    If saliency_aware=True, matching genes are inherited from the parent with higher
    EMA saliency (|c.saliency|) rather than randomly. This preserves the most "tested"
    version of each gene - the one where the algorithm has the most information about
    how it affects fitness. This is the GATE principle applied to crossover.
    """
    if parent_a.fitness < parent_b.fitness:
        parent_a, parent_b = parent_b, parent_a
    # parent_a is now the fitter (or equal) parent.
    a_conns = parent_a.connections
    b_conns = parent_b.connections
    child = Genome()
    all_innovs = sorted(set(a_conns.keys()) | set(b_conns.keys()))
    for inv in all_innovs:
        if inv in a_conns and inv in b_conns:
            if saliency_aware:
                # Inherit from the parent with higher EMA saliency.
                sal_a = abs(a_conns[inv].saliency)
                sal_b = abs(b_conns[inv].saliency)
                if sal_a > sal_b + 1e-9:
                    src = a_conns[inv]
                elif sal_b > sal_a + 1e-9:
                    src = b_conns[inv]
                else:
                    # Tie: random.
                    src = a_conns[inv] if rng.random() < 0.5 else b_conns[inv]
            else:
                src = a_conns[inv] if rng.random() < 0.5 else b_conns[inv]
            enabled = src.enabled
            if (not a_conns[inv].enabled) or (not b_conns[inv].enabled):
                if rng.random() < 0.75:
                    enabled = False
            child_conn = ConnectionGene(
                innovation=inv, in_node=src.in_node, out_node=src.out_node,
                weight=src.weight, enabled=enabled,
                saliency=src.saliency,  # inherit saliency too
            )
            child.connections[inv] = child_conn
        elif inv in a_conns:
            src = a_conns[inv]
            child.connections[inv] = ConnectionGene(
                innovation=inv, in_node=src.in_node, out_node=src.out_node,
                weight=src.weight, enabled=src.enabled,
                saliency=src.saliency,
            )
        # else: in b only -> skip

    # Inherit nodes from parent_a, plus any referenced by child.connections.
    for nid, node in parent_a.nodes.items():
        child.nodes[nid] = NodeGene(
            id=node.id, type=node.type, activation=node.activation,
            saliency=node.saliency,
        )
    for c in child.connections.values():
        if c.in_node not in child.nodes and c.in_node in parent_b.nodes:
            src = parent_b.nodes[c.in_node]
            child.nodes[c.in_node] = NodeGene(
                id=src.id, type=src.type, activation=src.activation, saliency=src.saliency)
        if c.out_node not in child.nodes and c.out_node in parent_b.nodes:
            src = parent_b.nodes[c.out_node]
            child.nodes[c.out_node] = NodeGene(
                id=src.id, type=src.type, activation=src.activation, saliency=src.saliency)
    return child


def compatibility_distance(
    g1: Genome, g2: Genome,
    c1: float = 1.0, c2: float = 1.0, c3: float = 0.4,
) -> float:
    """Compute NEAT compatibility distance δ.
    δ = c1 * E / N + c2 * D / N + c3 * W
    where E = excess genes, D = disjoint genes, W = avg weight difference of matching genes,
    N = number of genes in larger genome (or 1 if small).
    """
    conn1 = g1.connections
    conn2 = g2.connections
    innovs1 = sorted(conn1.keys())
    innovs2 = sorted(conn2.keys())
    if not innovs1 or not innovs2:
        return float("inf")
    max1 = innovs1[-1]
    max2 = innovs2[-1]
    # Determine excess vs disjoint.
    # Excess: innovation numbers beyond the max of the other genome.
    # Disjoint: innovation numbers in one genome but not the other, up to the max of the smaller.
    all_innovs = set(innovs1) | set(innovs2)
    max_min = min(max1, max2)  # boundary for excess vs disjoint
    excess = 0
    disjoint = 0
    matching_w_diff = []
    for inv in all_innovs:
        in1 = inv in conn1
        in2 = inv in conn2
        if in1 and in2:
            matching_w_diff.append(abs(conn1[inv].weight - conn2[inv].weight))
        else:
            if inv > max_min:
                excess += 1
            else:
                disjoint += 1
    n = max(len(innovs1), len(innovs2))
    n = max(n, 1)
    w_avg = sum(matching_w_diff) / len(matching_w_diff) if matching_w_diff else 0.0
    # Also consider node gene differences for hidden node activations.
    # (Optional: include node delta in the distance.)
    distance = (c1 * excess) / n + (c2 * disjoint) / n + c3 * w_avg
    return distance


def behavioral_distance(g1: Genome, g2: Genome, behavioral_weight: float = 1.0) -> float:
    """Distance incorporating behavioral signature (end-of-episode trajectory statistics).
    If behavior is None for either genome, falls back to pure structural distance."""
    d_struct = compatibility_distance(g1, g2)
    if g1.behavior is None or g2.behavior is None:
        return d_struct
    b1 = g1.behavior
    b2 = g2.behavior
    # Euclidean distance between behavioral signatures.
    d_beh = math.sqrt(sum((a - b) ** 2 for a, b in zip(b1, b2)))
    return d_struct + behavioral_weight * d_beh


class Species:
    """A species is a cluster of genomes with similar topology/behavior."""
    def __init__(self, id: int, representative: Genome):
        self.id = id
        self.representative = representative
        self.members: List[Genome] = [representative]
        self.best_fitness: float = -float("inf")
        self.stagnation: int = 0  # generations since improvement
        self.fitness_history: List[float] = []

    def add(self, genome: Genome):
        genome.species_id = self.id
        self.members.append(genome)

    def reset(self, representative: Genome):
        self.representative = representative
        self.members = [representative]

    def update_stagnation(self):
        cur_best = max(g.fitness for g in self.members) if self.members else -float("inf")
        if cur_best > self.best_fitness + 1e-6:
            self.best_fitness = cur_best
            self.stagnation = 0
        else:
            self.stagnation += 1
        self.fitness_history.append(cur_best)


class Speciator:
    """Assigns genomes to species based on compatibility distance to existing representatives."""
    def __init__(self, threshold: float = 3.0, distance_fn=None):
        self.threshold = threshold
        self.distance_fn = distance_fn or compatibility_distance
        self.species: List[Species] = []
        self._next_species_id = 0

    def speciate(self, population: List[Genome]):
        # Clear member lists of existing species, keep representatives.
        for sp in self.species:
            sp.members = []
        # Assign each genome.
        for genome in population:
            placed = False
            for sp in self.species:
                d = self.distance_fn(sp.representative, genome)
                if d < self.threshold:
                    sp.add(genome)
                    placed = True
                    break
            if not placed:
                # New species.
                sp = Species(self._next_species_id, genome)
                self._next_species_id += 1
                self.species.append(sp)
        # Remove empty species.
        self.species = [sp for sp in self.species if sp.members]
        # Update stagnation for each species.
        for sp in self.species:
            sp.update_stagnation()

    def adjust_fitness(self):
        """Apply explicit fitness sharing within each species."""
        for sp in self.species:
            n = len(sp.members)
            if n == 0:
                continue
            for g in sp.members:
                g.adjusted_fitness = g.fitness / n
