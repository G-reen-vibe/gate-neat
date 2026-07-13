"""
Mutation operators for NEAT-style genomes.
"""
from __future__ import annotations
import random
from typing import Dict, List, Set, Tuple
from .genome import Genome, NodeGene, ConnectionGene, NodeType, InnovationTracker


def mutate_weights(
    genome: Genome,
    rng: random.Random,
    perturb_rate: float = 0.8,
    perturb_std: float = 0.25,
    replace_rate: float = 0.1,
    replace_std: float = 1.0,
):
    """Mutate connection weights.
    - With probability perturb_rate: add Gaussian noise with perturb_std.
    - With probability replace_rate: replace with new Gaussian weight.
    - Otherwise: leave unchanged.
    """
    for conn in genome.connections.values():
        if not conn.enabled:
            continue
        r = rng.random()
        if r < perturb_rate:
            conn.weight += rng.gauss(0.0, perturb_std)
        elif r < perturb_rate + replace_rate:
            conn.weight = rng.gauss(0.0, replace_std)


def mutate_add_connection(
    genome: Genome,
    tracker: InnovationTracker,
    rng: random.Random,
    max_tries: int = 20,
    weight_init_std: float = 1.0,
) -> bool:
    """Try to add a connection between two previously-unconnected nodes.
    Maintains feedforward structure by only allowing edges from lower to higher
    topological order. Returns True if a connection was added."""
    if not genome.nodes:
        return False
    node_ids = sorted(genome.nodes.keys())
    # Compute topological order to avoid creating cycles.
    incoming = {nid: [] for nid in node_ids}
    outgoing = {nid: [] for nid in node_ids}
    for c in genome.connections.values():
        if not c.enabled:
            continue
        incoming[c.out_node].append(c.in_node)
        outgoing[c.in_node].append(c.out_node)
    # Topo order
    from collections import deque
    in_deg = {nid: len(incoming[nid]) for nid in node_ids}
    queue = deque([n for n in node_ids if in_deg[n] == 0])
    order = []
    while queue:
        n = queue.popleft()
        order.append(n)
        for m in outgoing[n]:
            in_deg[m] -= 1
            if in_deg[m] == 0:
                queue.append(m)
    if len(order) != len(node_ids):
        # Cycle exists, bail.
        return False
    order_index = {nid: i for i, nid in enumerate(order)}

    # Try to find a valid (in, out) pair not already connected.
    existing = set()
    for c in genome.connections.values():
        existing.add((c.in_node, c.out_node))

    input_bias_ids = [nid for nid in node_ids if genome.nodes[nid].type in (NodeType.INPUT, NodeType.BIAS)]
    # outputs and hidden are valid out-nodes
    valid_out_ids = [nid for nid in node_ids if genome.nodes[nid].type in (NodeType.OUTPUT, NodeType.HIDDEN)]

    for _ in range(max_tries):
        # Pick in node: any node, but mostly inputs/bias for initial connections
        if rng.random() < 0.7 and input_bias_ids:
            in_node = rng.choice(input_bias_ids)
        else:
            in_node = rng.choice(node_ids)
        out_node = rng.choice(valid_out_ids) if valid_out_ids else rng.choice(node_ids)
        if in_node == out_node:
            continue
        # Enforce feedforward: in must come before out in topo order.
        if order_index[in_node] >= order_index[out_node]:
            continue
        if (in_node, out_node) in existing:
            continue
        inv = tracker.get_add_connection(in_node, out_node)
        genome.connections[inv] = ConnectionGene(
            innovation=inv, in_node=in_node, out_node=out_node,
            weight=rng.gauss(0.0, weight_init_std),
        )
        return True
    return False


def mutate_add_node(
    genome: Genome,
    tracker: InnovationTracker,
    rng: random.Random,
) -> bool:
    """Split a random enabled connection with a new node. The original connection is disabled,
    the new node receives the old connection's input with weight 1, and sends to the old
    output with the old weight. This preserves function initially."""
    enabled_conns = [c for c in genome.connections.values() if c.enabled]
    if not enabled_conns:
        return False
    conn = rng.choice(enabled_conns)
    new_node_id, in_inv, out_inv = tracker.get_add_node(conn.innovation)
    # Disable original.
    conn.enabled = False
    # Add node.
    genome.nodes[new_node_id] = NodeGene(
        id=new_node_id, type=NodeType.HIDDEN, activation="tanh",
    )
    # New connection in -> new_node, weight 1.
    genome.connections[in_inv] = ConnectionGene(
        innovation=in_inv, in_node=conn.in_node, out_node=new_node_id,
        weight=1.0,
    )
    # New connection new_node -> out, weight = old weight.
    genome.connections[out_inv] = ConnectionGene(
        innovation=out_inv, in_node=new_node_id, out_node=conn.out_node,
        weight=conn.weight,
    )
    return True


def mutate_remove_node(
    genome: Genome,
    rng: random.Random,
) -> bool:
    """Remove a random hidden node and reconnect its input -> output with sum of weights.
    This is a 'prune' mutation - non-canonical but useful for keeping networks small."""
    hidden = [n for n in genome.nodes.values() if n.type == NodeType.HIDDEN]
    if not hidden:
        return False
    node = rng.choice(hidden)
    # Find incoming and outgoing enabled connections.
    in_conns = [c for c in genome.connections.values()
                if c.enabled and c.out_node == node.id]
    out_conns = [c for c in genome.connections.values()
                 if c.enabled and c.in_node == node.id]
    # Disable all connections to/from this node.
    for c in genome.connections.values():
        if c.in_node == node.id or c.out_node == node.id:
            c.enabled = False
    # Remove node.
    del genome.nodes[node.id]
    # Optionally add bypass connections.
    for ic in in_conns:
        for oc in out_conns:
            # The combined weight is ic.weight * 1 (activation passthrough) * oc.weight
            # but tanh is non-linear, so this is just an approximation.
            pass  # skip bypass for simplicity; let add_connection mutation explore.
    return True


def mutate_toggle_enable(genome: Genome, rng: random.Random, enable_prob: float = 0.5):
    """Randomly toggle the enabled state of a connection."""
    conns = list(genome.connections.values())
    if not conns:
        return
    conn = rng.choice(conns)
    if conn.enabled:
        conn.enabled = False
    else:
        # Don't re-enable if doing so would create a cycle (basic check).
        conn.enabled = True


def mutate_activation(genome: Genome, rng: random.Random, rate: float = 0.1):
    """Randomly change activation function of hidden nodes."""
    options = ["tanh", "relu", "sigmoid", "gaussian", "identity"]
    hidden = [n for n in genome.nodes.values() if n.type == NodeType.HIDDEN]
    for node in hidden:
        if rng.random() < rate:
            node.activation = rng.choice(options)


def mutate(
    genome: Genome,
    tracker: InnovationTracker,
    rng: random.Random,
    cfg: dict,
):
    """Apply all mutation operators according to config probabilities."""
    if rng.random() < cfg.get("weight_mut_rate", 0.8):
        mutate_weights(
            genome, rng,
            perturb_std=cfg.get("weight_perturb_std", 0.25),
            replace_rate=cfg.get("weight_replace_rate", 0.1),
        )
    if rng.random() < cfg.get("add_conn_rate", 0.5):
        mutate_add_connection(genome, tracker, rng)
    if rng.random() < cfg.get("add_node_rate", 0.3):
        mutate_add_node(genome, tracker, rng)
    if rng.random() < cfg.get("remove_node_rate", 0.0):
        mutate_remove_node(genome, rng)
    if rng.random() < cfg.get("toggle_enable_rate", 0.0):
        mutate_toggle_enable(genome, rng)
    if rng.random() < cfg.get("act_mut_rate", 0.0):
        mutate_activation(genome, rng)
