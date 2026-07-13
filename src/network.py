"""
Feedforward network evaluation from a Genome.

We compile a Genome into a topologically-sorted evaluation order, then evaluate it
in a vectorized way using numpy for speed. Activation functions are inlined for the
common cases.
"""
from __future__ import annotations
import math
from typing import List, Optional, Tuple
import numpy as np

from .genome import Genome, NodeType, Activation


class FeedForwardNetwork:
    """Evaluates a genome as a feedforward network.

    Construction:
      1. Topologically sort nodes by enabled connections.
      2. Build a list of (in_nodes_array, weights_array, out_node, activation).
    Eval:
      For each node in topo order, compute weighted sum of incoming activations, apply
      activation, store into the activation array.
    """

    __slots__ = ("n_inputs", "n_outputs", "input_ids", "output_ids", "bias_ids",
                 "_order", "_in_lists", "_in_weights", "_out_ids", "_acts", "_activations",
                 "_node_index", "_num_nodes")

    def __init__(self, genome: Genome):
        self.input_ids = genome.input_ids()
        self.output_ids = genome.output_ids()
        self.bias_ids = genome.bias_ids()
        self.n_inputs = len(self.input_ids)
        self.n_outputs = len(self.output_ids)

        # Build adjacency for topological sort.
        all_node_ids = sorted(genome.nodes.keys())
        # Map node id -> index into activation vector.
        self._node_index = {nid: i for i, nid in enumerate(all_node_ids)}
        self._num_nodes = len(all_node_ids)

        incoming = {nid: [] for nid in all_node_ids}  # nid -> list of (in_nid, weight)
        for c in genome.connections.values():
            if not c.enabled:
                continue
            if c.in_node not in self._node_index or c.out_node not in self._node_index:
                continue
            incoming[c.out_node].append((c.in_node, c.weight))

        # Topological sort (Kahn's algorithm). Input/bias nodes go first naturally
        # because they have no incoming edges.
        in_deg = {nid: len(incoming[nid]) for nid in all_node_ids}
        from collections import deque
        queue = deque([nid for nid in all_node_ids if in_deg[nid] == 0])
        order: List[int] = []
        # Process queue
        while queue:
            n = queue.popleft()
            order.append(n)
            # Need outgoing edges. Build them once.
        # Actually we need outgoing. Let me build it.
        outgoing = {nid: [] for nid in all_node_ids}
        for nid in all_node_ids:
            for (in_nid, w) in incoming[nid]:
                outgoing[in_nid].append(nid)
        # Redo Kahn's with outgoing
        queue = deque([nid for nid in all_node_ids if in_deg[nid] == 0])
        order = []
        while queue:
            n = queue.popleft()
            order.append(n)
            for m in outgoing[n]:
                in_deg[m] -= 1
                if in_deg[m] == 0:
                    queue.append(m)
        if len(order) != self._num_nodes:
            # Cycle detected; should not happen for feedforward genomes.
            # Fallback: just use node id order (will be wrong, but won't crash).
            order = all_node_ids

        # Build evaluation lists, skipping input/bias nodes (they are set externally).
        input_set = set(self.input_ids) | set(self.bias_ids)
        self._order = [n for n in order if n not in input_set]
        self._in_lists = []
        self._in_weights = []
        self._out_ids = []
        self._activations = []
        for nid in self._order:
            in_nodes = incoming[nid]
            if not in_nodes:
                # Standalone hidden node with no inputs -> always outputs activation(0).
                # Keep it but with empty inputs.
                self._in_lists.append(np.array([], dtype=np.int64))
                self._in_weights.append(np.array([], dtype=np.float64))
            else:
                in_arr = np.array([self._node_index[i] for (i, _) in in_nodes], dtype=np.int64)
                w_arr = np.array([w for (_, w) in in_nodes], dtype=np.float64)
                self._in_lists.append(in_arr)
                self._in_weights.append(w_arr)
            self._out_ids.append(self._node_index[nid])
            self._activations.append(genome.nodes[nid].activation)
        # Preallocate activation buffer.
        self._acts = np.zeros(self._num_nodes, dtype=np.float64)

    def activate(self, inputs: np.ndarray | List[float]) -> np.ndarray:
        """Evaluate the network on a single input vector. Returns the output activations."""
        acts = self._acts
        # Reset all to 0 (only the inputs/bias will be set, others computed).
        acts.fill(0.0)
        # Set input nodes.
        for i, nid in enumerate(self.input_ids):
            acts[self._node_index[nid]] = float(inputs[i])
        # Set bias nodes to 1.0.
        for nid in self.bias_ids:
            acts[self._node_index[nid]] = 1.0
        # Evaluate hidden/output nodes in topo order.
        for i in range(len(self._order)):
            in_arr = self._in_lists[i]
            if in_arr.size == 0:
                val = 0.0
            else:
                val = float(np.dot(acts[in_arr], self._in_weights[i]))
            acts[self._out_ids[i]] = self._apply_activation(val, self._activations[i])
        # Return output activations.
        return np.array([acts[self._node_index[o]] for o in self.output_ids], dtype=np.float64)

    @staticmethod
    def _apply_activation(x: float, name: str) -> float:
        if name == "tanh":
            return math.tanh(x)
        elif name == "relu":
            return x if x > 0.0 else 0.0
        elif name == "sigmoid":
            if x >= 0:
                z = math.exp(-x)
                return 1.0 / (1.0 + z)
            z = math.exp(x)
            return z / (1.0 + z)
        elif name == "identity":
            return x
        elif name == "gaussian":
            return math.exp(-(x * x) / 2.0)
        else:
            return math.tanh(x)


def evaluate_episode(
    net: FeedForwardNetwork,
    env,
    seed: Optional[int] = None,
    max_steps: int = 1000,
    render: bool = False,
) -> Tuple[float, int, dict]:
    """Roll out the network in env for one episode. Returns (total_reward, steps, info_dict)."""
    obs, _ = env.reset(seed=seed) if seed is not None else env.reset()
    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False
    # For behavioral characterization: collect mean/var of obs and action stats.
    obs_sum = np.zeros_like(obs, dtype=np.float64)
    obs_sq_sum = np.zeros_like(obs, dtype=np.float64)
    action_sum = 0.0
    action_sq_sum = 0.0
    while not (terminated or truncated) and steps < max_steps:
        out = net.activate(obs)
        # For discrete 2-output: pick argmax. For single-output: threshold at 0.
        if out.size > 1:
            action = int(np.argmax(out))
        else:
            action = 1 if out[0] > 0.0 else 0
        obs_sum += obs
        obs_sq_sum += obs * obs
        action_sum += action
        action_sq_sum += action * action
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        if render:
            env.render()
    info = {
        "obs_mean": obs_sum / max(1, steps),
        "obs_var": obs_sq_sum / max(1, steps) - (obs_sum / max(1, steps)) ** 2,
        "action_mean": action_sum / max(1, steps),
        "action_var": action_sq_sum / max(1, steps) - (action_sum / max(1, steps)) ** 2,
        "steps": steps,
    }
    return total_reward, steps, info
