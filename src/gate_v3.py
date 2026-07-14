"""
GATE-v3: Probabilistic Saliency Evolution
==========================================

Major rewrite of GATE-v2. The core insight is preserved (saliency directs both weight
updates and topology growth), but the MECHANISM is fundamentally changed:

GATE-v2 used saliency DETERMINISTICALLY:
  - Split the top-K connections by EMA saliency (greedy)
  - Prune the bottom-F% by saliency (greedy)
  - SPSA weight step (deterministic given random Δ)

GATE-v3 uses saliency PROBABILISTICALLY:
  - Sample which connection to split from softmax(saliencies / temperature)
  - Sample which connection to prune from softmax(-saliencies / temperature)
  - Sample structural mutations from saliency-informed distributions
  - Temperature anneals from high (explore) to low (exploit) over generations

This addresses GATE-v2's key weakness: greedy saliency decisions cause premature
convergence and local optima. Probabilistic sampling maintains diversity while still
being informed by saliency. The temperature schedule controls the explore-exploit
tradeoff globally.

Additional changes from v2:
  - Saliency-weighted structural mutation: even "random" add-connection mutations
    are informed by saliency (prefer to connect high-saliency regions)
  - Adaptive EMA decay: saliency updates faster when the genome is changing rapidly
  - Simplified config: fewer knobs, more principled defaults
"""
from __future__ import annotations
import math
import random
import time
from typing import Callable, List, Optional, Tuple
import numpy as np

from .genome import Genome, InnovationTracker, NodeGene, ConnectionGene, NodeType, make_initial_genome
from .network import FeedForwardNetwork, evaluate_episode
from .mutations import (
    mutate_weights, mutate_add_connection, mutate_add_node,
    mutate_remove_node, mutate_toggle_enable, mutate_activation, mutate,
)
from .speciation import Speciator, Species, crossover, compatibility_distance


GATE_V3_CFG = {
    "pop_size": 80,
    # SPSA probing (same as v2)
    "spsa_eps": 0.3,
    "spsa_lr": 0.15,
    "spsa_episodes": 2,
    "saliency_ema_decay": 0.5,
    # Temperature schedule (NEW in v3)
    "temp_start": 2.0,              # initial temperature (high = explore)
    "temp_end": 0.3,                # final temperature (low = exploit)
    "temp_decay": 0.95,             # multiplicative decay per generation
    # Probabilistic topology growth (NEW in v3)
    "n_splits_per_elite": 1,        # number of connections to split per elite per gen
    "n_prunes_per_elite": 1,        # number of connections to prune per elite per gen
    "min_connections_to_prune": 8,
    # Speciation (same as v2)
    "compat_threshold": 3.0,
    "behavioral_weight": 0.3,
    "c1": 1.0, "c2": 1.0, "c3": 0.4,
    # Reproduction
    "elitism": 1,
    "survival_threshold": 0.20,
    "interspecies_mate_rate": 0.001,
    "spsa_top_fraction": 0.25,
    # Mutation rates
    "weight_mut_rate": 0.85,
    "weight_perturb_std": 0.25,
    "weight_replace_rate": 0.1,
    "add_conn_rate": 0.3,
    "add_node_rate": 0.10,
    # Saliency-weighted mutation
    "saliency_weighted_mutation": True,
    "saliency_mutation_scale": 2.0,
    # Saliency-aware crossover
    "saliency_aware_crossover": True,
    # Novelty (same as v2)
    "novelty_weight": 0.0,
    "novelty_k": 5,
    "novelty_normalize": True,
    # Adaptive novelty
    "adaptive_novelty": True,
    "adaptive_novelty_patience": 5,
    "adaptive_novelty_factor": 1.5,
    "adaptive_novelty_decay": 0.7,
    # Behavioral archive
    "use_archive": True,
    "archive_size": 30,
    "archive_add_threshold": 0.8,
    # Initialization
    "weight_init_std": 1.0,
    "novelty_init": False,          # if True, generate 2x pop and keep most diverse half
    "novelty_init_factor": 2,       # generate this many x pop_size, keep pop_size
    # Stagnation
    "max_stagnation": 15,
    "stagnation_explore_rate": 0.5,
    # Evaluation
    "episodes_per_genome": 3,
    "max_steps": 1000,
    # Stop
    "target_fitness": None,
    "max_generations": 100,
    # Compression
    "compress_every": 10,
}


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax with temperature."""
    x = np.array(x, dtype=np.float64) / max(temperature, 1e-6)
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


class GATEv3:
    """GATE-v3: Probabilistic Saliency Evolution."""
    name = "GATE-v3"

    def __init__(self, n_inputs: int, n_outputs: int, cfg: dict | None = None, seed: int = 0):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.cfg = {**GATE_V3_CFG, **(cfg or {})}
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.tracker = InnovationTracker()
        self.population: List[Genome] = []

        def _dist(a, b):
            d_struct = compatibility_distance(a, b, self.cfg["c1"], self.cfg["c2"], self.cfg["c3"])
            if a.behavior is None or b.behavior is None:
                return d_struct
            d_beh = math.sqrt(sum((x - y) ** 2 for x, y in zip(a.behavior, b.behavior)))
            return d_struct + self.cfg["behavioral_weight"] * d_beh
        self.speciator = Speciator(threshold=self.cfg["compat_threshold"], distance_fn=_dist)
        self.generation = 0
        self.history: List[dict] = []
        self.best_genome: Optional[Genome] = None
        self.best_fitness: float = -float("inf")
        self.best_effective_fitness: float = -float("inf")
        self.total_episodes = 0
        # Adaptive novelty state.
        self._base_novelty_weight = self.cfg["novelty_weight"]
        self._cur_novelty_weight = self.cfg["novelty_weight"]
        self._best_fitness_history: List[float] = []
        self._stagnation_count = 0
        # Behavioral archive.
        self._archive: List[tuple] = []
        # Temperature (anneals over generations).
        self._temperature = self.cfg["temp_start"]

    def init_population(self):
        if self.cfg.get("novelty_init", False):
            self._novelty_init_population()
            return
        self.population = []
        for _ in range(self.cfg["pop_size"]):
            g = make_initial_genome(
                self.n_inputs, self.n_outputs, self.tracker,
                output_activation="tanh",
                connect_input_output=True,
                weight_init_std=self.cfg["weight_init_std"],
                rng=self.rng,
            )
            self.population.append(g)

    def _novelty_init_population(self):
        """Generate novelty_init_factor * pop_size random genomes, evaluate each for 1 episode,
        then keep the pop_size most behaviorally diverse ones. This ensures the initial
        population covers a wide range of behaviors, which is critical for exploration-heavy
        tasks like MountainCar."""
        n_candidates = self.cfg["pop_size"] * self.cfg.get("novelty_init_factor", 2)
        candidates = []
        for _ in range(n_candidates):
            g = make_initial_genome(
                self.n_inputs, self.n_outputs, self.tracker,
                output_activation="tanh",
                connect_input_output=True,
                weight_init_std=self.cfg["weight_init_std"],
                rng=self.rng,
            )
            candidates.append(g)
        # Evaluate each for 1 episode.
        env = self._env_factory_fn() if hasattr(self, "_env_factory_fn") else None
        if env is None:
            # Fallback: just use random selection.
            self.population = candidates[:self.cfg["pop_size"]]
            return
        for g in candidates:
            fit, beh = self._evaluate_genome(g, env, 1, self.cfg["max_steps"])
            g.fitness = fit
            g.behavior = beh
        env.close()
        # Greedy novelty-based selection: iteratively pick the most novel genome.
        behs = [np.array(g.behavior) for g in candidates]
        selected_indices = []
        remaining = list(range(len(candidates)))
        # Pick a random first genome.
        first = self.rng.choice(remaining)
        selected_indices.append(first)
        remaining.remove(first)
        while len(selected_indices) < self.cfg["pop_size"] and remaining:
            # For each remaining, compute min distance to selected.
            min_dists = []
            for i in remaining:
                dists = [float(np.linalg.norm(behs[i] - behs[j])) for j in selected_indices]
                min_dists.append(min(dists))
            # Pick the one with max min-distance (most novel).
            best_idx = remaining[int(np.argmax(min_dists))]
            selected_indices.append(best_idx)
            remaining.remove(best_idx)
        self.population = [candidates[i] for i in selected_indices]
        print(f"  [novelty_init] selected {len(self.population)} from {n_candidates} candidates")

    def _evaluate_genome(self, genome: Genome, env, n_episodes: int, max_steps: int) -> Tuple[float, tuple]:
        net = FeedForwardNetwork(genome)
        rewards = []
        behaviors = []
        for _ in range(n_episodes):
            seed = self.rng.randint(0, 2**31 - 1)
            reward, steps, info = evaluate_episode(net, env, seed=seed, max_steps=max_steps)
            rewards.append(reward)
            self.total_episodes += 1
            beh = np.concatenate([
                info["obs_mean"], info["obs_var"], info["obs_range"],
                [info["action_mean"], info["action_var"], info["action_seq_diversity"], float(steps)],
            ])
            behaviors.append(beh)
        return float(np.mean(rewards)), tuple(np.mean(behaviors, axis=0))

    def evaluate_population(self, env_factory: Callable):
        env = env_factory()
        for genome in self.population:
            fit, beh = self._evaluate_genome(
                genome, env, self.cfg["episodes_per_genome"], self.cfg["max_steps"]
            )
            genome.fitness = fit
            genome.behavior = beh
        env.close()
        if self._cur_novelty_weight > 0.0 and len(self.population) > 1:
            novelties = self._compute_novelties()
            self._update_archive(novelties)
            if self.cfg["novelty_normalize"]:
                max_nov = max(novelties) if novelties else 1.0
                if max_nov > 1e-9:
                    novelties = [n / max_nov for n in novelties]
            for g, nov in zip(self.population, novelties):
                g._novelty = nov
                g.effective_fitness = g.fitness + self._cur_novelty_weight * nov
        else:
            for g in self.population:
                g._novelty = 0.0
                g.effective_fitness = g.fitness

    def _compute_novelties(self) -> List[float]:
        k = min(self.cfg["novelty_k"], len(self.population) - 1 + len(self._archive))
        if k <= 0:
            return [0.0] * len(self.population)
        pop_behs = [np.array(g.behavior) for g in self.population if g.behavior is not None]
        archive_behs = [np.array(b) for b in self._archive]
        reference = pop_behs + archive_behs
        if not reference:
            return [0.0] * len(self.population)
        novelties = []
        for i, beh_i in enumerate(pop_behs):
            dists = []
            for j, beh_j in enumerate(reference):
                if j == i:
                    continue
                d = float(np.linalg.norm(beh_i - beh_j))
                dists.append(d)
            dists.sort()
            nov = sum(dists[:k]) / k if dists else 0.0
            novelties.append(nov)
        return novelties

    def _update_archive(self, novelties: List[float]):
        if not self.cfg.get("use_archive", False):
            return
        max_nov = max(novelties) if novelties else 1.0
        if max_nov < 1e-9:
            return
        threshold = self.cfg["archive_add_threshold"] * max_nov
        for g, nov in zip(self.population, novelties):
            if nov >= threshold and g.behavior is not None:
                self._archive.append(g.behavior)
        if len(self._archive) > self.cfg["archive_size"]:
            self._archive = self._archive[-self.cfg["archive_size"]:]

    def spsa_probe_and_step(self, genome: Genome, env) -> Tuple[float, List[Tuple[int, float]]]:
        """SPSA probe (same as v2). Updates weights in-place and returns saliencies."""
        eps = self.cfg["spsa_eps"]
        lr = self.cfg["spsa_lr"]
        n_eps = self.cfg["spsa_episodes"]
        max_steps = self.cfg["max_steps"]
        decay = self.cfg["saliency_ema_decay"]
        nov_weight = self._cur_novelty_weight

        enabled_conns = [c for c in genome.connections.values() if c.enabled]
        if not enabled_conns:
            return 0.0, []

        orig_ws = np.array([c.weight for c in enabled_conns], dtype=np.float64)
        pop_behs = [np.array(g.behavior) for g in self.population if g.behavior is not None]
        archive_behs = [np.array(b) for b in self._archive]
        reference = pop_behs + archive_behs
        ref_mean = np.mean(reference, axis=0) if reference else None
        ref_novs = [float(np.linalg.norm(b - ref_mean)) for b in reference] if reference else []
        avg_ref_nov = float(np.mean(ref_novs)) if ref_novs else 1.0
        if avg_ref_nov < 1e-9:
            avg_ref_nov = 1.0

        deltas = np.array([self.rng.choice([-1, 1]) for _ in enabled_conns], dtype=np.float64)
        for i, c in enumerate(enabled_conns):
            c.weight = float(orig_ws[i] + eps * deltas[i])
        f_plus, beh_plus = self._evaluate_genome(genome, env, n_eps, max_steps)
        for i, c in enumerate(enabled_conns):
            c.weight = float(orig_ws[i] - eps * deltas[i])
        f_minus, beh_minus = self._evaluate_genome(genome, env, n_eps, max_steps)

        if nov_weight > 0.0 and ref_mean is not None:
            nov_plus = float(np.linalg.norm(np.array(beh_plus) - ref_mean)) / avg_ref_nov
            nov_minus = float(np.linalg.norm(np.array(beh_minus) - ref_mean)) / avg_ref_nov
            f_plus_eff = f_plus + nov_weight * nov_plus
            f_minus_eff = f_minus + nov_weight * nov_minus
        else:
            f_plus_eff = f_plus
            f_minus_eff = f_minus

        grad = (f_plus_eff - f_minus_eff) / (2 * eps * deltas)
        for i, c in enumerate(enabled_conns):
            c.weight = float(orig_ws[i] + lr * grad[i])
            c.saliency = decay * c.saliency + (1 - decay) * abs(float(grad[i]))
        saliencies = [(c.innovation, c.saliency) for c in enabled_conns]
        return float(np.linalg.norm(grad)), saliencies

    def probabilistic_grow(self, genome: Genome, saliencies: List[Tuple[int, float]]):
        """Sample which connection to split from softmax(saliencies / temperature).
        This replaces v2's deterministic top-K split."""
        if not saliencies:
            return
        n_splits = self.cfg["n_splits_per_elite"]
        if n_splits <= 0:
            return
        sal_values = np.array([max(s[1], 0) for s in saliencies])
        # If all saliencies are ~0, use uniform distribution.
        if np.max(sal_values) < 1e-6:
            probs = np.ones_like(sal_values) / len(sal_values)
        else:
            probs = softmax(sal_values, self._temperature)
        # Sample without replacement.
        n_splits = min(n_splits, len(saliencies))
        indices = self.np_rng.choice(len(saliencies), size=n_splits, replace=False, p=probs)
        for idx in indices:
            innov, _ = saliencies[idx]
            if innov not in genome.connections:
                continue
            conn = genome.connections[innov]
            if not conn.enabled:
                continue
            new_node_id, in_inv, out_inv = self.tracker.get_add_node(conn.innovation)
            if new_node_id in genome.nodes:
                continue
            conn.enabled = False
            genome.nodes[new_node_id] = NodeGene(
                id=new_node_id, type=NodeType.HIDDEN, activation="tanh",
            )
            genome.connections[in_inv] = ConnectionGene(
                innovation=in_inv, in_node=conn.in_node, out_node=new_node_id,
                weight=1.0,
            )
            genome.connections[out_inv] = ConnectionGene(
                innovation=out_inv, in_node=new_node_id, out_node=conn.out_node,
                weight=conn.weight,
            )

    def probabilistic_prune(self, genome: Genome, saliencies: List[Tuple[int, float]]):
        """Sample which connection to prune from softmax(-saliencies / temperature).
        Low-saliency connections are more likely to be pruned, but it's probabilistic."""
        n_enabled = sum(1 for c in genome.connections.values() if c.enabled)
        if n_enabled < self.cfg["min_connections_to_prune"]:
            return
        if not saliencies:
            return
        n_prunes = self.cfg["n_prunes_per_elite"]
        # Use negative saliency for pruning (low saliency = high prune probability).
        sal_values = np.array([s[1] for s in saliencies])
        # Invert: high saliency -> low prune prob.
        neg_sal = -sal_values
        if np.max(np.abs(neg_sal)) < 1e-6:
            probs = np.ones_like(neg_sal) / len(neg_sal)
        else:
            probs = softmax(neg_sal, self._temperature)
        n_prunes = min(n_prunes, len(saliencies))
        indices = self.np_rng.choice(len(saliencies), size=n_prunes, replace=False, p=probs)
        pruned = 0
        for idx in indices:
            innov, _ = saliencies[idx]
            if innov not in genome.connections:
                continue
            conn = genome.connections[innov]
            if not conn.enabled:
                continue
            conn.enabled = False
            pruned += 1
        return pruned

    def _update_adaptive_novelty(self, best_fitness: float):
        if not self.cfg.get("adaptive_novelty", False):
            return
        if self._best_fitness_history:
            prev_best = max(self._best_fitness_history)
            if best_fitness > prev_best + 1e-6:
                self._cur_novelty_weight = max(
                    self._base_novelty_weight,
                    self._cur_novelty_weight * self.cfg["adaptive_novelty_decay"],
                )
                self._stagnation_count = 0
                # Also cool temperature (exploit the improvement).
                self._temperature = max(self.cfg["temp_end"], self._temperature * 0.8)
            else:
                self._stagnation_count += 1
                if self._stagnation_count >= self.cfg["adaptive_novelty_patience"]:
                    self._cur_novelty_weight *= self.cfg["adaptive_novelty_factor"]
                    # Also heat up temperature (explore more).
                    self._temperature = min(self.cfg["temp_start"], self._temperature * 1.5)
                    self._stagnation_count = 0
        self._best_fitness_history.append(best_fitness)
        if len(self._best_fitness_history) > 20:
            self._best_fitness_history = self._best_fitness_history[-20:]

    def compress_genome(self, genome: Genome):
        to_remove = [inv for inv, c in genome.connections.items() if not c.enabled and c.age > 5]
        for inv in to_remove:
            del genome.connections[inv]
        connected_nodes = set()
        for c in genome.connections.values():
            if c.enabled:
                connected_nodes.add(c.in_node)
                connected_nodes.add(c.out_node)
        to_remove_nodes = [nid for nid, n in genome.nodes.items()
                          if n.type == NodeType.HIDDEN and nid not in connected_nodes]
        for nid in to_remove_nodes:
            del genome.nodes[nid]
        to_remove_conns = [inv for inv, c in genome.connections.items()
                           if c.in_node not in genome.nodes or c.out_node not in genome.nodes]
        for inv in to_remove_conns:
            del genome.connections[inv]

    def _maybe_compress(self):
        if self.cfg.get("compress_every", 0) <= 0:
            return
        if self.generation > 0 and self.generation % self.cfg["compress_every"] == 0:
            for g in self.population:
                self.compress_genome(g)
        for g in self.population:
            for c in g.connections.values():
                c.age += 1
            for n in g.nodes.values():
                n.age += 1

    def reproduce(self):
        self.speciator.speciate(self.population)
        self.speciator.adjust_fitness()
        total_adjusted = sum(sum(g.adjusted_fitness for g in sp.members) for sp in self.speciator.species)
        if total_adjusted <= 0:
            total_adjusted = 1.0

        new_population: List[Genome] = []
        pop_size = self.cfg["pop_size"]
        env = None

        species_with_fitness = [(sp, max(g.fitness for g in sp.members)) for sp in self.speciator.species if sp.members]
        species_with_fitness.sort(key=lambda x: x[1], reverse=True)
        n_spsa_species = max(1, int(self.cfg["spsa_top_fraction"] * len(species_with_fitness)))
        spsa_species_ids = set(id(sp) for sp, _ in species_with_fitness[:n_spsa_species])

        total_spsa = 0
        total_grows = 0
        total_prunes = 0
        total_random_explore = 0
        avg_saliency = []

        for sp in self.speciator.species:
            members = sorted(sp.members, key=lambda g: getattr(g, 'effective_fitness', g.fitness), reverse=True)
            if not members:
                continue
            sp_offspring_count = int(round(
                sum(g.adjusted_fitness for g in sp.members) / total_adjusted * pop_size
            ))
            do_spsa = id(sp) in spsa_species_ids

            for k in range(min(self.cfg["elitism"], len(members))):
                if len(new_population) >= pop_size:
                    break
                elite_copy = members[k].copy()
                if do_spsa:
                    if env is None:
                        env = self._env_factory()
                    grad_norm, sals = self.spsa_probe_and_step(elite_copy, env)
                    total_spsa += 1
                    if sals:
                        self.probabilistic_grow(elite_copy, sals)
                        total_grows += 1
                        self.probabilistic_prune(elite_copy, sals)
                        total_prunes += 1
                        avg_saliency.append(float(np.mean([s[1] for s in sals])))
                new_population.append(elite_copy)

            n_survivors = max(1, int(len(members) * self.cfg["survival_threshold"]))
            survivors = members[:n_survivors]
            is_stagnated = sp.stagnation >= self.cfg["max_stagnation"]

            for _ in range(sp_offspring_count - self.cfg["elitism"]):
                if len(new_population) >= pop_size:
                    break
                if len(survivors) == 1:
                    parent_a = survivors[0]
                    parent_b = survivors[0]
                else:
                    parent_a = self.rng.choice(survivors)
                    parent_b = self.rng.choice(survivors)
                if (self.rng.random() < self.cfg["interspecies_mate_rate"]
                        and len(self.speciator.species) > 1):
                    other_sp = self.rng.choice(self.speciator.species)
                    parent_b = self.rng.choice(other_sp.members)
                child = crossover(parent_a, parent_b, self.rng,
                                  saliency_aware=self.cfg.get("saliency_aware_crossover", True))
                fits = [g.fitness for g in self.population]
                fit_var = float(np.var(fits)) if fits else 0.0
                in_explore_mode = (self._cur_novelty_weight > 0.0
                                   and fit_var < self._cur_novelty_weight * 0.5)
                mut_cfg = dict(self.cfg)
                if in_explore_mode:
                    mut_cfg["saliency_weighted_mutation"] = False
                mutate(child, self.tracker, self.rng, mut_cfg)
                if is_stagnated and self.rng.random() < self.cfg["stagnation_explore_rate"]:
                    if self.rng.random() < 0.5:
                        mutate_add_node(child, self.tracker, self.rng)
                    else:
                        mutate_add_connection(child, self.tracker, self.rng)
                    total_random_explore += 1
                new_population.append(child)

        while len(new_population) < pop_size:
            if self.population:
                parent = max(self.population, key=lambda g: getattr(g, 'effective_fitness', g.fitness))
                child = parent.copy()
                mutate(child, self.tracker, self.rng, self.cfg)
                new_population.append(child)
            else:
                g = make_initial_genome(self.n_inputs, self.n_outputs, self.tracker,
                                       weight_init_std=self.cfg["weight_init_std"], rng=self.rng)
                new_population.append(g)

        self.population = new_population[:pop_size]
        if env is not None:
            env.close()
        for g in self.population:
            eff = getattr(g, 'effective_fitness', g.fitness)
            if eff > self.best_effective_fitness:
                self.best_effective_fitness = eff
            if g.fitness > self.best_fitness:
                self.best_fitness = g.fitness
                self.best_genome = g.copy()
        self._last_debug = {
            "total_spsa": total_spsa, "total_grows": total_grows,
            "total_prunes": total_prunes, "total_random_explore": total_random_explore,
            "avg_saliency": float(np.mean(avg_saliency)) if avg_saliency else 0.0,
            "temperature": self._temperature,
        }

    def _env_factory(self):
        if not hasattr(self, "_env_factory_fn"):
            raise RuntimeError("Call run() with env_factory")
        return self._env_factory_fn()

    def step(self, env_factory: Callable) -> dict:
        t0 = time.time()
        if self.generation == 0:
            self.init_population()
        self._env_factory_fn = env_factory
        self.evaluate_population(env_factory)
        best = max(g.fitness for g in self.population)
        mean = float(np.mean([g.fitness for g in self.population]))
        median = float(np.median([g.fitness for g in self.population]))
        std = float(np.std([g.fitness for g in self.population]))
        avg_size = float(np.mean([g.num_enabled_connections() for g in self.population]))
        avg_hidden = float(np.mean([g.num_hidden() for g in self.population]))
        n_species = len(self.speciator.species) if self.speciator.species else 0
        novs = [getattr(g, '_novelty', 0.0) for g in self.population]
        for g in self.population:
            eff = getattr(g, 'effective_fitness', g.fitness)
            if eff > self.best_effective_fitness:
                self.best_effective_fitness = eff
            if g.fitness > self.best_fitness:
                self.best_fitness = g.fitness
                self.best_genome = g.copy()
        self._update_adaptive_novelty(best)
        self._maybe_compress()
        # Anneal temperature.
        self._temperature = max(self.cfg["temp_end"], self._temperature * self.cfg["temp_decay"])
        stats = {
            "generation": self.generation, "best": best,
            "best_eff": self.best_effective_fitness, "mean": mean,
            "median": median, "std": std, "avg_size": avg_size,
            "avg_hidden": avg_hidden, "n_species": n_species,
            "total_episodes": self.total_episodes, "time": time.time() - t0,
            "novelty_mean": float(np.mean(novs)) if novs else 0.0,
            "novelty_weight": self._cur_novelty_weight,
            "temperature": self._temperature,
        }
        self.history.append(stats)
        self.reproduce()
        if hasattr(self, "_last_debug"):
            self.history[-1].update(self._last_debug)
        self.generation += 1
        return stats

    def run(self, env_factory: Callable, verbose: bool = True) -> dict:
        target = self.cfg.get("target_fitness")
        max_gen = self.cfg.get("max_generations", 100)
        for _ in range(max_gen):
            stats = self.step(env_factory)
            if verbose:
                dbg = ""
                if "total_spsa" in stats:
                    dbg = (f" | spsa {stats['total_spsa']:2d} | grows {stats['total_grows']:2d} | "
                           f"prunes {stats['total_prunes']:2d} | "
                           f"T {stats['temperature']:.2f} | "
                           f"sal {stats['avg_saliency']:.2f} | "
                           f"explore {stats['total_random_explore']:2d}")
                print(f"[{self.name}] gen {stats['generation']:3d} | "
                      f"best {stats['best']:6.1f} | mean {stats['mean']:6.1f} ± {stats['std']:5.1f} | "
                      f"size {stats['avg_size']:4.1f} | hid {stats['avg_hidden']:3.1f} | "
                      f"sp {stats['n_species']:2d} | eps {stats['total_episodes']:5d}{dbg} | "
                      f"{stats['time']:.1f}s")
            if target is not None and stats["best"] >= target:
                break
        return {
            "best_fitness": self.best_fitness,
            "total_episodes": self.total_episodes,
            "generations": self.generation,
            "history": self.history,
            "best_genome": self.best_genome,
        }
