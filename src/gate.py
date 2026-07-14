"""
GATE v2: Gradient-Anchored Topology Evolution (refined)
=======================================================

Core insight: a single SPSA (Simultaneous Perturbation Stochastic Approximation) probe
gives us BOTH a weight-update direction AND a topology-growth signal.

SPSA probe:
  - Sample random ±1 perturbation vector Δ (one bit per enabled weight).
  - Evaluate f(w + ε·Δ) and f(w - ε·Δ).
  - Estimate per-weight gradient: g_i ≈ (f+ - f-) / (2·ε·Δ_i).
  - Cost: 2 episodes per probe (regardless of network size!).

From this single probe we derive:
  - Weight update: w_i ← w_i + lr · g_i  (gradient ascent on fitness).
  - Saliency: s_i ← EMA(|g_i|). The connections with highest EMA saliency are the ones
    where fitness is most sensitive to weight changes - they are the network's bottlenecks,
    the places where adding capacity (a new hidden node) is most likely to help.

This is the unifying principle of GATE: the SAME signal that updates weights also directs
topology growth. There is no separate "saliency probing" pass - it's all one mechanism.

Other modernizations over NEAT:
  - Behavioral speciation: species are defined by behavioral signature (trajectory stats)
    + structural distance, not just structural distance. This protects behavioral diversity.
  - Directed pruning: connections with low saliency AND low weight are pruned (disabled).
  - Adaptive exploration: when a species stagnates, fall back to random structural mutations.
  - No per-connection probing (which was the bottleneck in v1). SPSA is O(1) per elite.
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


GATE_CFG = {
    "pop_size": 80,
    # SPSA probing (the core mechanism)
    "spsa_eps": 0.3,                # perturbation magnitude
    "spsa_lr": 0.15,                # weight step size
    "spsa_episodes": 2,             # episodes per SPSA evaluation (f+ and f-)
    "spsa_n_probes": 1,             # number of SPSA probes to average (more = less noise)
    "saliency_ema_decay": 0.5,      # EMA decay for |g_i| (smaller = faster adaptation)
    # Topology growth (driven by saliency)
    "saliency_top_k": 1,            # split this many top-saliency connections per elite per gen
    "saliency_prune_frac": 0.10,    # disable this fraction of low-saliency+low-weight connections
    "saliency_min_weight": 0.5,     # only prune if |weight| < this
    "min_connections_to_prune": 8,  # don't prune if network is too small
    # Speciation (behavioral + structural)
    "compat_threshold": 3.0,
    "behavioral_weight": 0.3,
    "c1": 1.0, "c2": 1.0, "c3": 0.4,
    # Reproduction
    "elitism": 1,
    "survival_threshold": 0.20,
    "interspecies_mate_rate": 0.001,
    # Mutation rates (fallback / diversity)
    "weight_mut_rate": 0.85,
    "weight_perturb_std": 0.25,
    "weight_replace_rate": 0.1,
    "add_conn_rate": 0.3,
    "add_node_rate": 0.10,
    "remove_node_rate": 0.0,
    "toggle_enable_rate": 0.0,
    "act_mut_rate": 0.0,
    # Adaptive exploration
    "max_stagnation": 15,
    "stagnation_explore_rate": 0.5,
    # Evaluation
    "episodes_per_genome": 3,
    "max_steps": 1000,
    # Stop
    "target_fitness": None,
    "max_generations": 100,
    # Apply SPSA + directed growth only to top elites (saves compute)
    "spsa_top_fraction": 0.25,      # apply SPSA only to top 25% of species (by best fitness)
    # Novelty-weighted fitness (for exploration in deceptive tasks)
    "novelty_weight": 0.0,          # weight on novelty bonus in effective fitness
    "novelty_k": 5,                 # k-nearest-neighbors for novelty computation
    "novelty_normalize": True,      # normalize novelty to [0, 1] range
    # Initialization
    "weight_init_std": 1.0,         # std of initial connection weights (larger = more diverse)
    # Dual SPSA: when fitness is flat, also probe novelty gradient
    "dual_spsa": True,              # if True, run novelty SPSA when fitness is uninformative
    "dual_spsa_fitness_threshold": 1e-3,  # if fitness variance across pop < this, use novelty SPSA
    # Saliency-weighted mutation (GATE principle applied to weight perturbation)
    "saliency_weighted_mutation": True,   # scale mutation magnitude by inverse saliency
    "saliency_mutation_scale": 2.0,       # max scale factor for low-saliency connections
    # Adaptive novelty weight: increase when global fitness stagnates, decrease when improving
    "adaptive_novelty": True,             # enable adaptive novelty weight
    "adaptive_novelty_patience": 5,       # gens without improvement before increasing novelty
    "adaptive_novelty_factor": 1.5,       # multiply novelty_weight by this when stagnating
    "adaptive_novelty_decay": 0.7,        # multiply novelty_weight by this when improving (toward base)
}


class GATE:
    """Gradient-Anchored Topology Evolution v2."""
    name = "GATE-v2"

    def __init__(self, n_inputs: int, n_outputs: int, cfg: dict | None = None, seed: int = 0):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.cfg = {**GATE_CFG, **(cfg or {})}
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
        self.best_fitness: float = -float("inf")  # tracks RAW fitness for stopping criterion
        self.best_effective_fitness: float = -float("inf")  # tracks effective fitness for selection
        self.total_episodes = 0
        # Adaptive novelty state.
        self._base_novelty_weight = self.cfg["novelty_weight"]
        self._cur_novelty_weight = self.cfg["novelty_weight"]
        self._best_fitness_history: List[float] = []
        self._stagnation_count = 0

    def init_population(self):
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

    def _evaluate_genome(self, genome: Genome, env, n_episodes: int, max_steps: int) -> Tuple[float, tuple]:
        net = FeedForwardNetwork(genome)
        rewards = []
        behaviors = []
        for _ in range(n_episodes):
            seed = self.rng.randint(0, 2**31 - 1)
            reward, steps, info = evaluate_episode(net, env, seed=seed, max_steps=max_steps)
            rewards.append(reward)
            self.total_episodes += 1
            # Richer behavioral signature: obs_mean, obs_var, obs_range, action stats, action seq diversity, steps.
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
        # Compute novelty bonus (if enabled).
        if self.cfg["novelty_weight"] > 0.0 and len(self.population) > 1:
            novelties = self._compute_novelties()
            # Normalize novelties to [0, 1] for stable weighting across envs.
            if self.cfg["novelty_normalize"]:
                max_nov = max(novelties) if novelties else 1.0
                if max_nov > 1e-9:
                    novelties = [n / max_nov for n in novelties]
            # Blend novelty into fitness.
            for g, nov in zip(self.population, novelties):
                g._novelty = nov
                # Don't overwrite raw fitness; use effective fitness in selection.
                g.effective_fitness = g.fitness + self.cfg["novelty_weight"] * nov
        else:
            for g in self.population:
                g._novelty = 0.0
                g.effective_fitness = g.fitness

    def _compute_novelties(self) -> List[float]:
        """For each genome, compute average behavioral distance to k nearest neighbors.
        This is the standard novelty-search criterion, integrated with GATE's behavioral
        characterization (no separate archive needed - the population IS the archive)."""
        k = min(self.cfg["novelty_k"], len(self.population) - 1)
        if k <= 0:
            return [0.0] * len(self.population)
        novelties = []
        behs = [np.array(g.behavior) for g in self.population]
        for i, beh_i in enumerate(behs):
            dists = []
            for j, beh_j in enumerate(behs):
                if i == j:
                    continue
                d = float(np.linalg.norm(beh_i - beh_j))
                dists.append(d)
            dists.sort()
            nov = sum(dists[:k]) / k if dists else 0.0
            novelties.append(nov)
        return novelties

    def spsa_probe_and_step(self, genome: Genome, env) -> Tuple[float, List[Tuple[int, float]]]:
        """Run SPSA probe(s) on genome. Updates weights in-place and returns
        (gradient_norm, [(innovation, |g_i|), ...]) for saliency tracking.

        If spsa_n_probes > 1, runs multiple probes with independent random Δ vectors
        and averages the gradient estimates. This reduces noise at the cost of more
        evaluations. Averaging is the standard variance-reduction technique for SPSA.

        The EMA saliency on each connection is updated in-place.

        The SPSA objective is the EFFECTIVE fitness = fitness + novelty_weight * novelty.
        """
        eps = self.cfg["spsa_eps"]
        lr = self.cfg["spsa_lr"]
        n_eps = self.cfg["spsa_episodes"]
        max_steps = self.cfg["max_steps"]
        decay = self.cfg["saliency_ema_decay"]
        nov_weight = self.cfg["novelty_weight"]
        n_probes = max(1, self.cfg.get("spsa_n_probes", 1))

        enabled_conns = [c for c in genome.connections.values() if c.enabled]
        if not enabled_conns:
            return 0.0, []

        orig_ws = np.array([c.weight for c in enabled_conns], dtype=np.float64)
        # Pre-compute population behaviors for novelty computation.
        pop_behs = [np.array(g.behavior) for g in self.population if g.behavior is not None]
        pop_mean = np.mean(pop_behs, axis=0) if pop_behs else None
        pop_novs = [float(np.linalg.norm(b - pop_mean)) for b in pop_behs] if pop_behs else []
        avg_pop_nov = float(np.mean(pop_novs)) if pop_novs else 1.0
        if avg_pop_nov < 1e-9:
            avg_pop_nov = 1.0

        # Accumulate gradient over multiple probes.
        grad_accum = np.zeros_like(orig_ws)
        for _ in range(n_probes):
            deltas = np.array([self.rng.choice([-1, 1]) for _ in enabled_conns], dtype=np.float64)
            # f(w + c*Δ)
            for i, c in enumerate(enabled_conns):
                c.weight = float(orig_ws[i] + eps * deltas[i])
            f_plus, beh_plus = self._evaluate_genome(genome, env, n_eps, max_steps)
            # f(w - c*Δ)
            for i, c in enumerate(enabled_conns):
                c.weight = float(orig_ws[i] - eps * deltas[i])
            f_minus, beh_minus = self._evaluate_genome(genome, env, n_eps, max_steps)
            # Effective fitness (fitness + novelty bonus, normalized).
            if nov_weight > 0.0 and pop_mean is not None:
                nov_plus = float(np.linalg.norm(np.array(beh_plus) - pop_mean)) / avg_pop_nov
                nov_minus = float(np.linalg.norm(np.array(beh_minus) - pop_mean)) / avg_pop_nov
                f_plus_eff = f_plus + nov_weight * nov_plus
                f_minus_eff = f_minus + nov_weight * nov_minus
            else:
                f_plus_eff = f_plus
                f_minus_eff = f_minus
            grad = (f_plus_eff - f_minus_eff) / (2 * eps * deltas)
            grad_accum += grad

        grad_avg = grad_accum / n_probes
        # Update weights (gradient ascent on effective fitness)
        for i, c in enumerate(enabled_conns):
            c.weight = float(orig_ws[i] + lr * grad_avg[i])
            c.saliency = decay * c.saliency + (1 - decay) * abs(float(grad_avg[i]))

        saliencies = [(c.innovation, c.saliency) for c in enabled_conns]
        return float(np.linalg.norm(grad_avg)), saliencies

    def saliency_directed_grow(self, genome: Genome, saliencies: List[Tuple[int, float]]):
        """Split the top-K connections by EMA saliency."""
        if not saliencies:
            return
        k = max(1, self.cfg["saliency_top_k"])
        sorted_sals = sorted(saliencies, key=lambda x: x[1], reverse=True)
        # Filter out saliencies that are essentially zero (no signal yet).
        top_k = [s for s in sorted_sals if s[1] > 1e-6][:k]
        for innov, _ in top_k:
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

    def saliency_directed_prune(self, genome: Genome, saliencies: List[Tuple[int, float]]):
        """Disable low-saliency + low-weight connections."""
        n_enabled = sum(1 for c in genome.connections.values() if c.enabled)
        if n_enabled < self.cfg["min_connections_to_prune"]:
            return
        if not saliencies:
            return
        frac = self.cfg["saliency_prune_frac"]
        sorted_sals = sorted(saliencies, key=lambda x: x[1])  # ascending
        n_to_prune = max(1, int(frac * len(sorted_sals)))
        min_w = self.cfg["saliency_min_weight"]
        pruned = 0
        for innov, sal in sorted_sals[:n_to_prune]:
            if innov not in genome.connections:
                continue
            conn = genome.connections[innov]
            if not conn.enabled:
                continue
            if abs(conn.weight) < min_w:
                conn.enabled = False
                pruned += 1
        return pruned

    def reproduce(self):
        self.speciator.speciate(self.population)
        self.speciator.adjust_fitness()
        total_adjusted = sum(sum(g.adjusted_fitness for g in sp.members) for sp in self.speciator.species)
        if total_adjusted <= 0:
            total_adjusted = 1.0

        new_population: List[Genome] = []
        pop_size = self.cfg["pop_size"]
        env = None

        # Decide which species get SPSA probing (top fraction by best fitness).
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

            # Elitism: keep top K unchanged, but apply SPSA + directed growth to the elite.
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
                        self.saliency_directed_grow(elite_copy, sals)
                        total_grows += 1
                        self.saliency_directed_prune(elite_copy, sals)
                        total_prunes += 1
                        avg_saliency.append(float(np.mean([s[1] for s in sals])))
                new_population.append(elite_copy)

            # Survivors for crossover.
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
                child = crossover(parent_a, parent_b, self.rng)
                # Determine if we're in exploration mode (fitness contributes little to effective fitness).
                # This is true when novelty_weight > 0 and fitness variance is small relative to novelty scale.
                fits = [g.fitness for g in self.population]
                fit_var = float(np.var(fits)) if fits else 0.0
                in_explore_mode = (self.cfg["novelty_weight"] > 0.0
                                   and fit_var < self.cfg["novelty_weight"] * 0.5)
                # Disable saliency-weighted mutation in explore mode (saliency is novelty-based, not fitness-based).
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

        # Fill short population.
        while len(new_population) < pop_size:
            if self.population:
                parent = max(self.population, key=lambda g: getattr(g, 'effective_fitness', g.fitness))
                child = parent.copy()
                mutate(child, self.tracker, self.rng, self.cfg)
                new_population.append(child)
            else:
                g = make_initial_genome(self.n_inputs, self.n_outputs, self.tracker, rng=self.rng)
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
            "total_spsa": total_spsa,
            "total_grows": total_grows,
            "total_prunes": total_prunes,
            "total_random_explore": total_random_explore,
            "avg_saliency": float(np.mean(avg_saliency)) if avg_saliency else 0.0,
        }

    def _env_factory(self):
        if not hasattr(self, "_env_factory_fn"):
            raise RuntimeError("Call run() with env_factory")
        return self._env_factory_fn()

    def _update_adaptive_novelty(self, best_fitness: float):
        """Adjust novelty_weight based on whether best fitness is improving."""
        if not self.cfg.get("adaptive_novelty", False):
            return
        if self._best_fitness_history:
            prev_best = max(self._best_fitness_history)
            if best_fitness > prev_best + 1e-6:
                # Improvement: decay novelty weight toward base.
                self._cur_novelty_weight = max(
                    self._base_novelty_weight,
                    self._cur_novelty_weight * self.cfg["adaptive_novelty_decay"],
                )
                self._stagnation_count = 0
            else:
                self._stagnation_count += 1
                if self._stagnation_count >= self.cfg["adaptive_novelty_patience"]:
                    self._cur_novelty_weight *= self.cfg["adaptive_novelty_factor"]
                    self._stagnation_count = 0
        self._best_fitness_history.append(best_fitness)
        # Keep last 20 generations.
        if len(self._best_fitness_history) > 20:
            self._best_fitness_history = self._best_fitness_history[-20:]
        # Update the config value so SPSA uses the new weight.
        self.cfg["novelty_weight"] = self._cur_novelty_weight

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
        # Update adaptive novelty weight based on improvement.
        self._update_adaptive_novelty(best)
        stats = {
            "generation": self.generation,
            "best": best,
            "best_eff": self.best_effective_fitness,
            "mean": mean,
            "median": median,
            "std": std,
            "avg_size": avg_size,
            "avg_hidden": avg_hidden,
            "n_species": n_species,
            "total_episodes": self.total_episodes,
            "time": time.time() - t0,
            "novelty_mean": float(np.mean(novs)) if novs else 0.0,
            "novelty_weight": self._cur_novelty_weight,
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
