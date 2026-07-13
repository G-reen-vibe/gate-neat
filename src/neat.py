"""
The canonical NEAT algorithm. This is our baseline.

Reference: Stanley & Miikkulainen (2002), "Evolving Neural Networks through
Augmenting Topologies".
"""
from __future__ import annotations
import random
import time
from typing import Callable, List, Optional, Tuple
import numpy as np

from .genome import Genome, InnovationTracker, make_initial_genome
from .network import FeedForwardNetwork, evaluate_episode
from .mutations import mutate
from .speciation import Speciator, Species, crossover, compatibility_distance


DEFAULT_CFG = {
    "pop_size": 80,
    # Mutation rates
    "weight_mut_rate": 0.85,
    "weight_perturb_std": 0.25,
    "weight_replace_rate": 0.1,
    "add_conn_rate": 0.5,
    "add_node_rate": 0.25,
    "remove_node_rate": 0.0,
    "toggle_enable_rate": 0.0,
    "act_mut_rate": 0.0,
    # Speciation
    "compat_threshold": 3.0,
    "c1": 1.0, "c2": 1.0, "c3": 0.4,
    # Reproduction
    "elitism": 1,
    "survival_threshold": 0.20,
    "interspecies_mate_rate": 0.001,
    # Stagnation
    "max_stagnation": 15,
    # Evaluation
    "episodes_per_genome": 3,
    "max_steps": 1000,
    # Stop
    "target_fitness": None,  # None means run for max_generations
    "max_generations": 100,
}


class NEAT:
    """Reference NEAT implementation."""
    name = "NEAT"

    def __init__(self, n_inputs: int, n_outputs: int, cfg: dict | None = None, seed: int = 0):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.cfg = {**DEFAULT_CFG, **(cfg or {})}
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.tracker = InnovationTracker()
        self.population: List[Genome] = []
        self.speciator = Speciator(
            threshold=self.cfg["compat_threshold"],
            distance_fn=lambda a, b: compatibility_distance(a, b, self.cfg["c1"], self.cfg["c2"], self.cfg["c3"]),
        )
        self.generation = 0
        self.history: List[dict] = []
        self.best_genome: Optional[Genome] = None
        self.best_fitness: float = -float("inf")
        self.total_episodes = 0  # cumulative environment interactions

    def init_population(self):
        self.population = []
        for _ in range(self.cfg["pop_size"]):
            g = make_initial_genome(
                self.n_inputs, self.n_outputs, self.tracker,
                output_activation="tanh",
                connect_input_output=True,
                rng=self.rng,
            )
            self.population.append(g)

    def evaluate_population(self, env_factory: Callable):
        """Evaluate every genome on the environment. Multiple episodes for stability."""
        env = env_factory()
        for i, genome in enumerate(self.population):
            net = FeedForwardNetwork(genome)
            rewards = []
            behaviors = []
            for ep in range(self.cfg["episodes_per_genome"]):
                seed = self.rng.randint(0, 2**31 - 1)
                reward, steps, info = evaluate_episode(
                    net, env, seed=seed, max_steps=self.cfg["max_steps"],
                )
                rewards.append(reward)
                self.total_episodes += 1
                # Behavioral signature: mean obs, var obs, action stats.
                beh = np.concatenate([
                    info["obs_mean"], info["obs_var"],
                    [info["action_mean"], info["action_var"], float(steps)],
                ])
                behaviors.append(beh)
            genome.fitness = float(np.mean(rewards))
            # Average behavior across episodes.
            genome.behavior = tuple(np.mean(behaviors, axis=0))
        env.close()

    def reproduce(self):
        """Produce next generation from current population via speciation, crossover, mutation."""
        # 1. Speciate.
        self.speciator.speciate(self.population)
        self.speciator.adjust_fitness()
        # 2. Sort each species by adjusted fitness, compute offspring counts.
        total_adjusted = sum(sum(g.adjusted_fitness for g in sp.members) for sp in self.speciator.species)
        if total_adjusted <= 0:
            total_adjusted = 1.0
        new_population: List[Genome] = []
        pop_size = self.cfg["pop_size"]

        for sp in self.speciator.species:
            # Sort members by fitness descending.
            members = sorted(sp.members, key=lambda g: g.fitness, reverse=True)
            if not members:
                continue
            sp_offspring_count = int(round(
                sum(g.adjusted_fitness for g in sp.members) / total_adjusted * pop_size
            ))
            # Elitism: keep top K unchanged.
            for k in range(min(self.cfg["elitism"], len(members))):
                if len(new_population) >= pop_size:
                    break
                new_population.append(members[k].copy())
            # Determine survivors (parents for crossover).
            n_survivors = max(1, int(len(members) * self.cfg["survival_threshold"]))
            survivors = members[:n_survivors]
            # Fill rest with offspring.
            for _ in range(sp_offspring_count - self.cfg["elitism"]):
                if len(new_population) >= pop_size:
                    break
                if len(survivors) == 1:
                    parent_a = survivors[0]
                    parent_b = survivors[0]
                else:
                    parent_a = self.rng.choice(survivors)
                    parent_b = self.rng.choice(survivors)
                # Interspecies mating.
                if (self.rng.random() < self.cfg["interspecies_mate_rate"]
                        and len(self.speciator.species) > 1):
                    other_sp = self.rng.choice(self.speciator.species)
                    parent_b = self.rng.choice(other_sp.members)
                child = crossover(parent_a, parent_b, self.rng)
                mutate(child, self.tracker, self.rng, self.cfg)
                new_population.append(child)

        # If population is short (rounding), fill with random mutants of the best.
        while len(new_population) < pop_size:
            if self.population:
                parent = max(self.population, key=lambda g: g.fitness)
                child = parent.copy()
                mutate(child, self.tracker, self.rng, self.cfg)
                new_population.append(child)
            else:
                # Should not happen.
                g = make_initial_genome(self.n_inputs, self.n_outputs, self.tracker, rng=self.rng)
                new_population.append(g)

        # Truncate to pop_size if we overshoot.
        self.population = new_population[:pop_size]
        # Update best genome.
        for g in self.population:
            if g.fitness > self.best_fitness:
                self.best_fitness = g.fitness
                self.best_genome = g.copy()

    def step(self, env_factory: Callable) -> dict:
        """Run one generation: evaluate, log, reproduce."""
        t0 = time.time()
        if self.generation == 0:
            self.init_population()
        self.evaluate_population(env_factory)
        # Track stats before reproduction (population is current gen).
        best = max(g.fitness for g in self.population)
        mean = float(np.mean([g.fitness for g in self.population]))
        median = float(np.median([g.fitness for g in self.population]))
        std = float(np.std([g.fitness for g in self.population]))
        avg_size = float(np.mean([g.num_enabled_connections() for g in self.population]))
        avg_hidden = float(np.mean([g.num_hidden() for g in self.population]))
        n_species = len(self.speciator.species) if self.speciator.species else 0
        # Update best.
        for g in self.population:
            if g.fitness > self.best_fitness:
                self.best_fitness = g.fitness
                self.best_genome = g.copy()
        stats = {
            "generation": self.generation,
            "best": best,
            "mean": mean,
            "median": median,
            "std": std,
            "avg_size": avg_size,
            "avg_hidden": avg_hidden,
            "n_species": n_species,
            "total_episodes": self.total_episodes,
            "time": time.time() - t0,
        }
        self.history.append(stats)
        # Reproduce for next generation.
        self.reproduce()
        self.generation += 1
        return stats

    def run(self, env_factory: Callable, verbose: bool = True) -> dict:
        """Run NEAT until target fitness or max generations."""
        target = self.cfg.get("target_fitness")
        max_gen = self.cfg.get("max_generations", 100)
        for _ in range(max_gen):
            stats = self.step(env_factory)
            if verbose:
                print(f"[{self.name}] gen {stats['generation']:3d} | "
                      f"best {stats['best']:6.1f} | mean {stats['mean']:6.1f} ± {stats['std']:5.1f} | "
                      f"size {stats['avg_size']:4.1f} | sp {stats['n_species']:2d} | "
                      f"eps {stats['total_episodes']:5d} | {stats['time']:.1f}s")
            if target is not None and stats["best"] >= target:
                break
        return {
            "best_fitness": self.best_fitness,
            "total_episodes": self.total_episodes,
            "generations": self.generation,
            "history": self.history,
            "best_genome": self.best_genome,
        }
