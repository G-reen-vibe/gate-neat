"""
Comprehensive benchmark: NEAT vs GATE vs Random Search on CartPole-v1, MountainCar-v0, Acrobot-v1.

Runs multiple seeds per algorithm per environment, saves detailed history, and produces
summary statistics + comparison plots.
"""
import sys, json, time, os, argparse
sys.path.insert(0, "/home/z/my-project/gate-neat")

from src.neat import NEAT
from src.gate import GATE
from src.envs import make_env, get_env_io
import numpy as np


class RandomSearch:
    """Random search baseline: sample random genomes each generation, keep best.
    Uses NEAT's genome representation but no crossover/speciation - just random sampling.
    """
    name = "RandomSearch"

    def __init__(self, n_inputs, n_outputs, cfg=None, seed=0):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.cfg = cfg or {}
        self.rng = __import__("random").Random(seed)
        from src.genome import InnovationTracker, make_initial_genome
        self.tracker = InnovationTracker()
        self.make_initial = lambda: make_initial_genome(
            n_inputs, n_outputs, self.tracker, output_activation="tanh",
            connect_input_output=True, weight_init_std=2.0, rng=self.rng,
        )
        self.population = []
        self.generation = 0
        self.history = []
        self.best_genome = None
        self.best_fitness = -float("inf")
        self.total_episodes = 0

    def step(self, env_factory):
        from src.network import FeedForwardNetwork, evaluate_episode
        from src.genome import InnovationTracker, make_initial_genome
        t0 = time.time()
        pop_size = self.cfg.get("pop_size", 80)
        eps = self.cfg.get("episodes_per_genome", 3)
        max_steps = self.cfg.get("max_steps", 1000)
        # Generate fresh random population each generation.
        self.population = [self.make_initial() for _ in range(pop_size)]
        env = env_factory()
        for genome in self.population:
            net = FeedForwardNetwork(genome)
            rewards = []
            for _ in range(eps):
                seed = self.rng.randint(0, 2**31 - 1)
                r, _, _ = evaluate_episode(net, env, seed=seed, max_steps=max_steps)
                rewards.append(r)
                self.total_episodes += 1
            genome.fitness = float(np.mean(rewards))
        env.close()
        best = max(g.fitness for g in self.population)
        mean = float(np.mean([g.fitness for g in self.population]))
        for g in self.population:
            if g.fitness > self.best_fitness:
                self.best_fitness = g.fitness
                self.best_genome = g
        stats = {
            "generation": self.generation, "best": best, "mean": mean,
            "total_episodes": self.total_episodes, "time": time.time() - t0,
        }
        self.history.append(stats)
        self.generation += 1
        return stats

    def run(self, env_factory, verbose=True):
        target = self.cfg.get("target_fitness")
        max_gen = self.cfg.get("max_generations", 100)
        for _ in range(max_gen):
            s = self.step(env_factory)
            if verbose:
                print(f"[RS] gen {s['generation']:3d} | best {s['best']:6.1f} | "
                      f"mean {s['mean']:6.1f} | eps {s['total_episodes']:5d} | {s['time']:.1f}s")
            if target is not None and s["best"] >= target:
                break
        return {"best_fitness": self.best_fitness, "total_episodes": self.total_episodes,
                "generations": self.generation, "history": self.history, "best_genome": self.best_genome}


def run_one(algo_name, env_name, seed, cfg):
    n_inputs, n_outputs = get_env_io(env_name)
    if algo_name == "NEAT":
        algo = NEAT(n_inputs, n_outputs, cfg=cfg, seed=seed)
    elif algo_name == "GATE":
        algo = GATE(n_inputs, n_outputs, cfg=cfg, seed=seed)
    elif algo_name == "RandomSearch":
        algo = RandomSearch(n_inputs, n_outputs, cfg=cfg, seed=seed)
    else:
        raise ValueError(algo_name)
    env_factory = lambda: make_env(env_name)
    t0 = time.time()
    result = algo.run(env_factory, verbose=False)
    return {
        "algo": algo_name, "env": env_name, "seed": seed,
        "best_fitness": result["best_fitness"],
        "total_episodes": result["total_episodes"],
        "generations": result["generations"],
        "wall_time": time.time() - t0,
        "history": result["history"],
    }


ENV_CONFIGS = {
    "CartPole-v1": {"target": 475.0, "max_steps": 500, "gens": 50, "pop": 80, "eps": 3},
    "MountainCar-v0": {"target": -110.0, "max_steps": 200, "gens": 80, "pop": 80, "eps": 2},
    "Acrobot-v1": {"target": -100.0, "max_steps": 500, "gens": 50, "pop": 80, "eps": 2},
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="CartPole-v1")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--algos", default="NEAT,GATE,RandomSearch")
    args = p.parse_args()

    ec = ENV_CONFIGS[args.env]
    algos = args.algos.split(",")

    base_cfg = {
        "pop_size": ec["pop"],
        "max_generations": ec["gens"],
        "episodes_per_genome": ec["eps"],
        "max_steps": ec["max_steps"],
        "target_fitness": ec["target"],
    }
    # GATE-specific overrides per env.
    gate_extras = {}
    if args.env == "MountainCar-v0":
        gate_extras = {
            "novelty_weight": 100.0,
            "novelty_k": 5,
            "weight_init_std": 2.0,
            "spsa_top_fraction": 0.5,
            "spsa_episodes": 1,
            "saliency_top_k": 2,
            "max_stagnation": 8,
            "stagnation_explore_rate": 0.7,
            "dual_spsa": True,
            "compat_threshold": 4.0,
            "behavioral_weight": 1.0,
        }
    elif args.env == "Acrobot-v1":
        gate_extras = {
            "novelty_weight": 20.0,
            "weight_init_std": 1.5,
            "spsa_top_fraction": 0.3,
            "spsa_episodes": 1,
            "saliency_top_k": 2,
            "dual_spsa": True,
            "compat_threshold": 3.5,
            "behavioral_weight": 0.5,
        }
    else:  # CartPole
        gate_extras = {
            "novelty_weight": 0.0,  # not needed
            "spsa_top_fraction": 0.25,
            "spsa_episodes": 2,
        }

    all_results = []
    for algo_name in algos:
        print(f"\n=== {algo_name} on {args.env} ===")
        for s in range(args.seeds):
            cfg = dict(base_cfg)
            if algo_name == "GATE":
                cfg.update(gate_extras)
            r = run_one(algo_name, args.env, s, cfg)
            all_results.append(r)
            print(f"  seed {s}: best={r['best_fitness']:.1f}, gens={r['generations']}, "
                  f"eps={r['total_episodes']}, wall={r['wall_time']:.1f}s")

    out_dir = "/home/z/my-project/gate-neat/results"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"bench_{args.env.lower()}.json")
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)

    import statistics
    print(f"\n=== Summary: {args.env} (target={ec['target']}) ===")
    for algo in algos:
        rs = [r for r in all_results if r["algo"] == algo]
        if not rs:
            continue
        bests = [r["best_fitness"] for r in rs]
        eps = [r["total_episodes"] for r in rs]
        gens = [r["generations"] for r in rs]
        walls = [r["wall_time"] for r in rs]
        solved = sum(1 for b in bests if (
            b >= ec["target"] if args.env != "MountainCar-v0" and args.env != "Acrobot-v1"
            else b >= ec["target"]
        ))
        print(f"{algo:13s}: best={statistics.mean(bests):7.1f}±{statistics.stdev(bests) if len(bests)>1 else 0:.1f} | "
              f"eps={statistics.mean(eps):6.0f}±{statistics.stdev(eps) if len(eps)>1 else 0:.0f} | "
              f"gens={statistics.mean(gens):4.1f} | wall={statistics.mean(walls):.1f}s | "
              f"solved {solved}/{len(bests)}")


if __name__ == "__main__":
    main()
