"""Compare GATE-v2 vs GATE-v3 on CartPole and MountainCar, multiple seeds."""
import sys, json, time, os
sys.path.insert(0, "/home/z/my-project/gate-neat")

from src.gate import GATE
from src.gate_v3 import GATEv3
from src.envs import make_env, get_env_io
import numpy as np


def run_one(algo_class, algo_name, env_name, seed, cfg):
    n_in, n_out = get_env_io(env_name)
    algo = algo_class(n_in, n_out, cfg=cfg, seed=seed)
    env_factory = lambda: make_env(env_name)
    t0 = time.time()
    r = algo.run(env_factory, verbose=False)
    return {
        "algo": algo_name, "env": env_name, "seed": seed,
        "best_fitness": r["best_fitness"], "total_episodes": r["total_episodes"],
        "generations": r["generations"], "wall_time": time.time() - t0,
        "history": r["history"],
    }


def main():
    out_dir = "/home/z/my-project/gate-neat/results"
    os.makedirs(out_dir, exist_ok=True)
    
    # CartPole comparison
    print("=== CartPole-v1: GATE-v2 vs GATE-v3 (5 seeds) ===")
    cp_cfg = {
        "pop_size": 80, "max_generations": 30, "episodes_per_genome": 3,
        "max_steps": 500, "target_fitness": 475.0,
        "spsa_episodes": 2,
    }
    results_cp = []
    for algo_name, algo_class in [("GATE-v2", GATE), ("GATE-v3", GATEv3)]:
        for s in range(5):
            r = run_one(algo_class, algo_name, "CartPole-v1", s, cp_cfg)
            results_cp.append(r)
            print(f"  {algo_name} seed {s}: best={r['best_fitness']:.1f}, eps={r['total_episodes']}, gens={r['generations']}")
    with open(os.path.join(out_dir, "v2_vs_v3_cartpole.json"), "w") as f:
        json.dump(results_cp, f, indent=2)

    # MountainCar comparison
    print("\n=== MountainCar-v0: GATE-v2 vs GATE-v3 (3 seeds) ===")
    mc_cfg = {
        "pop_size": 80, "max_generations": 80, "episodes_per_genome": 2,
        "max_steps": 200, "target_fitness": -110.0,
        "novelty_weight": 100.0, "novelty_k": 5, "weight_init_std": 2.0,
        "spsa_top_fraction": 0.5, "spsa_episodes": 1,
        "max_stagnation": 8, "stagnation_explore_rate": 0.7,
        "compat_threshold": 4.0, "behavioral_weight": 1.0,
        "adaptive_novelty": True, "adaptive_novelty_patience": 3,
    }
    # v3-specific
    mc_cfg_v3 = dict(mc_cfg)
    mc_cfg_v3["n_splits_per_elite"] = 2
    mc_cfg_v3["temp_start"] = 3.0
    mc_cfg_v3["temp_end"] = 0.5
    results_mc = []
    for algo_name, algo_class, cfg in [("GATE-v2", GATE, mc_cfg), ("GATE-v3", GATEv3, mc_cfg_v3)]:
        for s in range(3):
            r = run_one(algo_class, algo_name, "MountainCar-v0", s, cfg)
            results_mc.append(r)
            print(f"  {algo_name} seed {s}: best={r['best_fitness']:.1f}, eps={r['total_episodes']}, gens={r['generations']}")
    with open(os.path.join(out_dir, "v2_vs_v3_mountaincar.json"), "w") as f:
        json.dump(results_mc, f, indent=2)

    # Summary
    import statistics
    print("\n=== Summary ===")
    for env_name, results in [("CartPole-v1", results_cp), ("MountainCar-v0", results_mc)]:
        print(f"\n{env_name}:")
        for algo in ["GATE-v2", "GATE-v3"]:
            rs = [r for r in results if r["algo"] == algo]
            bests = [r["best_fitness"] for r in rs]
            eps = [r["total_episodes"] for r in rs]
            print(f"  {algo}: best={statistics.mean(bests):.1f}±{statistics.stdev(bests) if len(bests)>1 else 0:.1f}, "
                  f"eps={statistics.mean(eps):.0f}±{statistics.stdev(eps) if len(eps)>1 else 0:.0f}")


if __name__ == "__main__":
    main()
