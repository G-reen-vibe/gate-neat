"""
Run baseline NEAT on CartPole-v1 across multiple seeds to establish variance.
"""
import sys, json, time, os
sys.path.insert(0, "/home/z/my-project/gate-neat")

from src.neat import NEAT
from src.envs import make_env, get_env_io

OUT_DIR = "/home/z/my-project/gate-neat/results/baseline_neat"
os.makedirs(OUT_DIR, exist_ok=True)


def run_seed(seed, env_name="CartPole-v1"):
    n_inputs, n_outputs = get_env_io(env_name)
    cfg = {
        "pop_size": 80,
        "max_generations": 50,
        "episodes_per_genome": 3,
        "max_steps": 500,
        "target_fitness": 475.0,
    }
    neat = NEAT(n_inputs, n_outputs, cfg=cfg, seed=seed)
    env_factory = lambda: make_env(env_name)
    t0 = time.time()
    result = neat.run(env_factory, verbose=False)
    wall = time.time() - t0
    return {
        "seed": seed,
        "best_fitness": result["best_fitness"],
        "total_episodes": result["total_episodes"],
        "generations": result["generations"],
        "wall_time": wall,
        "history": result["history"],
    }


def main():
    seeds = [0, 1, 2, 3, 4, 5, 6, 7]
    results = []
    for s in seeds:
        r = run_seed(s)
        results.append(r)
        print(f"seed {s}: best={r['best_fitness']:.1f}, gens={r['generations']}, "
              f"eps={r['total_episodes']}, wall={r['wall_time']:.1f}s")
    # Save
    with open(os.path.join(OUT_DIR, "cartpole_v1_seeds.json"), "w") as f:
        json.dump(results, f, indent=2)
    # Summary
    bests = [r["best_fitness"] for r in results]
    eps = [r["total_episodes"] for r in results]
    gens = [r["generations"] for r in results]
    import statistics
    print("\n=== Summary ===")
    print(f"Best fitness: mean={statistics.mean(bests):.1f}, std={statistics.stdev(bests):.1f}, "
          f"min={min(bests):.1f}, max={max(bests):.1f}")
    print(f"Episodes to solve: mean={statistics.mean(eps):.0f}, std={statistics.stdev(eps):.0f}")
    print(f"Generations: mean={statistics.mean(gens):.1f}, std={statistics.stdev(gens):.1f}")
    solved = sum(1 for b in bests if b >= 475)
    print(f"Solved: {solved}/{len(bests)}")


if __name__ == "__main__":
    main()
