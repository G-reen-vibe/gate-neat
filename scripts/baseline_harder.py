"""
Run baseline NEAT on MountainCar-v0 and Acrobot-v1 to establish harder benchmarks.
These require exploration and longer-horizon planning.
"""
import sys, json, time, os
sys.path.insert(0, "/home/z/my-project/gate-neat")

from src.neat import NEAT
from src.envs import make_env, get_env_io

OUT_DIR = "/home/z/my-project/gate-neat/results/baseline_neat"
os.makedirs(OUT_DIR, exist_ok=True)


def run_seed(seed, env_name, cfg):
    n_inputs, n_outputs = get_env_io(env_name)
    neat = NEAT(n_inputs, n_outputs, cfg=cfg, seed=seed)
    env_factory = lambda: make_env(env_name)
    t0 = time.time()
    result = neat.run(env_factory, verbose=False)
    wall = time.time() - t0
    return {
        "seed": seed,
        "env": env_name,
        "best_fitness": result["best_fitness"],
        "total_episodes": result["total_episodes"],
        "generations": result["generations"],
        "wall_time": wall,
        "history": result["history"],
    }


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="MountainCar-v0")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--gens", type=int, default=80)
    p.add_argument("--pop", type=int, default=80)
    p.add_argument("--eps", type=int, default=3)
    p.add_argument("--target", type=float, default=None)
    args = p.parse_args()

    cfg = {
        "pop_size": args.pop,
        "max_generations": args.gens,
        "episodes_per_genome": args.eps,
        "max_steps": 1000,
    }
    if args.target is not None:
        cfg["target_fitness"] = args.target

    results = []
    for s in range(args.seeds):
        r = run_seed(s, args.env, cfg)
        results.append(r)
        print(f"seed {s}: best={r['best_fitness']:.1f}, gens={r['generations']}, "
              f"eps={r['total_episodes']}, wall={r['wall_time']:.1f}s")
    out_file = os.path.join(OUT_DIR, f"{args.env.lower()}_seeds.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    import statistics
    bests = [r["best_fitness"] for r in results]
    print(f"\n=== {args.env} ===")
    print(f"Best: mean={statistics.mean(bests):.1f}, std={statistics.stdev(bests) if len(bests)>1 else 0:.1f}, "
          f"min={min(bests):.1f}, max={max(bests):.1f}")


if __name__ == "__main__":
    main()
