"""
Multi-seed comparison: NEAT vs GATE on CartPole-v1.
Measures episodes-to-solve, generations-to-solve, and final best fitness.
"""
import sys, json, time, os, argparse
sys.path.insert(0, "/home/z/my-project/gate-neat")

from src.neat import NEAT
from src.gate import GATE
from src.envs import make_env, get_env_io


def run_one(algo_name, env_name, seed, cfg):
    n_inputs, n_outputs = get_env_io(env_name)
    if algo_name == "NEAT":
        algo = NEAT(n_inputs, n_outputs, cfg=cfg, seed=seed)
    elif algo_name == "GATE":
        algo = GATE(n_inputs, n_outputs, cfg=cfg, seed=seed)
    else:
        raise ValueError(algo_name)
    env_factory = lambda: make_env(env_name)
    t0 = time.time()
    result = algo.run(env_factory, verbose=False)
    return {
        "algo": algo_name,
        "env": env_name,
        "seed": seed,
        "best_fitness": result["best_fitness"],
        "total_episodes": result["total_episodes"],
        "generations": result["generations"],
        "wall_time": time.time() - t0,
        "history": result["history"],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="CartPole-v1")
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--gens", type=int, default=50)
    p.add_argument("--pop", type=int, default=80)
    p.add_argument("--eps", type=int, default=3)
    p.add_argument("--target", type=float, default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg = {
        "pop_size": args.pop,
        "max_generations": args.gens,
        "episodes_per_genome": args.eps,
        "max_steps": 500 if "CartPole" in args.env else 1000,
    }
    if args.target is not None:
        cfg["target_fitness"] = args.target
    elif args.env == "CartPole-v1":
        cfg["target_fitness"] = 475.0
    elif args.env == "CartPole-v0":
        cfg["target_fitness"] = 195.0

    all_results = []
    for algo_name in ["NEAT", "GATE"]:
        print(f"\n=== {algo_name} on {args.env} ===")
        for s in range(args.seeds):
            r = run_one(algo_name, args.env, s, cfg)
            all_results.append(r)
            print(f"  seed {s}: best={r['best_fitness']:.1f}, gens={r['generations']}, "
                  f"eps={r['total_episodes']}, wall={r['wall_time']:.1f}s")

    out_file = args.out or f"/home/z/my-project/gate-neat/results/comparison_{args.env.lower()}.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)

    import statistics
    print(f"\n=== Summary: {args.env} ===")
    for algo in ["NEAT", "GATE"]:
        rs = [r for r in all_results if r["algo"] == algo]
        bests = [r["best_fitness"] for r in rs]
        eps = [r["total_episodes"] for r in rs]
        gens = [r["generations"] for r in rs]
        walls = [r["wall_time"] for r in rs]
        print(f"{algo:6s}: best mean={statistics.mean(bests):6.1f} ± {statistics.stdev(bests) if len(bests)>1 else 0:.1f} | "
              f"eps mean={statistics.mean(eps):5.0f} ± {statistics.stdev(eps) if len(eps)>1 else 0:.0f} | "
              f"gens mean={statistics.mean(gens):4.1f} | "
              f"wall mean={statistics.mean(walls):.1f}s")


if __name__ == "__main__":
    main()
