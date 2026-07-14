"""Run benchmark in chunks, saving after each algo. Resumable."""
import sys, json, time, os, argparse
sys.path.insert(0, "/home/z/my-project/gate-neat")
from scripts.benchmark import run_one, ENV_CONFIGS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--algo", required=True)
    p.add_argument("--offset", type=int, default=0)
    args = p.parse_args()

    ec = ENV_CONFIGS[args.env]
    base_cfg = {
        "pop_size": ec["pop"], "max_generations": ec["gens"],
        "episodes_per_genome": ec["eps"], "max_steps": ec["max_steps"],
        "target_fitness": ec["target"],
    }
    gate_extras = {}
    if args.env == "MountainCar-v0":
        gate_extras = {
            "novelty_weight": 100.0, "novelty_k": 5, "weight_init_std": 2.0,
            "spsa_top_fraction": 0.5, "spsa_episodes": 1, "saliency_top_k": 2,
            "max_stagnation": 8, "stagnation_explore_rate": 0.7,
            "dual_spsa": True, "compat_threshold": 4.0, "behavioral_weight": 1.0,
        }
    elif args.env == "Acrobot-v1":
        gate_extras = {
            "novelty_weight": 20.0, "weight_init_std": 1.5,
            "spsa_top_fraction": 0.3, "spsa_episodes": 1, "saliency_top_k": 2,
            "dual_spsa": True, "compat_threshold": 3.5, "behavioral_weight": 0.5,
        }

    cfg = dict(base_cfg)
    if args.algo == "GATE":
        cfg.update(gate_extras)

    out_dir = "/home/z/my-project/gate-neat/results"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"chunk_{args.env.lower()}_{args.algo}.json")
    # Load existing if resuming.
    if os.path.exists(out_file):
        with open(out_file) as f:
            results = json.load(f)
    else:
        results = []
    print(f"Already have {len(results)} runs in {out_file}")
    for s in range(args.offset, args.seeds):
        print(f"\n=== {args.algo} on {args.env} seed {s} ===")
        t0 = time.time()
        r = run_one(args.algo, args.env, s, cfg)
        print(f"  best={r['best_fitness']:.1f}, gens={r['generations']}, "
              f"eps={r['total_episodes']}, wall={time.time()-t0:.1f}s")
        results.append(r)
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)
    print(f"\nDone. Saved to {out_file}")


if __name__ == "__main__":
    main()
