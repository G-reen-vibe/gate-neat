"""Test GATE with novelty bonus on MountainCar-v0."""
import sys, time
sys.path.insert(0, "/home/z/my-project/gate-neat")

from src.gate import GATE
from src.envs import make_env, get_env_io


def main():
    env_name = "MountainCar-v0"
    n_in, n_out = get_env_io(env_name)
    print(f"Env: {env_name}, io=({n_in},{n_out})")
    cfg = {
        "pop_size": 80,
        "max_generations": 80,
        "episodes_per_genome": 2,
        "max_steps": 200,
        "target_fitness": -110.0,
        "novelty_weight": 100.0,    # very strong novelty pressure (raw reward is ~-200)
        "novelty_k": 5,
        "weight_init_std": 2.0,     # larger initial weights = diverse behaviors
        "spsa_top_fraction": 0.5,
        "spsa_episodes": 1,
        "saliency_top_k": 2,
        "max_stagnation": 8,
        "stagnation_explore_rate": 0.7,
        "dual_spsa": True,
        "compat_threshold": 4.0,    # wider species to maintain diversity
        "behavioral_weight": 1.0,
    }
    gate = GATE(n_in, n_out, cfg=cfg, seed=0)
    t0 = time.time()
    r = gate.run(lambda: make_env(env_name), verbose=True)
    print(f"\\nwall: {time.time()-t0:.1f}s, best fitness: {r['best_fitness']:.1f}, eps: {r['total_episodes']}")


if __name__ == "__main__":
    main()
