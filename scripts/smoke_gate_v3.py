"""Test GATE-v3 on CartPole and MountainCar."""
import sys, time
sys.path.insert(0, "/home/z/my-project/gate-neat")

from src.gate_v3 import GATEv3
from src.envs import make_env, get_env_io


def test_cartpole():
    env_name = "CartPole-v1"
    n_in, n_out = get_env_io(env_name)
    cfg = {
        "pop_size": 50, "max_generations": 15, "episodes_per_genome": 2,
        "spsa_episodes": 1, "max_steps": 500, "target_fitness": 475.0,
    }
    gate = GATEv3(n_in, n_out, cfg=cfg, seed=42)
    t0 = time.time()
    r = gate.run(lambda: make_env(env_name), verbose=True)
    print(f"\nCartPole: best={r['best_fitness']:.1f}, eps={r['total_episodes']}, gens={r['generations']}, wall={time.time()-t0:.1f}s")


def test_mountaincar():
    env_name = "MountainCar-v0"
    n_in, n_out = get_env_io(env_name)
    cfg = {
        "pop_size": 80, "max_generations": 80, "episodes_per_genome": 2,
        "max_steps": 200, "target_fitness": -110.0,
        "novelty_weight": 100.0, "novelty_k": 5, "weight_init_std": 2.0,
        "spsa_top_fraction": 0.5, "spsa_episodes": 1, "n_splits_per_elite": 2,
        "max_stagnation": 8, "stagnation_explore_rate": 0.7,
        "compat_threshold": 4.0, "behavioral_weight": 1.0,
        "adaptive_novelty": True, "adaptive_novelty_patience": 3,
        "temp_start": 3.0, "temp_end": 0.5, "temp_decay": 0.95,
    }
    gate = GATEv3(n_in, n_out, cfg=cfg, seed=0)
    t0 = time.time()
    r = gate.run(lambda: make_env(env_name), verbose=False)
    print(f"MountainCar: best={r['best_fitness']:.1f}, eps={r['total_episodes']}, gens={r['generations']}, wall={time.time()-t0:.1f}s")


if __name__ == "__main__":
    test_cartpole()
    print()
    test_mountaincar()
