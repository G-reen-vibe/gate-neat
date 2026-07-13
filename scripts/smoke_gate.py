"""Smoke test for GATE on CartPole-v1."""
import sys
sys.path.insert(0, "/home/z/my-project/gate-neat")

from src.gate import GATE
from src.envs import make_env, get_env_io


def main():
    env_name = "CartPole-v1"
    n_inputs, n_outputs = get_env_io(env_name)
    print(f"Env: {env_name}, inputs={n_inputs}, outputs={n_outputs}")

    cfg = {
        "pop_size": 50,
        "max_generations": 15,
        "episodes_per_genome": 2,
        "saliency_episodes": 1,  # very fast probing
        "max_steps": 500,
        "target_fitness": 475.0,
        "saliency_top_k": 1,
    }
    gate = GATE(n_inputs, n_outputs, cfg=cfg, seed=42)
    env_factory = lambda: make_env(env_name)
    result = gate.run(env_factory, verbose=True)
    print("\n--- Result ---")
    print(f"Best fitness: {result['best_fitness']}")
    print(f"Total episodes: {result['total_episodes']}")
    print(f"Generations: {result['generations']}")


if __name__ == "__main__":
    main()
