"""
Smoke test: run NEAT on CartPole-v1 for a few generations to verify the pipeline works.
"""
import sys
sys.path.insert(0, "/home/z/my-project/gate-neat")

from src.neat import NEAT
from src.envs import make_env, get_env_io


def main():
    env_name = "CartPole-v1"
    n_inputs, n_outputs = get_env_io(env_name)
    print(f"Env: {env_name}, inputs={n_inputs}, outputs={n_outputs}")

    cfg = {
        "pop_size": 50,
        "max_generations": 10,
        "episodes_per_genome": 2,
        "max_steps": 500,
        "target_fitness": 475.0,
    }
    neat = NEAT(n_inputs, n_outputs, cfg=cfg, seed=42)
    env_factory = lambda: make_env(env_name)
    result = neat.run(env_factory, verbose=True)
    print("\n--- Result ---")
    print(f"Best fitness: {result['best_fitness']}")
    print(f"Total episodes: {result['total_episodes']}")
    print(f"Generations: {result['generations']}")


if __name__ == "__main__":
    main()
