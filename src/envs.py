"""
Environment factory for RL benchmarks (CartPole, LunarLander, MountainCar, Acrobot).
All envs are Gymnasium discrete-action environments, which keeps the NEAT side simple.
"""
from __future__ import annotations
import gymnasium as gym
from typing import Tuple


def make_env(env_name: str):
    """Create a gymnasium env. Returns the env object."""
    # Use older v* versions when available (CartPole-v1, LunarLander-v2, etc.)
    # Gymnasium will pick a sensible default.
    try:
        env = gym.make(env_name)
    except Exception:
        # Fallback: try without version
        env = gym.make(env_name.split("-")[0])
    return env


def get_env_io(env_name: str) -> Tuple[int, int]:
    """Return (n_inputs, n_outputs) for a given environment."""
    env = make_env(env_name)
    obs_space = env.observation_space
    act_space = env.action_space
    env.close()
    if hasattr(obs_space, "shape"):
        n_inputs = int(obs_space.shape[0])
    else:
        n_inputs = obs_space.n
    if hasattr(act_space, "n"):
        n_outputs = int(act_space.n)
    else:
        # Continuous action space - not supported by this NEAT impl.
        raise ValueError(f"Continuous action space not supported for {env_name}")
    return n_inputs, n_outputs


ENV_REGISTRY = {
    "CartPole-v1": {"target": 475.0, "max_steps": 500},
    "CartPole-v0": {"target": 195.0, "max_steps": 200},
    "MountainCar-v0": {"target": -110.0, "max_steps": 200},
    "Acrobot-v1": {"target": -100.0, "max_steps": 500},
    "LunarLander-v2": {"target": 200.0, "max_steps": 1000},
}
