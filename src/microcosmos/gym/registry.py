from .base import Environment
from .line import LineEnv
from .line_ring import LineRingEnv
from .multi_agent import MultiAgentEnv
from .random_food import RandomFoodLineEnv
from .ring import RingEnv
from .tadpole import TadpoleEnv
from .ecosystem import EcosystemEnv


_REGISTRY: dict[str, type[Environment]] = {
    "line": LineEnv,
    "line-random-food": RandomFoodLineEnv,
    "line-ring": LineRingEnv,
    "multi-agent": MultiAgentEnv,
    "ring": RingEnv,
    "tadpole": TadpoleEnv,
    "ecosystem": EcosystemEnv,
}


def make(name: str, **kwargs) -> Environment:
    """Build a predefined environment by name.

    Available envs are exposed by ``registered_envs``. Extra kwargs are forwarded
    to the env's constructor (e.g. ``num_nodes``, ``grid_shape``, ``max_steps``).
    """
    if name not in _REGISTRY:
        raise ValueError(
            f"unknown env {name!r}; choices: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name](**kwargs)


def registered_envs() -> list[str]:
    return sorted(_REGISTRY)
