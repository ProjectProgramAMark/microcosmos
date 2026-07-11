"""Warm-JIT fixed-capacity ecosystem benchmark."""

import argparse
import time

import jax

from microcosmos.gym import EcosystemEnv, LineTopology, MultiAgentEnv
from microcosmos.solver.config import PBD_SCHEME, PBD_SCHEME_NO_FLUID


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--max-creatures", type=int, default=64)
    parser.add_argument("--nodes-per-creature", type=int, default=16)
    parser.add_argument("--no-fluid", action="store_true")
    args = parser.parse_args()

    solver = PBD_SCHEME_NO_FLUID if args.no_fluid else PBD_SCHEME
    env = EcosystemEnv(
        topology=LineTopology(num_nodes=args.nodes_per_creature),
        max_creatures=args.max_creatures,
        initial_population=max(1, args.max_creatures // 2),
        solver_config=solver,
        grid_shape=(128, 128),
        max_steps=args.steps,
    )
    baseline = MultiAgentEnv(
        creatures=tuple(
            LineTopology(num_nodes=args.nodes_per_creature)
            for _ in range(args.max_creatures)
        ),
        solver_config=solver,
        grid_shape=(128, 128),
        max_steps=args.steps,
    )

    def time_ecosystem():
        key = jax.random.PRNGKey(0)
        _, state = env.reset(key)
        step = jax.jit(env.step)
        _, state, _, _, _ = step(key, state)
        jax.block_until_ready(state.population.energy)
        start = time.perf_counter()
        for index in range(args.steps):
            key = jax.random.fold_in(key, index)
            _, state, _, _, _ = step(key, state)
        jax.block_until_ready(state.population.energy)
        return time.perf_counter() - start

    def time_baseline():
        key = jax.random.PRNGKey(0)
        _, state = baseline.reset(key)
        action = baseline.zero_action()
        step = jax.jit(baseline.step)
        _, state, _, _, _ = step(key, state, action)
        jax.block_until_ready(state.nodes.position)
        start = time.perf_counter()
        for index in range(args.steps):
            key = jax.random.fold_in(key, index)
            _, state, _, _, _ = step(key, state, action)
        jax.block_until_ready(state.nodes.position)
        return time.perf_counter() - start

    baseline_elapsed = time_baseline()
    elapsed = time_ecosystem()
    print(f"device={jax.devices()[0].device_kind}")
    print(
        f"baseline steps={args.steps} seconds={baseline_elapsed:.6f} "
        f"ms_per_step={baseline_elapsed * 1000 / args.steps:.4f}"
    )
    print(
        f"ecosystem steps={args.steps} seconds={elapsed:.6f} "
        f"ms_per_step={elapsed * 1000 / args.steps:.4f}"
    )
    print(f"overhead_ratio={elapsed / baseline_elapsed:.4f}")


if __name__ == "__main__":
    main()
