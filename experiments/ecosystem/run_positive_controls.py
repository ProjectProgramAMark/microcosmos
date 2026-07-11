"""Run and record the frozen fluid-on ecosystem positive controls."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from microcosmos.positive_controls import (
    foraging_control_environment,
    foraging_gate_metrics,
    movement_control_environment,
    movement_gate_metrics,
    paired_rollout,
    sensory_foraging_genome,
    sensor_disabled_genome,
    traveling_wave_genome,
    with_genome,
    with_relative_resource_patch,
    zero_action_genome,
)
from microcosmos.rendering import animate


def _stack_states(states):
    return jax.tree.map(lambda *values: jnp.stack(values), *states)


def _json_scalars(metrics):
    converted = {}
    for name, value in metrics.items():
        if value.dtype == jnp.bool_:
            converted[name] = bool(value)
        elif jnp.issubdtype(value.dtype, jnp.integer):
            converted[name] = int(value)
        else:
            converted[name] = float(value)
    return converted


def run_controls(output_dir: Path, horizon: int = 5_000) -> dict:
    if jax.default_backend() != "gpu":
        raise RuntimeError("positive controls require the fluid-on GPU fixture")
    output_dir.mkdir(parents=True, exist_ok=True)

    movement_env = movement_control_environment(horizon)
    _, movement_base = movement_env.reset(jax.random.PRNGKey(11))
    movement_keys = jax.random.split(jax.random.PRNGKey(12), horizon)
    movement_final, movement_trace = jax.jit(
        lambda state_a, state_b, keys: paired_rollout(
            movement_env, state_a, state_b, keys
        )
    )(
        with_genome(movement_base, traveling_wave_genome()),
        with_genome(movement_base, zero_action_genome()),
        movement_keys,
    )
    jax.block_until_ready(movement_trace[0])
    movement_metrics = movement_gate_metrics(
        movement_trace[0][-1],
        movement_trace[1][-1],
        body_length=14.0,
    )

    foraging_env = foraging_control_environment(horizon)
    seeds = range(8)
    bases = [
        with_relative_resource_patch(
            foraging_env,
            foraging_env.reset(jax.random.PRNGKey(seed))[1],
        )
        for seed in seeds
    ]
    sensory_states = _stack_states(
        [with_genome(state, sensory_foraging_genome()) for state in bases]
    )
    control_states = _stack_states(
        [with_genome(state, sensor_disabled_genome()) for state in bases]
    )
    paired_keys = jnp.stack(
        [
            jax.random.split(
                jax.random.fold_in(jax.random.PRNGKey(100), seed), horizon
            )
            for seed in seeds
        ]
    )
    _, foraging_trace = jax.jit(
        jax.vmap(
            lambda state_a, state_b, keys: paired_rollout(
                foraging_env, state_a, state_b, keys
            )
        )
    )(sensory_states, control_states, paired_keys)
    jax.block_until_ready(foraging_trace[2])
    sensory_uptake = foraging_trace[2][:, -1]
    control_uptake = foraging_trace[3][:, -1]
    foraging_metrics = foraging_gate_metrics(sensory_uptake, control_uptake)

    np.savez_compressed(
        output_dir / "positive_control_traces.npz",
        wave_displacement=np.asarray(movement_trace[0]),
        zero_displacement=np.asarray(movement_trace[1]),
        sensory_cumulative_uptake=np.asarray(foraging_trace[2]),
        control_cumulative_uptake=np.asarray(foraging_trace[3]),
        resource_capacity=np.asarray(bases[0].resource_capacity_map),
        initial_resource_stock=np.asarray(bases[0].fields.energy),
    )

    summary = {
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "horizon": horizon,
        "movement": _json_scalars(movement_metrics),
        "foraging": _json_scalars(foraging_metrics),
        "sensory_uptake": np.asarray(sensory_uptake).tolist(),
        "control_uptake": np.asarray(control_uptake).tolist(),
        "finite": bool(
            jnp.all(jnp.isfinite(movement_final[0].nodes.position))
            & jnp.all(jnp.isfinite(movement_trace[0]))
            & jnp.all(jnp.isfinite(foraging_trace[2]))
        ),
    }
    (output_dir / "positive_control_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    capacity = np.asarray(bases[0].resource_capacity_map)
    stock = np.asarray(bases[0].fields.energy)
    image_options = dict(vmin=0.0, vmax=foraging_env.resource_config.capacity)
    axes[0].imshow(capacity, origin="lower", **image_options)
    axes[0].set_title("Capacity (fixed scale)")
    axes[1].imshow(stock, origin="lower", **image_options)
    axes[1].set_title("Initial stock (fixed scale)")
    axes[2].plot(np.asarray(foraging_trace[2]).T, color="tab:green", alpha=0.35)
    axes[2].plot(np.asarray(foraging_trace[3]).T, color="tab:gray", alpha=0.25)
    axes[2].set_title("Cumulative mouth uptake")
    axes[2].set_xlabel("step")
    figure.tight_layout()
    figure.savefig(output_dir / "resource_diagnostic.png", dpi=160)
    plt.close(figure)

    if not summary["finite"]:
        raise RuntimeError("positive-control rollout became non-finite")
    if not summary["movement"]["passed"]:
        raise RuntimeError("movement positive control failed")
    if not summary["foraging"]["passed"]:
        raise RuntimeError("foraging positive control failed")
    return summary


def write_movement_video(output_dir: Path, steps: int = 1_000) -> None:
    """Render the prescribed wave through the ordinary Microcosmos renderer."""
    env = movement_control_environment(steps)
    _, initial = env.reset(jax.random.PRNGKey(11))
    initial = with_genome(initial, traveling_wave_genome())
    keys = jax.random.split(jax.random.PRNGKey(12), steps)

    def scan_step(state, key):
        _, new_state, _, _, _ = env.step(key, state)
        fields_for_render = replace(new_state.fields, f_grid=None)
        return new_state, (new_state.nodes, fields_for_render)

    _, (nodes, fields) = jax.jit(
        lambda state, scan_keys: jax.lax.scan(scan_step, state, scan_keys)
    )(initial, keys)
    jax.block_until_ready(nodes.position)
    animate(
        nodes,
        fields,
        filename=str(output_dir / "traveling_wave_native.mp4"),
        subsample=10,
        fps=30,
        animate_filament=False,
        animate_fluid_velocity=True,
        animate_energy=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/ecosystem_positive_controls"),
    )
    parser.add_argument("--horizon", type=int, default=5_000)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video-steps", type=int, default=1_000)
    arguments = parser.parse_args()
    summary = run_controls(arguments.output_dir, arguments.horizon)
    if arguments.video:
        write_movement_video(arguments.output_dir, arguments.video_steps)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
