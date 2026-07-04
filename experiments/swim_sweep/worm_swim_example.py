import jax
import jax.numpy as jnp
from dataclasses import replace

from microcosmos.env import MicrocosmosEnv
from microcosmos.rendering import animate
from microcosmos.simulate import step as sim_step
from microcosmos.solver.config import PBD_SCHEME, COSSERAT_SCHEME

from experiments.base_experiment import Experiment
from experiments.swim_sweep.swimmers.worm import Worm


class WormSwimExample(Experiment):
    """Minimal example: a hard-coded sine-wave swimmer.

    In realtime mode: renders the worm swimming interactively.
    In batch mode: runs a JAX-scan rollout and saves an animation.

    Run with:
        uv run python main.py experiment=worm_swim_example
    Supports physics.scheme: 'pbd' (default) or 'stable_cosserat'.
    """

    def _build_solver_config(self):
        phys = self.cfg.get('physics', {})
        scheme_name = str(phys.get('scheme', 'pbd'))
        dt = float(self.cfg.simulation.dt)

        base = COSSERAT_SCHEME if scheme_name == 'stable_cosserat' else PBD_SCHEME

        overrides = {}
        for key in ('viscosity', 'max_fluid_velocity', 'synthetic_node_width'):
            if key in phys:
                overrides[key] = float(phys[key])
        if 'ibm_kernel_size' in phys:
            overrides['ibm_kernel_size'] = int(phys['ibm_kernel_size'])

        # Scheme-specific stiffness keys
        stretch_key = f'stiffness_stretch_{scheme_name}' if scheme_name != 'pbd' else 'stiffness_stretch_pbd'
        shear_key   = f'stiffness_shear_{scheme_name}'   if scheme_name != 'pbd' else 'stiffness_shear_pbd'
        if phys.get(stretch_key) is not None:
            overrides['stiffness_stretch'] = float(phys[stretch_key])
        if phys.get(shear_key) is not None:
            overrides['stiffness_shear'] = float(phys[shear_key])

        if scheme_name == 'stable_cosserat':
            # dt must match simulation dt so inertia weight w=1/dt² is correct
            overrides['dt'] = dt
            for key in ('damping', 'cycles_per_step'):
                if key in phys:
                    overrides[key] = float(phys[key]) if key != 'cycles_per_step' else int(phys[key])

        return base.__replace__(**overrides)

    def setup(self):
        self.worm = Worm(self.cfg, self.output_dir)
        self.worm.setup()
        solver_config = self._build_solver_config()
        self.env = MicrocosmosEnv(
            grid_shape=tuple(self.cfg.experiment.grid_shape),
            dt=self.cfg.simulation.dt,
            num_steps=self.cfg.simulation.num_steps,
            solver_config=solver_config,
        )
        print(f"Solver: {self.env.solver_config.name}")

    def loss_fn(self, _params):
        raise NotImplementedError

    def run(self):
        if self.cfg.simulation.get('realtime', True):
            self.worm.run_realtime()
        else:
            control_fn = self.worm.make_control_fn(self.env)
            dt = self.env.dt
            solver_config = self.env.solver_config

            def scan_step(carry, t):
                nodes, edges, fields = carry
                edges = control_fn(nodes, edges, t)
                nodes, edges, fields = sim_step(nodes, edges, fields, dt, solver_config)
                fields_out = replace(fields, f_grid=None)
                return (nodes, edges, fields), (nodes, fields_out)

            ts = jnp.arange(self.env.num_steps)
            _, (nodes_ts, fields_ts) = jax.lax.scan(
                scan_step, (self.worm.nodes, self.worm.edges, self.worm.fields), ts
            )

            # Displacement: COM movement from start to end (raw, no periodic correction)
            positions = nodes_ts.position          # (T, N, 2)
            com = jnp.mean(positions, axis=1)      # (T, 2)
            disp = jnp.linalg.norm(com[-1] - com[0])
            print(f"Net displacement: {float(disp):.4f}  (scheme: {solver_config.name})")

            subsample = self.cfg.simulation.get('subsample', 4)
            out = str(self.output_dir / 'worm_swim.gif')
            animate(nodes_ts, fields_ts, filename=out, subsample=subsample)
            print(f"Saved {out}")
