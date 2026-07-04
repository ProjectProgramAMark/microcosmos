from typing import Callable
from dataclasses import dataclass

from microcosmos.structs.nodes import Nodes
from microcosmos.structs.edges import Edges
from .pbd import pbd_rod_constraint
from .stable_cosserat import stable_cosserat_constraint

@dataclass(frozen=True)
class ConstraintSolverConfig:
    # PBD settings
    bending_constraint: Callable[[int, tuple[Nodes, Edges], "ConstraintSolverConfig"], tuple[Nodes, Edges]]
    name: str
    cycles_per_constraint: int
    cycles_per_step: int
    stiffness_stretch: float
    stiffness_shear: float

    # Dynamics settings
    damping: float

    # Fluid simulation settings
    enable_fluid: bool
    viscosity: float
    max_fluid_velocity: float

    # Solver selection: "pbd" or "cosserat"
    solver_type: str = "pbd"

    # Timestep — must match simulation dt when solver_type == "cosserat".
    # Used to compute inertia weight w = 1/dt².
    dt: float = 0.05

    # Immersed boundary settings
    ibm_iterations: int = 5
    synthetic_node_width: float = 0.0
    ibm_kernel_size: int = 3  # Stencil width per axis (N×N grid points per node); smaller = faster, less smooth
    ibm_relaxation: float = 0.3  # Per-iter velocity correction gain in IBM multi-direct forcing loop


    # Node mass in lattice units.
    # Each lattice cell has mass ≈ ρ·dx² ≈ 1 (density~1, spacing=1).
    # Set to ~1 so one node ≈ one lattice cell's worth of fluid inertia.
    # Automatically scaled by fluid_grid_shape/grid_shape ratio inside IBM.
    node_mass: float = 1.0

    # TRT magic parameter Λ = (1/ω⁺ - 0.5)(1/ω⁻ - 0.5).
    # Controls the free (antisymmetric) relaxation rate.
    lambda_trt: float = 0.25
    # lambda_trt: float = 0.16667  # 0.25

    # Synthetic IBM nodes: offset distance (in domain units) for the perpendicular node pair.
    # Each control node spawns two synthetic nodes at ± this distance along the chain normal.
    # These interact with the fluid instead of the control node, increasing effective drag width.
    # 0.0 = disabled (use control nodes directly).

    # Clamp force field magnitude before LBM collision (prevents NaN from dense nodes).
    # Breaks momentum conservation — off by default.
    enable_force_clamp: bool = False

    # Apply steric (self-avoidance) forces to predicted positions before constraint solve.
    # FFT density grid, curl-free bilinear sampling, analytic n-neighbor subtraction.
    enable_steric: bool = False
    steric_strength: float = 1.0
    steric_sigma: float = 1.5
    # Number of chain neighbors on EACH side to subtract from the sampled force.
    # 0 = no subtraction (each node feels its own bump + neighbors').
    steric_neighbor_skip: int = 5
    # Per-node weight scattered into the density grid before Gaussian smoothing.
    # Lower = gentler force baseline.
    steric_scatter_value: float = 1.0


PBD_SCHEME = ConstraintSolverConfig(
    # PBD settings
    bending_constraint = pbd_rod_constraint,
    name = "cosserat rods",
    cycles_per_constraint = 2,
    cycles_per_step = 20,

    stiffness_stretch = 0.6,
    stiffness_shear = 1.0,

    # Dynamics settings
    damping = 1.0,

    # Fluid simulation settings
    enable_fluid = True,
    viscosity = 0.01,
    max_fluid_velocity = 0.2,
    synthetic_node_width = 0.0,  # offset in domain units; ~1 lattice cell at default resolution

    # Immersed boundary settings
    ibm_iterations = 5,
    ibm_kernel_size = 2,  # use even numbers with peskin kernel. odd for hat kernel. peskin best w 4 or 6.

)

# Preset without fluid simulation
PBD_SCHEME_NO_FLUID = ConstraintSolverConfig(
    # PBD settings
    bending_constraint = pbd_rod_constraint,
    name = "cosserat rods (no fluid)",
    cycles_per_constraint = 2,
    cycles_per_step = 10,

    stiffness_stretch = 0.7,
    stiffness_shear = 0.7,

    # Dynamics settings (no fluid, so no momentum to preserve)
    damping = 0.0,

    # Fluid simulation settings
    enable_fluid = False,
    viscosity = 0.1,
    max_fluid_velocity = 0.1,
    enable_steric = False,
    # steric_strength=1000.0
)

# Stable Cosserat preset (no fluid) — Projective Dynamics implicit solver.
# stiffness_stretch and stiffness_shear are physical stiffness coefficients
# (not 0–1 fractions like in PBD). At dt=0.05, inertia weight w = 400;
# k_s=50 and l_e≈0.3 gives k_s/l_e²≈555, so constraints converge in a few
# iterations while inertia prevents oscillation at any stiffness value.
COSSERAT_SCHEME_NO_FLUID = ConstraintSolverConfig(
    bending_constraint = stable_cosserat_constraint,
    solver_type = "cosserat",
    name = "stable cosserat (no fluid)",
    cycles_per_constraint = 2,
    cycles_per_step = 20,

    stiffness_stretch = 50.0,
    stiffness_shear = 50.0,

    damping = 0.0,
    dt = 0.05,  # must match the simulation dt passed to simulate()

    enable_fluid = False,
    viscosity = 0.1,
    max_fluid_velocity = 0.1,
    enable_steric = False,
    steric_strength = 100.0,
)

# Stable Cosserat preset with fluid coupling.
# damping=1.0 so velocity carries between steps — required for swimming dynamics.
# dt MUST be replaced with actual simulation dt (sets inertia weight w = 1/dt²).
# stiffness scales as k_s ∝ l_edge² · w so the constraint/inertia ratio stays ~1:
#   reference (filament_folding): k_s=50,  l_e=0.3, dt=0.05 → k_s/l²≈556, w=400
#   worm (l_e=2.0, dt=0.01):     k_s=50000 → k_s/l²=12500, w=10000, ratio≈1.25
COSSERAT_SCHEME = ConstraintSolverConfig(
    bending_constraint = stable_cosserat_constraint,
    solver_type = "cosserat",
    name = "stable cosserat with fluid",
    cycles_per_constraint = 2,
    cycles_per_step = 20,

    stiffness_stretch = 50.0,   # placeholder; scale to l_edge² · (1/dt²) for your experiment
    stiffness_shear = 50.0,

    damping = 1.0,              # keep velocity so momentum accumulates (unlike NO_FLUID's 0.0)
    dt = 0.05,                  # placeholder — MUST replace with actual simulation dt

    enable_fluid = True,
    viscosity = 0.01,
    max_fluid_velocity = 0.2,
    synthetic_node_width = 0.0,
    ibm_iterations = 5,
    ibm_kernel_size = 2,
)
