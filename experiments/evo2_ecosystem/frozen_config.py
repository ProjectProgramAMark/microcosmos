"""Ecology selected by the one-time Evo² viability calibration."""

from types import MappingProxyType


CALIBRATED_ECOLOGY = MappingProxyType(
    {
        "actuation_power_coefficient": 0.002,
        "assimilation_efficiency": 0.9,
        "basal_metabolism": 0.01,
        "birth_transfer_efficiency": 0.8,
        "initial_population": 8,
        "initial_resource_fraction": 1.0,
        "maturity_age": 50,
        "maximum_lifespan": 5_000,
        "reproduction_cost": 1.5,
        "reproduction_threshold": 3.0,
        "resource_regeneration_rate": 0.03,
        "uptake_rate": 1.0,
    }
)
CALIBRATED_ECOLOGY_SHA256 = "ceab22348b95db80ec8141628e9f7740a8753cf0448573d0a09bfe5eb072ee4d"

# Numerical measurements of one fixed ecological panel. These are never
# counted as independent ecological/world replicates.
NUMERICAL_REPEATS = 3
