from .swim_sweep.swim_sweep import SwimSweepExperiment
from .swim_sweep.worm_swim_example import WormSwimExample
from .filament_folding.experiment import FilamentFoldingExperiment
from .encodings.locomotion_cppn import LocomotionCPPNExperiment
from .encodings.qd_locomotion_cppn import QDLocomotionCPPNExperiment
from .text_filaments.experiment import TextFilamentsExperiment

EXPERIMENTS = {
    "swim_sweep": SwimSweepExperiment,
    "worm_swim_example": WormSwimExample,
    "filament_folding": FilamentFoldingExperiment,
    "locomotion_cppn": LocomotionCPPNExperiment,
    "qd_locomotion_cppn": QDLocomotionCPPNExperiment,
    "text_filaments": TextFilamentsExperiment,
}
