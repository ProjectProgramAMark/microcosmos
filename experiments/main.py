import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from experiments import EXPERIMENTS

import jax
import os

# Enable persistent compilation cache to avoid recompiling CUDA kernels on each run
jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/.cache/microcosmos/jax_cache"))

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    devices = jax.devices()
    print("Available JAX devices:")
    for i, device in enumerate(devices):
        print(f"* Device {i}: {device.device_kind} (id={device.id}, platform={device.platform})")

    print(OmegaConf.to_yaml(cfg))
    hydra_cfg = HydraConfig.get()
    
    experiment_class = EXPERIMENTS[cfg.experiment.type]
    experiment = experiment_class(cfg, output_dir=Path(hydra_cfg.runtime.output_dir))
    experiment.setup()
    experiment.run()

if __name__ == "__main__":
    main()