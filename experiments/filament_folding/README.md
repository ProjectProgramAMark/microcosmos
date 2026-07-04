# Filament Folding

Optimizes a 1000-particle filament to match an MNIST digit via gradient descent.

```bash
cd experiments
uv run python main.py experiment=filament_folding experiment.digit=7
```

Outputs saved to `outputs/<date>/<time>/`.

uv run tensorboard --logdir outputs/filament_folding

## some history on previous modifications

Things that made a big difference:

- Instead of SDF loss function, convert target to point cloud. Use chamfer distance to calculate loss. Is much faster and stable at this scale.
- because of random init, and zero-g, the filament cannot control its rotation. Fastest solution was using PCA to find the optimal rotation of the end result and target, and take the loss over that.
  Compared PBD with stable cosserat (2025 siggraph paper) to see what the limitations are of PBD. It does not make a groundbreaking difference. Maybe slightly better, should run a comparison again. It will help with less error accumulation head to tail with the swimmers (e.g. the tail of a sine swimmer can be floppy with PBD).
- switched from random-walk to a fourrier serries initialization, with only 3 low frequency components to fold it compactly like a wiggly snake.
  Nice way to do controlled random initialization; We can dial up the randomness in the phase of the components.
- Use curriculum learning: No random init at first, and a shorter simulation. Thereafter ramp it up in stages.
- Cosine learning rate schedule with bumps back up for each new curriculum stage
- larger batch size when randomness gets higher
- taking the loss not over the final frame, but the final 3 or even 30 frames for a more stable end result and smoother loss landscape
