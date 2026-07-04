# For NEAT/CPPN experiments

## Run NEAT

```bash
uv run main.py experiment=locomotion_cppn neat=<PROFILE>
```

`<PROFILE>` can be `test`, `medium`, `heavy` 

## Continue NEAT progress

Load the latest save

```bash
uv run main.py experiment=locomotion_cppn neat=<PROFILE> neat.cont=true
```

or load a previous save at a specific date-time (inside `outputs/locomotion_cppn` folder)

```bash
uv run main.py experiment=locomotion_cppn neat=<PROFILE> neat.cont=true neat.load_state_path=<DATE>/<TIME>/neat/final_state.pkl
```

## Visualize a result

Show the best genome in the Nth generation ran at a specific date-time

```bash
uv run main.py experiment=neural simulation.realtime=true neat.load_genome_path=<DATE>/<TIME>/neat/genomes/<N>.npz
```

(`realtime` can be false)
