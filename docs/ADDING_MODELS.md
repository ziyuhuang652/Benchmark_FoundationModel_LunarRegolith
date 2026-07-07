# Adding New Foundation Models

This project treats each model as an ASE calculator. To add a model, implement it in the calculator loader functions and then rerun the same structures, same MD parameters, and same analysis scripts.

## Files To Edit

Update these calculator loaders:

```text
scripts/bulk_md/run_benchmark.py
scripts/bulk_md/profile_memory.py
scripts/oh_surface/run_benchmark.py
scripts/dft_force/benchmark_dft_fm_forces.py
```

Look for functions named `get_calculator(model_name)` or `get_calculator(model_name, device)`.

## Minimal Pattern

Add a new branch like this:

```python
elif model_name == "newmodel":
    from newmodel.ase import NewModelCalculator
    return NewModelCalculator(device="cuda")
```

Also update the `argparse` choices list in each script, for example:

```python
choices=["mace", "upet", "mattersim", "sevennet", "newmodel"]
```

Then add the model to batch scripts:

```bash
MODELS=(sevennet mattersim upet mace newmodel)
```

## Rules For Fair Comparison

Use the exact same input structures:

```text
structures/balanced_xyz/*_balanced.xyz
structures/hydroxylated_stable/*_stable_surface_NBO_H.xyz
```

Use the exact same MD protocol unless the manuscript explicitly states otherwise:

```text
NVT Langevin MD
300 K
1.0 fs timestep
friction = 0.01 fs^-1
bulk benchmark = 1000 steps
surface O-H benchmark = 2000 steps, analyze last 1000 frames
```

For performance comparisons, record:

```text
simulation_time_s
time_per_step_ms
peak GPU memory from torch.cuda.max_memory_allocated()
number of atoms
GPU model and CUDA/PyTorch versions
```

## Validation Checklist

Before including a new model in the manuscript:

1. Run one smoke test on `Mg2SiO4_balanced.xyz`.
2. Confirm trajectory writes to `results/<model>/<structure>/md.traj`.
3. Confirm no atom ejection or unphysical bond explosion.
4. Run all four bulk structures.
5. Run memory profiling on all four structures.
6. For hydroxylated slabs, inspect O-H detachment using `scripts/oh_surface/analyze_oh_trajectories.py`.
7. Regenerate notebooks and figures from the new local benchmark outputs.

## Notes On Model Environments

Model packages have different CUDA/PyTorch constraints. If a single environment becomes unstable, use separate environments named after the model:

```text
newmodel_env
mace_env
upet_env
mattersim_env
sevennet_env
```

The batch scripts can be edited to activate `newmodel_env` before running the model.
