# Benchmark_FoundationModel_LunarRegolith

Code and input-structure release for benchmarking universal machine-learning
interatomic potential foundation models on lunar-regolith minerals.

This release intentionally excludes pre-generated trajectories, analysis tables,
figures, and manuscript build products. The repository is meant to let users set
up the model environments, rerun the benchmark, and regenerate outputs locally.

## Models

The benchmark launcher supports the current six-model set:

- SevenNet-0
- MatterSim
- UPET / PET-OMAT-M
- MACE-MH
- UMA
- NequIP-OAM-L

## Benchmark Systems

| Mineral | Formula | Input file | Atoms |
|---|---|---|---:|
| Forsterite | Mg2SiO4 | `structures/balanced_xyz/Mg2SiO4_balanced.xyz` | 336 |
| Fayalite | Fe2SiO4 | `structures/balanced_xyz/Fe2SiO4_balanced.xyz` | 336 |
| Ilmenite | FeTiO3 | `structures/balanced_xyz/TiFeO3_balanced.xyz` | 360 |
| Anorthite | CaAl2Si2O8 | `structures/balanced_xyz/CaAl2Si2O8_balanced.xyz` | 208 |

Hydroxylated slab inputs are under `structures/hydroxylated_stable/`.

## Repository Contents

| Path | Contents |
|---|---|
| `structures/` | Bulk cells, source CIFs, generated reference cells, and hydroxylated slabs |
| `scripts/` | Core benchmark, memory profiling, OH analysis, and structure-generation scripts |
| `setup/` | Conda environment installation and import smoke-test scripts |
| `env/` | Minimal environment notes |
| `models/nequip/` | NequIP-OAM-L compiled model artifact used by the runner |
| `docs/` | Setup and extension notes |
| `*.ipynb` | Two cleaned analysis notebooks kept as reproducible workflow templates |

## Environment Setup

The recommended setup uses separate conda environments because the six model
stacks have conflicting dependencies.

```bash
cd Benchmark_FoundationModel_LunarRegolith
bash setup/setup_envs_and_smoke_test.sh
```

Environment mapping:

| Model | Conda environment |
|---|---|
| SevenNet-0 | `sevennet_env` |
| MatterSim | `mattersim_env` |
| UPET / PET-OMAT-M | `upet_env` |
| MACE-MH | `mace_latest_env` |
| UMA | `fm_latest_models` |
| NequIP-OAM-L | `fm_latest_models` |

UMA requires Hugging Face access:

```bash
export HF_TOKEN=<your_huggingface_token>
```

The NequIP-OAM-L compiled model is expected at:

```bash
models/nequip/mir-group__NequIP-OAM-L__0.1.nequip.pth
```

## Running Benchmarks

Bulk crystalline MD:

```bash
bash scripts/run_bulk_all_models.sh
```

Hydroxylated-surface MD and OH analysis:

```bash
bash scripts/run_oh_stable_all_models.sh
```

Common overrides:

```bash
MODELS="mace_mh1 uma" STEPS=1000 bash scripts/run_bulk_all_models.sh
MODELS="sevennet mattersim upet" STEPS=2000 bash scripts/run_oh_stable_all_models.sh
```

Generated outputs are intentionally ignored by Git through `.gitignore`.

## Analysis Notebooks

The cleaned notebooks retained in the repository are:

- `FM_MD_analysis_six_models_v2.ipynb`
- `Surface_NBO_OH_results_six_models_last500_v2.ipynb`

They are provided as workflow templates. Run the benchmark scripts first to
regenerate trajectories and summary tables before executing notebook cells.
