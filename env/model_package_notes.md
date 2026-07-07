# Foundation Model Environment Notes

The original project used separate conda environments named:

| Model | Environment | Calculator import used in scripts |
|---|---|---|
| MACE-MP-0 | `mace_env` | `from mace.calculators import mace_mp` |
| UPET | `upet_env` | `from upet.calculator import UPETCalculator` |
| MatterSim | `mattersim_env` | `from mattersim.forcefield import MatterSimCalculator` |
| SevenNet-0 | `sevennet_env` | `from sevenn.calculator import SevenNetCalculator` |
| Analysis / DFT force comparison | `FM_compare` | ASE, pandas, matplotlib, PySCF/gpu4pyscf when needed |

This folder includes three kinds of environment records:

1. `*_full_export.yml`: exact exports from the current machine. These are best for reproducing the same local setup, but may contain machine-specific build constraints.
2. `*_from_history.yml`: minimal conda history exports. These are portable but incomplete because most model packages were installed with pip.
3. `*_pip_freeze.txt`: pip package records from each environment.

Recommended setup on a new CUDA workstation:

```bash
conda env create -f env/fm_models_minimal.yml
conda activate fm_models_base
python -m pip install mace-torch mattersim sevenn upet
python -m ipykernel install --user --name fm_models_base --display-name "fm_models_base"
```

If a model fails because of CUDA or PyTorch version constraints, install that model in its own environment and keep the same environment names used by the run scripts: `mace_env`, `upet_env`, `mattersim_env`, and `sevennet_env`.

Smoke-test each model after installation:

```bash
python scripts/bulk_md/run_benchmark.py --model sevennet --structure structures/balanced_xyz/Mg2SiO4_balanced.xyz
python scripts/bulk_md/run_benchmark.py --model mattersim --structure structures/balanced_xyz/Mg2SiO4_balanced.xyz
python scripts/bulk_md/run_benchmark.py --model upet --structure structures/balanced_xyz/Mg2SiO4_balanced.xyz
python scripts/bulk_md/run_benchmark.py --model mace --structure structures/balanced_xyz/Mg2SiO4_balanced.xyz
```

The scripts default to `device="cuda"` for model calculators where supported. For CPU fallback, edit the calculator loader or use the hydroxylation runner with `--device cpu`.
