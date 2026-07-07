# Foundation Models Setup and Usage Guide

**Document Purpose**: Complete reference for setting up and using four foundation model machine learning force fields (SevenNet, MatterSim, MACE, UPET) for molecular dynamics simulations.

**Last Updated**: March 13, 2026  
**Hardware**: NVIDIA RTX 4090 (24 GB VRAM), WSL2 Ubuntu, CUDA 12.1  
**Tested**: All models validated with GPU acceleration on EMI-BF₄ ionic liquid

---

## Table of Contents

1. [Quick Reference](#quick-reference)
2. [SevenNet](#sevennet)
3. [MatterSim](#mattersim)
4. [MACE](#mace)
5. [UPET](#upet)
6. [Performance Comparison](#performance-comparison)
7. [Troubleshooting](#troubleshooting)
8. [Best Practices](#best-practices)

---

## Quick Reference

### Model Selection Guide

| Use Case | Recommended Model | Reason |
|----------|------------------|---------|
| **Production MD simulations** | UPET | Fastest (30.7s), 2nd best accuracy |
| **Maximum bond accuracy** | MatterSim | Best bond RMSD (0.0381 Å) |
| **Angular geometry studies** | SevenNet | Best angle RMSD (0.73°) |
| **General-purpose testing** | MACE | Balanced speed and accuracy |
| **Large-scale screening** | UPET | 2.7× throughput advantage |

### Quick Installation

```bash
# SevenNet
conda create -n sevennet_env python=3.10
conda activate sevennet_env
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install e3nn==0.6.0 torch-geometric sevenn

# MatterSim
conda create -n mattersim_env python=3.10
conda activate mattersim_env
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install e3nn==0.6.0 torch-geometric mattersim

# MACE
conda create -n mace_env python=3.10
conda activate mace_env
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install e3nn==0.4.4 mace-torch==0.3.15

# UPET (requires special GPU fix for WSL2)
conda create -n upet_env python=3.10
conda activate upet_env
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install upet
# See UPET section for WSL2 GPU configuration
```

---

## SevenNet

### Overview

- **Architecture**: Equivariant Graph Neural Network (GNN)
- **Model**: 7net-omni (universal pretrained model)
- **Strengths**: Best angular accuracy, explicit rotational equivariance
- **Best For**: Conformational analysis, molecular structure studies

### Environment Setup

#### Step 1: Create Conda Environment

```bash
conda create -n sevennet_env python=3.10
conda activate sevennet_env
```

#### Step 2: Install PyTorch with CUDA Support

**CRITICAL**: Use PyTorch 2.5.1+cu121. Do NOT use 2.8.0 (causes torchvision circular import errors).

```bash
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
```

#### Step 3: Install Dependencies

```bash
pip install e3nn==0.6.0
pip install torch-geometric
pip install ase  # Atomic Simulation Environment
```

#### Step 4: Install SevenNet

```bash
pip install sevenn
```

### Package Versions (Verified Working)

```
python=3.10
torch=2.5.1+cu121
torchvision=0.20.1+cu121
torchaudio=2.5.1+cu121
e3nn=0.6.0
torch-geometric=2.7.0
sevenn=0.9.3 (or latest)
ase=3.22.1 (or latest)
```

### Basic Usage

```python
from sevenn.calculator import SevenNetCalculator
from ase.io import read

# Load structure
atoms = read('structure.xyz')

# Create calculator
calc = SevenNetCalculator(
    model='7net-omni',    # Universal pretrained model
    modal='mpa',          # Message passing architecture
    device='cuda'         # GPU mode
)

# Attach calculator
atoms.calc = calc

# Get properties
energy = atoms.get_potential_energy()  # eV
forces = atoms.get_forces()            # eV/Å
```

### Molecular Dynamics Example

```python
from sevenn.calculator import SevenNetCalculator
from ase.io import read
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units

# Load structure
atoms = read('EMIBF4.xyz')

# Setup calculator
calc = SevenNetCalculator(model='7net-omni', modal='mpa', device='cuda')
atoms.calc = calc

# Initialize velocities (300 K)
MaxwellBoltzmannDistribution(atoms, temperature_K=300)

# Create MD integrator (NVT ensemble)
dyn = Langevin(
    atoms,
    timestep=0.5 * units.fs,  # 0.5 fs timestep
    temperature_K=300,        # Target temperature
    friction=0.01             # Friction coefficient
)

# Run simulation
for step in range(1000):
    dyn.run(1)
    if step % 100 == 0:
        epot = atoms.get_potential_energy()
        ekin = atoms.get_kinetic_energy()
        print(f"Step {step}: Epot={epot:.2f} eV, Ekin={ekin:.2f} eV")
```

### Performance Metrics

- **Bond RMSD**: 0.0411 Å (3rd best)
- **Angle RMSD**: 0.73° (**BEST**)
- **Computational Time**: 69.6s for 1000 MD steps
- **GPU Memory**: ~4 GB for 24-atom system

### Notes

- Excellent for angle-dependent properties (conformational analysis, molecular docking)
- Moderate computational cost (2.3× slower than UPET)
- Stable and well-documented

---

## MatterSim

### Overview

- **Architecture**: Attention-based Graph Neural Network
- **Model**: Universal pretrained model
- **Strengths**: Best bond accuracy, dynamic attention for long-range interactions
- **Best For**: Bond-critical applications, reaction coordinate mapping

### Environment Setup

#### Step 1: Create Conda Environment

```bash
conda create -n mattersim_env python=3.10
conda activate mattersim_env
```

#### Step 2: Install PyTorch with CUDA Support

```bash
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
```

#### Step 3: Install Dependencies

```bash
pip install e3nn==0.6.0
pip install torch-geometric
pip install ase
```

#### Step 4: Install MatterSim

```bash
pip install mattersim
```

### Package Versions (Verified Working)

```
python=3.10
torch=2.5.1+cu121
torchvision=0.20.1+cu121
torchaudio=2.5.1+cu121
e3nn=0.6.0
torch-geometric=2.7.0
mattersim=1.2.1
torch-ema=0.3
torchmetrics=1.9.0
ase=3.22.1 (or latest)
```

### Basic Usage

```python
from mattersim.forcefield import MatterSimCalculator
from ase.io import read

# Load structure
atoms = read('structure.xyz')

# Create calculator
calc = MatterSimCalculator(device='cuda')

# Attach calculator
atoms.calc = calc

# Get properties
energy = atoms.get_potential_energy()  # eV
forces = atoms.get_forces()            # eV/Å
```

### Molecular Dynamics Example

```python
from mattersim.forcefield import MatterSimCalculator
from ase.io import read
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units

# Load structure
atoms = read('EMIBF4.xyz')

# Setup calculator
calc = MatterSimCalculator(device='cuda')
atoms.calc = calc

# Initialize velocities
MaxwellBoltzmannDistribution(atoms, temperature_K=300)

# Create MD integrator
dyn = Langevin(
    atoms,
    timestep=0.5 * units.fs,
    temperature_K=300,
    friction=0.01
)

# Run simulation
for step in range(1000):
    dyn.run(1)
    if step % 100 == 0:
        print(f"Step {step}: E={atoms.get_potential_energy():.2f} eV")
```

### Performance Metrics

- **Bond RMSD**: 0.0381 Å (**BEST**)
- **Angle RMSD**: 0.96° (3rd)
- **Computational Time**: 81.5s for 1000 MD steps (slowest)
- **GPU Memory**: ~5 GB for 24-atom system

### Notes

- Highest bond accuracy, ideal for bond-critical applications
- Slowest of the four models (2.7× slower than UPET)
- Attention mechanism provides good long-range interactions
- Higher GPU memory usage due to attention mechanism

---

## MACE

### Overview

- **Architecture**: Atomic Cluster Expansion with equivariant message passing
- **Model**: MACE-MP (Materials Project trained)
- **Strengths**: Balanced performance, rigorous theoretical foundation
- **Best For**: General-purpose testing, interpretable predictions

### Environment Setup

#### Step 1: Create Conda Environment

```bash
conda create -n mace_env python=3.10
conda activate mace_env
```

#### Step 2: Install PyTorch with CUDA Support

```bash
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
```

#### Step 3: Install Dependencies

**CRITICAL**: MACE requires e3nn 0.4.4 (older version). Do NOT use 0.6.0.

```bash
pip install e3nn==0.4.4
pip install ase
```

#### Step 4: Install MACE

```bash
pip install mace-torch==0.3.15
```

### Package Versions (Verified Working)

```
python=3.10
torch=2.5.1+cu121
torchvision=0.20.1+cu121
torchaudio=2.5.1+cu121
e3nn=0.4.4  ⚠️ OLDER VERSION REQUIRED
mace-torch=0.3.15
torch-ema=0.3
torchmetrics=1.9.0
ase=3.22.1 (or latest)
```

### Basic Usage

```python
from mace.calculators import mace_mp
from ase.io import read

# Load structure
atoms = read('structure.xyz')

# Create calculator
calc = mace_mp(
    model="medium",   # Model size: small, medium, large
    device='cuda'     # GPU mode
)

# Attach calculator
atoms.calc = calc

# Get properties
energy = atoms.get_potential_energy()  # eV
forces = atoms.get_forces()            # eV/Å
```

### Molecular Dynamics Example

```python
from mace.calculators import mace_mp
from ase.io import read
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units

# Load structure
atoms = read('EMIBF4.xyz')

# Setup calculator
calc = mace_mp(model="medium", device='cuda')
atoms.calc = calc

# Initialize velocities
MaxwellBoltzmannDistribution(atoms, temperature_K=300)

# Create MD integrator
dyn = Langevin(
    atoms,
    timestep=0.5 * units.fs,
    temperature_K=300,
    friction=0.01
)

# Run simulation
for step in range(1000):
    dyn.run(1)
    if step % 100 == 0:
        print(f"Step {step}: E={atoms.get_potential_energy():.2f} eV")
```

### Performance Metrics

- **Bond RMSD**: 0.0419 Å (4th)
- **Angle RMSD**: 1.00° (4th)
- **Computational Time**: 49.0s for 1000 MD steps (2nd fastest)
- **GPU Memory**: ~3 GB for 24-atom system

### Notes

- **Dependency Warning**: Requires e3nn 0.4.4 (incompatible with other models using 0.6.0)
- Recommend separate conda environment to avoid conflicts
- Good balance between speed and accuracy
- Theoretically rigorous (cluster expansion framework)
- 1.6× slower than UPET but still reasonable

---

## UPET

### Overview

- **Architecture**: Equivariant Transformer
- **Model**: pet-omat-m (Open Materials trained)
- **Strengths**: **FASTEST** model, excellent accuracy, transformer-based
- **Best For**: Production MD, large-scale screening, high-throughput studies

### Environment Setup

#### Step 1: Create Conda Environment

```bash
conda create -n upet_env python=3.10
conda activate upet_env
```

#### Step 2: Install PyTorch with CUDA Support

```bash
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
```

#### Step 3: Install UPET

```bash
pip install upet
pip install ase
```

#### Step 4: WSL2 GPU Configuration (CRITICAL)

**⚠️ REQUIRED FOR GPU MODE ON WSL2**

If running on WSL2 (Windows Subsystem for Linux), UPET requires special CUDA library path configuration.

Create activation script:

```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
cat > $CONDA_PREFIX/etc/conda/activate.d/cuda_libs.sh << 'EOF'
#!/bin/bash

# Fix for UPET/vesin KernelFactory::initCudaDriver error
# 
# Root cause: vesin's libvesin.so calls dlopen("libcuda.so") which resolves to
# /lib/x86_64-linux-gnu/libcuda.so (old Linux driver stub, returns cuInit=100/no device)
# instead of /usr/lib/wsl/lib/libcuda.so (WSL2 real CUDA driver, returns cuInit=0/success)
#
# Fix: Put /usr/lib/wsl/lib FIRST in LD_LIBRARY_PATH so dlopen finds the correct driver

# WSL2 CUDA driver (MUST be first - overrides broken /lib/x86_64-linux-gnu/libcuda.so)
WSL2_CUDA_PATH="/usr/lib/wsl/lib"

# NVIDIA CUDA libraries bundled with PyTorch pip packages
CUDA_NVRTC_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib"
CUDA_RUNTIME_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cuda_runtime/lib"
CUDA_CUPTI_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cuda_cupti/lib"

export LD_LIBRARY_PATH="$WSL2_CUDA_PATH:$CUDA_NVRTC_PATH:$CUDA_RUNTIME_PATH:$CUDA_CUPTI_PATH:$LD_LIBRARY_PATH"
EOF

chmod +x $CONDA_PREFIX/etc/conda/activate.d/cuda_libs.sh
```

**Effect**: Automatic GPU configuration on `conda activate upet_env`

**Native Linux**: Skip this step if running on native Linux (not WSL2)

### Package Versions (Verified Working)

```
python=3.10
torch=2.5.1+cu121
torchvision=0.20.1+cu121
torchaudio=2.5.1+cu121
upet=0.1.2
metatensor-torch=0.8.4
metatomic-torch=0.1.11
ase=3.22.1 (or latest)
```

### Basic Usage

```python
from upet.calculator import UPETCalculator
from ase.io import read

# Load structure
atoms = read('structure.xyz')

# Create calculator
calc = UPETCalculator(
    model="pet-omat-m",     # Model name
    version="1.0.0",        # Model version
    device="cuda",          # GPU mode
    non_conservative=False  # Conservative forces
)

# Attach calculator
atoms.calc = calc

# Get properties
energy = atoms.get_potential_energy()  # eV
forces = atoms.get_forces()            # eV/Å
```

### Molecular Dynamics Example

```python
from upet.calculator import UPETCalculator
from ase.io import read
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units

# Load structure
atoms = read('EMIBF4.xyz')

# Setup calculator
calc = UPETCalculator(
    model="pet-omat-m",
    version="1.0.0",
    device="cuda",
    non_conservative=False
)
atoms.calc = calc

# Initialize velocities
MaxwellBoltzmannDistribution(atoms, temperature_K=300)

# Create MD integrator
dyn = Langevin(
    atoms,
    timestep=0.5 * units.fs,
    temperature_K=300,
    friction=0.01
)

# Run simulation
for step in range(1000):
    dyn.run(1)
    if step % 100 == 0:
        print(f"Step {step}: E={atoms.get_potential_energy():.2f} eV")
```

### Performance Metrics

- **Bond RMSD**: 0.0373 Å (2nd best)
- **Angle RMSD**: 0.85° (2nd best)
- **Computational Time**: 30.7s for 1000 MD steps (**FASTEST**)
- **GPU Memory**: ~2 GB for 24-atom system (lowest)
- **Speedup**: 2.7× faster than MatterSim, 2.3× faster than SevenNet

### Notes

- **FASTEST model** - exceptional computational efficiency
- **Second-best accuracy** in both bond and angle metrics
- **Requires WSL2 GPU fix** if running on Windows Subsystem for Linux
- **Best speed-accuracy balance** - ideal for production simulations
- Transformer architecture challenges conventional wisdom about computational costs
- Optimized metatensor-torch backend provides excellent GPU performance

---

## Performance Comparison

### Accuracy Table

| Model | Bond RMSD (Å) | Bond MAE (Å) | Angle RMSD (°) | Angle MAE (°) | Pass Threshold? |
|-------|---------------|--------------|----------------|---------------|-----------------|
| **MatterSim** | **0.0381** 🥇 | 0.0289 | 0.96 | 0.74 | ✅ Yes |
| **UPET** | **0.0373** 🥈 | 0.0284 | **0.85** 🥈 | 0.65 | ✅ Yes |
| **SevenNet** | 0.0411 | 0.0312 | **0.73** 🥇 | 0.56 | ✅ Yes |
| **MACE** | 0.0419 | 0.0319 | 1.00 | 0.76 | ✅ Yes |

**Validation Criteria**: Bond RMSD < 0.05 Å, Angle RMSD < 5.0° (all models pass)

### Speed Table

| Model | Time (1000 steps) | Relative Speed | Speedup vs UPET |
|-------|-------------------|----------------|-----------------|
| **UPET** | **30.7s** 🥇 | 1.00× (baseline) | 1.0× |
| **MACE** | 49.0s | 0.63× | 1.6× slower |
| **SevenNet** | 69.6s | 0.44× | 2.3× slower |
| **MatterSim** | 81.5s | 0.38× | 2.7× slower |

### Speed-Accuracy Pareto Frontier

**Optimal choices**:
- **Equilibrium production MD**: UPET (fastest + 2nd best accuracy)
- **Bond-critical applications**: MatterSim (best bonds, acceptable speed)
- **Angular geometry studies**: SevenNet (best angles, moderate speed)
- **General testing**: MACE (balanced)

### GPU Memory Usage (24-atom EMI-BF₄ system)

| Model | GPU Memory | Memory Rank |
|-------|------------|-------------|
| UPET | ~2 GB | Lowest |
| MACE | ~3 GB | Low |
| SevenNet | ~4 GB | Moderate |
| MatterSim | ~5 GB | Highest |

---

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: PyTorch Version Conflicts

**Symptom**: `ImportError: cannot import name 'X' from 'torchvision'`

**Cause**: Auto-upgrade to PyTorch 2.8.0 (has circular import bugs)

**Solution**: Pin PyTorch to 2.5.1+cu121
```bash
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

#### Issue 2: MACE e3nn Version Conflict

**Symptom**: `ModuleNotFoundError` or incompatibility errors with MACE

**Cause**: MACE requires e3nn 0.4.4, other models use 0.6.0

**Solution**: Use separate conda environments for MACE
```bash
conda create -n mace_env python=3.10
# Install MACE dependencies in isolated environment
```

#### Issue 3: UPET GPU Failure on WSL2

**Symptom**: 
```
KernelFactory::initCudaDriver() failed with error code 100: 'cuInit' failed
CUDA initialization failed, falling back to CPU
```

**Cause**: WSL2 has two `libcuda.so` libraries, wrong one gets loaded

**Solution**: Apply WSL2 GPU fix (see UPET section Step 4)
```bash
# Create activation script to prioritize correct CUDA driver
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
# Copy cuda_libs.sh from UPET section
```

**Verification**:
```bash
conda activate upet_env
python -c "from upet.calculator import UPETCalculator; calc = UPETCalculator(device='cuda'); print('GPU OK')"
```

#### Issue 4: CUDA Out of Memory

**Symptom**: `RuntimeError: CUDA out of memory`

**Cause**: GPU memory exhausted (usually for large systems)

**Solutions**:
1. Reduce batch size (if using multiple atoms objects)
2. Use CPU mode temporarily: `device='cpu'`
3. Use smaller model (UPET has lowest memory footprint)
4. Upgrade GPU (if possible)

#### Issue 5: Slow Neighbor List Updates

**Symptom**: Unexpectedly slow MD simulations

**Cause**: Inefficient neighbor list management

**Solutions**:
1. Verify GPU mode is active (check with `nvidia-smi` during simulation)
2. Ensure CUDA drivers are up to date
3. Check that timestep isn't too small (0.5 fs is typical)

---

## Best Practices

### Environment Management

1. **Isolate environments**: Each model in separate conda environment
   ```bash
   conda create -n sevennet_env python=3.10
   conda create -n mattersim_env python=3.10
   conda create -n mace_env python=3.10
   conda create -n upet_env python=3.10
   ```

2. **Pin critical packages**: Prevent auto-upgrades
   ```bash
   pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121
   # NOT: pip install torch torchvision (may upgrade to incompatible versions)
   ```

3. **Document dependencies**: Export environment for reproducibility
   ```bash
   conda env export > sevennet_env.yml
   ```

### GPU Optimization

1. **Verify GPU availability**:
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"GPU: {torch.cuda.get_device_name(0)}")
   ```

2. **Monitor GPU usage**:
   ```bash
   nvidia-smi -l 1  # Update every 1 second
   ```

3. **Optimize batch processing**: Process multiple structures in parallel when possible

### MD Simulation Tips

1. **Timestep selection**:
   - 0.5 fs: Good for most systems (tested in benchmark)
   - 1.0 fs: Can be used for heavier atoms, equilibrium MD
   - 0.1-0.2 fs: Required for high-frequency modes (light atoms)

2. **Temperature equilibration**:
   ```python
   # Equilibrate at 300 K before production MD
   for T in range(0, 300, 10):
       dyn = Langevin(atoms, 0.5*units.fs, temperature_K=T, friction=0.01)
       dyn.run(10)
   ```

3. **Trajectory saving**:
   ```python
   from ase.io.trajectory import Trajectory
   
   traj = Trajectory('md.traj', 'w', atoms)
   dyn.attach(traj.write, interval=10)  # Save every 10 steps
   dyn.run(10000)
   traj.close()
   ```

4. **Energy monitoring**:
   ```python
   def print_energy(a=atoms):
       epot = a.get_potential_energy()
       ekin = a.get_kinetic_energy()
       temp = ekin / (1.5 * units.kB * len(a))
       print(f"E={epot+ekin:.2f} eV, T={temp:.0f} K")
   
   dyn.attach(print_energy, interval=100)
   ```

### Model Selection Strategy

**Decision tree**:

1. **Speed critical? → UPET** (2.7× faster than alternatives)

2. **Highest bond accuracy needed? → MatterSim** (but 2.7× slower than UPET)

3. **Angular geometry focus? → SevenNet** (best angle RMSD)

4. **General testing? → MACE** (balanced, but requires e3nn 0.4.4)

5. **Reactive dynamics? → Ensemble approach** (run multiple models, analyze consensus)

### Code Organization

1. **Modular calculator setup**:
   ```python
   def get_calculator(model_name, device='cuda'):
       if model_name == 'sevennet':
           from sevenn.calculator import SevenNetCalculator
           return SevenNetCalculator(model='7net-omni', modal='mpa', device=device)
       elif model_name == 'mattersim':
           from mattersim.forcefield import MatterSimCalculator
           return MatterSimCalculator(device=device)
       elif model_name == 'mace':
           from mace.calculators import mace_mp
           return mace_mp(model="medium", device=device)
       elif model_name == 'upet':
           from upet.calculator import UPETCalculator
           return UPETCalculator(model="pet-omat-m", version="1.0.0", 
                                device=device, non_conservative=False)
       else:
           raise ValueError(f"Unknown model: {model_name}")
   
   # Usage
   calc = get_calculator('upet', device='cuda')
   atoms.calc = calc
   ```

2. **Error handling**:
   ```python
   try:
       energy = atoms.get_potential_energy()
   except Exception as e:
       print(f"Energy calculation failed: {e}")
       # Fallback to CPU or different model
       calc = get_calculator('upet', device='cpu')
       atoms.calc = calc
       energy = atoms.get_potential_energy()
   ```

### Performance Benchmarking

Template for comparing models on your system:

```python
import time
from ase.io import read
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units

atoms = read('your_structure.xyz')

models = ['sevennet', 'mattersim', 'mace', 'upet']
results = {}

for model_name in models:
    atoms_copy = atoms.copy()
    calc = get_calculator(model_name, device='cuda')
    atoms_copy.calc = calc
    
    MaxwellBoltzmannDistribution(atoms_copy, temperature_K=300)
    dyn = Langevin(atoms_copy, 0.5*units.fs, temperature_K=300, friction=0.01)
    
    start = time.time()
    dyn.run(1000)
    elapsed = time.time() - start
    
    results[model_name] = elapsed
    print(f"{model_name}: {elapsed:.1f} s")

# Find fastest
fastest = min(results, key=results.get)
print(f"\nFastest: {fastest} ({results[fastest]:.1f} s)")
```

---

## References

### Official Documentation

- **SevenNet**: https://github.com/MDIL-SNU/SevenNet
- **MatterSim**: https://github.com/microsoft/mattersim
- **MACE**: https://github.com/ACEsuit/mace
- **UPET**: https://github.com/lab-cosmo/upet
- **ASE**: https://wiki.fysik.dtu.dk/ase/

### Hardware Tested

- **GPU**: NVIDIA GeForce RTX 4090 (24 GB VRAM)
- **Driver**: NVIDIA 581.04 (WSL2)
- **CUDA**: 12.1
- **OS**: WSL2 Ubuntu 22.04
- **Python**: 3.10

### Citation

If you use this guide or the benchmark results, please cite:

```bibtex
@article{yourpaper2026,
  title={Beyond Equilibrium: Foundation Model Machine Learning Force Fields 
         Predict Divergent Fragmentation Pathways in Ionic Liquids},
  author={Your Name},
  journal={Machine Learning: Science and Technology},
  year={2026}
}
```

---

## Appendix: Complete Environment Export

### Export Current Environments

```bash
# Export all four environments
conda env export -n sevennet_env > sevennet_env.yml
conda env export -n mattersim_env > mattersim_env.yml
conda env export -n mace_env > mace_env.yml
conda env export -n upet_env > upet_env.yml
```

### Recreate Environments on New System

```bash
# Recreate from exported YAML files
conda env create -f sevennet_env.yml
conda env create -f mattersim_env.yml
conda env create -f mace_env.yml
conda env create -f upet_env.yml

# If on WSL2, apply UPET GPU fix
conda activate upet_env
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
# Copy cuda_libs.sh from UPET section
```

---

**End of Guide**

For questions or issues, please refer to the official documentation linked in the References section.
