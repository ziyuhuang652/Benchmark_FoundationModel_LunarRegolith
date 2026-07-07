#!/usr/bin/env python
"""
Runs a single NVT molecular dynamics simulation using a specified
foundation model force field and saves key results.

This script is designed to be called by the `run_all.sh` master script.

Key outputs for each run:
- final_structure.xyz: The structure after the MD run.
- md.traj: The full trajectory of the simulation.
- simulation_details.json: Metadata including timing and parameters.
- bond_analysis.json: Statistics on bond distances and angles.
"""
import argparse
import time
import json
import warnings
from pathlib import Path
import numpy as np
from packaging.version import Version

# ASE imports
from ase.io import read, write
from ase.md.langevin import Langevin
from ase import units, __version__ as ase_version

# Check ASE version to handle API changes for get_distances
if Version(ase_version) >= Version("3.28.0"):
    from ase.geometry import get_distances
else:
    # Fallback for older ASE versions
    from ase.atoms import Atoms
    def get_distances(atoms: Atoms, mic=False):
        return atoms.get_all_distances(mic=mic), None # Return None to match tuple format


# Suppress common warnings from calculators, especially from MACE
warnings.filterwarnings("ignore", category=UserWarning)

def get_calculator(model_name: str):
    """
    Returns the appropriate ASE calculator for the specified foundation model.
    This function assumes the script is run within a conda environment where
    the necessary packages for the selected model are installed.
    """
    print(f"Loading calculator for '{model_name}'...")
    if model_name == "mace":
        from mace.calculators import mace_mp
        return mace_mp(model="medium", device="cuda", default_dtype="float64")
    elif model_name == "upet":
        from upet.calculator import UPETCalculator
        # This model will automatically download its checkpoint file on first run
        return UPETCalculator(model="pet-omat-m", device="cuda")
    elif model_name == "mattersim":
        from mattersim.forcefield import MatterSimCalculator
        return MatterSimCalculator(device="cuda")
    elif model_name == "sevennet":
        from sevenn.calculator import SevenNetCalculator
        return SevenNetCalculator(model_name="7net-omni-v1-g5-e1", device="cuda")
    else:
        raise ValueError(f"Unknown or unsupported model: {model_name}")

def analyze_structure(atoms):
    """
    Calculates bond distances and angles for the final structure.
    """
    results = {"bond_distances": {}, "bond_angles": {}}
    
    # Use a try-except block to handle different ASE versions for calculating distances
    try:
        # Modern ASE versions (>= 3.28.0)
        from ase.geometry import get_distances
        dists_matrix, a_to_b = get_distances(atoms, mic=True)
    except (TypeError, ImportError):
        # Fallback for older ASE versions
        print("Falling back to older ASE distance calculation method.")
        dists_matrix = atoms.get_all_distances(mic=True)

    
    # --- Bond Distances ---
    # Iterate through unique pairs
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            dist = dists_matrix[i, j]
            if dist < 3.5:  # Cutoff for considering a 'bond'
                symbols = sorted((atoms.symbols[i], atoms.symbols[j]))
                bond_type = f"{symbols[0]}-{symbols[1]}"
                if bond_type not in results["bond_distances"]:
                    results["bond_distances"][bond_type] = []
                results["bond_distances"][bond_type].append(dist)

    # --- Bond Angles ---
    # This is computationally more intensive. We get angles for each atom as the center.
    for i in range(len(atoms)):
        # Find neighbors of atom i (within covalent radius + buffer)
        neighbors = [j for j, dist in enumerate(dists_matrix[i]) if 0.1 < dist < 3.0]
        
        if len(neighbors) >= 2:
            # Generate combinations of neighbors to form angles
            from itertools import combinations
            for j, k in combinations(neighbors, 2):
                angle = atoms.get_angle(j, i, k, mic=True)
                symbols = sorted((atoms.symbols[j], atoms.symbols[k]))
                angle_type = f"{symbols[0]}-{atoms.symbols[i]}-{symbols[1]}"
                if angle_type not in results["bond_angles"]:
                    results["bond_angles"][angle_type] = []
                results["bond_angles"][angle_type].append(angle)

    # --- Create JSON-serializable summary ---
    summary = {}
    for key, data_dict in results.items():
        summary[key] = {}
        for type_key, data_list in data_dict.items():
            if data_list:
                summary[key][type_key] = {
                    "mean": float(np.mean(data_list)),
                    "std": float(np.std(data_list)),
                    "min": float(np.min(data_list)),
                    "max": float(np.max(data_list)),
                    "count": len(data_list),
                }
    return summary

def main():
    parser = argparse.ArgumentParser(description="Run a foundation model MD simulation.")
    parser.add_argument("--model", required=True, choices=["mace", "upet", "mattersim", "sevennet"], help="Foundation model to use.")
    parser.add_argument("--structure", required=True, help="Path to the input structure file (e.g., .xyz, .cif).")
    args = parser.parse_args()

    # --- 1. Setup paths and load structure ---
    struct_path = Path(args.structure)
    atoms = read(struct_path)
    
    # Create a unique output directory for this run
    output_dir = Path("results") / args.model / struct_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loaded structure with {len(atoms)} atoms from {struct_path.name}.")
    print(f"Results will be saved in: {output_dir}")

    # --- 2. Setup ASE Calculator and MD Simulation ---
    try:
        calculator = get_calculator(args.model)
    except (ImportError, ValueError) as e:
        print(f"Error setting up calculator for '{args.model}': {e}")
        print("Please ensure you are in the correct conda environment and have installed the required packages.")
        return

    atoms.calc = calculator

    # NVT simulation using Langevin dynamics
    dyn = Langevin(
        atoms,
        timestep=1.0 * units.fs,
        temperature_K=300,
        friction=0.01 / units.fs,
        trajectory=str(output_dir / "md.traj"),
        logfile=str(output_dir / "md.log"),
    )

    # --- 3. Run Simulation and Track Time ---
    print("Starting NVT simulation for 1000 fs (1000 steps)...")
    start_time = time.time()
    dyn.run(1000)  # 1000 steps * 1.0 fs/step = 1000 fs
    end_time = time.time()
    
    simulation_time_s = end_time - start_time
    time_per_step_ms = (simulation_time_s / 1000) * 1000
    print(f"Simulation finished in {simulation_time_s:.2f} seconds ({time_per_step_ms:.2f} ms/step).")

    # --- 4. Save All Results ---
    print("Saving results...")
    # Save the final atomic structure
    write(str(output_dir / "final_structure.xyz"), atoms)

    # Save a JSON with all simulation details
    sim_details = {
        "model": args.model,
        "structure_file": str(struct_path),
        "num_atoms": len(atoms),
        "simulation_time_s": round(simulation_time_s, 4),
        "time_per_step_ms": round(time_per_step_ms, 4),
        "temperature_k": 300,
        "timestep_fs": 1.0,
        "num_steps": 1000,
        "total_time_fs": 1000.0,
    }
    with open(output_dir / "simulation_details.json", "w") as f:
        json.dump(sim_details, f, indent=4)

    # Perform and save the final structural analysis
    print("Performing final bond and angle analysis...")
    analysis_results = analyze_structure(atoms)
    with open(output_dir / "bond_analysis.json", "w") as f:
        json.dump(analysis_results, f, indent=4)

    print(f"All results for this run have been saved to '{output_dir}'.")

if __name__ == "__main__":
    main()
