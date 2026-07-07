#!/usr/bin/env python
"""Run NVT MD for an all-surface-O hydroxylated slab with a foundation model."""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
from ase import units
from ase.io import read, write
from ase.md.langevin import Langevin

warnings.filterwarnings("ignore", category=UserWarning)


def get_calculator(model_name: str, device: str):
    print(f"Loading calculator for '{model_name}' on {device}...")
    if model_name == "mace":
        from mace.calculators import mace_mp

        return mace_mp(model="medium", device=device, default_dtype="float64")
    if model_name == "upet":
        from upet.calculator import UPETCalculator

        return UPETCalculator(model="pet-omat-m", device=device)
    if model_name == "mattersim":
        from mattersim.forcefield import MatterSimCalculator

        return MatterSimCalculator(device=device)
    if model_name == "sevennet":
        from sevenn.calculator import SevenNetCalculator

        return SevenNetCalculator(model_name="7net-omni-v1-g5-e1", device=device)
    raise ValueError(f"Unknown model: {model_name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["mace", "upet", "mattersim", "sevennet"])
    parser.add_argument("--structure", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--timestep-fs", type=float, default=1.0)
    parser.add_argument("--friction", type=float, default=0.01)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-root", default="results")
    args = parser.parse_args()

    struct_path = Path(args.structure)
    atoms = read(struct_path)
    atoms.calc = get_calculator(args.model, args.device)

    out_dir = Path(args.out_root) / args.model / struct_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    dyn = Langevin(
        atoms,
        timestep=args.timestep_fs * units.fs,
        temperature_K=args.temperature,
        friction=args.friction / units.fs,
        trajectory=str(out_dir / "md.traj"),
        logfile=str(out_dir / "md.log"),
    )

    print(f"Running {args.steps} steps for {struct_path.name} ({len(atoms)} atoms).")
    t0 = time.perf_counter()
    dyn.run(args.steps)
    runtime = time.perf_counter() - t0

    write(out_dir / "final_structure.xyz", atoms)
    details = {
        "model": args.model,
        "structure_file": str(struct_path),
        "num_atoms": len(atoms),
        "temperature_k": args.temperature,
        "timestep_fs": args.timestep_fs,
        "friction_per_fs": args.friction,
        "num_steps": args.steps,
        "total_time_fs": args.steps * args.timestep_fs,
        "simulation_time_s": runtime,
        "time_per_step_ms": 1000.0 * runtime / args.steps,
        "final_potential_energy_ev": float(atoms.get_potential_energy()),
        "final_max_force_ev_a": float(np.linalg.norm(atoms.get_forces(), axis=1).max()),
    }
    (out_dir / "simulation_details.json").write_text(json.dumps(details, indent=2))
    print(f"saved {out_dir}")
    print(f"runtime_s {runtime:.3f} ms_per_step {details['time_per_step_ms']:.3f}")


if __name__ == "__main__":
    main()

