#!/usr/bin/env python
"""Run a short ASE MD benchmark for newer foundation-model calculators."""

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


def get_calculator(model_name: str, device: str, nequip_model: str | None = None):
    if model_name == "mace_mh1":
        from mace.calculators import mace_mp

        return mace_mp(
            model="mh-1",
            head="omat_pbe",
            device=device,
            default_dtype="float32",
        )
    if model_name == "nequip_oam_l":
        if not nequip_model:
            raise ValueError("--nequip-model is required for nequip_oam_l")
        from nequip.integrations.ase import NequIPCalculator

        return NequIPCalculator.from_compiled_model(
            nequip_model,
            device=device,
            chemical_species_to_atom_type_map=True,
        )
    if model_name == "uma":
        from fairchem.core import FAIRChemCalculator, pretrained_mlip

        predictor = pretrained_mlip.get_predict_unit("uma-s-1p2", device=device)
        return FAIRChemCalculator(predictor, task_name="omat")
    raise ValueError(f"Unknown model: {model_name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["mace_mh1", "nequip_oam_l", "uma"])
    parser.add_argument("--structure", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--timestep-fs", type=float, default=1.0)
    parser.add_argument("--friction", type=float, default=0.01)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-root", default="results_extended")
    parser.add_argument("--nequip-model")
    args = parser.parse_args()

    struct_path = Path(args.structure)
    atoms = read(struct_path)
    atoms.calc = get_calculator(args.model, args.device, args.nequip_model)

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

    t0 = time.perf_counter()
    dyn.run(args.steps)
    runtime = time.perf_counter() - t0

    final_energy = float(atoms.get_potential_energy())
    final_max_force = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
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
        "estimated_time_1000_steps_s": 1000.0 * runtime / args.steps,
        "final_potential_energy_ev": final_energy,
        "final_max_force_ev_a": final_max_force,
    }
    (out_dir / "simulation_details.json").write_text(json.dumps(details, indent=2))
    print(json.dumps(details, indent=2))


if __name__ == "__main__":
    main()
