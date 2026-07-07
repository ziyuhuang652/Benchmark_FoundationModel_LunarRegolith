#!/usr/bin/env python3
"""Memory profiler for newer foundation-model calculators."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
import warnings
from pathlib import Path

import psutil
import torch

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parents[2]
STRUCTURE_SETS = {
    "bulk_crystalline": {
        "Mg2SiO4_balanced": BASE / "structures/balanced_xyz/Mg2SiO4_balanced.xyz",
        "Fe2SiO4_balanced": BASE / "structures/balanced_xyz/Fe2SiO4_balanced.xyz",
        "TiFeO3_balanced": BASE / "structures/balanced_xyz/TiFeO3_balanced.xyz",
        "CaAl2Si2O8_balanced": BASE / "structures/balanced_xyz/CaAl2Si2O8_balanced.xyz",
    },
    "hydroxylated_stable": {
        "anorthite_1layer_stable_surface_NBO_H": BASE
        / "structures/hydroxylated_stable/anorthite_1layer_stable_surface_NBO_H.xyz",
        "fayalite_1layer_stable_surface_NBO_H": BASE
        / "structures/hydroxylated_stable/fayalite_1layer_stable_surface_NBO_H.xyz",
        "forsterite_1layer_stable_surface_NBO_H": BASE
        / "structures/hydroxylated_stable/forsterite_1layer_stable_surface_NBO_H.xyz",
        "ilmenite_1layer_stable_surface_NBO_H": BASE
        / "structures/hydroxylated_stable/ilmenite_1layer_stable_surface_NBO_H.xyz",
    },
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def cpu_ram_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2


def gpu_allocated_mb() -> float:
    return torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0


def gpu_max_allocated_mb() -> float:
    return torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0


def reset_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def gpu_info() -> dict[str, float | str]:
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        return {"name": props.name, "total_mb": props.total_memory / 1024**2}
    return {"name": "CPU only", "total_mb": 0.0}


def get_calculator(model_name: str, nequip_model: str | None):
    if model_name == "mace_mh1":
        from mace.calculators import mace_mp

        return mace_mp(model="mh-1", head="omat_pbe", device=DEVICE, default_dtype="float32")
    if model_name == "nequip_oam_l":
        if not nequip_model:
            raise ValueError("--nequip-model is required for nequip_oam_l")
        from nequip.integrations.ase import NequIPCalculator

        return NequIPCalculator.from_compiled_model(
            nequip_model,
            device=DEVICE,
            chemical_species_to_atom_type_map=True,
        )
    if model_name == "uma":
        from fairchem.core import FAIRChemCalculator, pretrained_mlip

        predictor = pretrained_mlip.get_predict_unit("uma-s-1p2", device=DEVICE)
        return FAIRChemCalculator(predictor, task_name="omat")
    raise ValueError(f"Unknown model: {model_name}")


def profile_model(model_name: str, structure_set: str, nequip_model: str | None) -> dict:
    from ase import units
    from ase.io import read
    from ase.md.langevin import Langevin

    structures = STRUCTURE_SETS[structure_set]
    results = {
        "model": model_name,
        "structure_set": structure_set,
        "device": DEVICE,
        "gpu_info": gpu_info(),
        "structures": {},
    }

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    baseline_cpu = cpu_ram_mb()
    baseline_gpu = gpu_allocated_mb()

    t0 = time.time()
    reset_peak()
    calc = get_calculator(model_name, nequip_model)
    sync()
    load_time = time.time() - t0

    after_load_cpu = cpu_ram_mb()
    after_load_gpu = gpu_allocated_mb()
    results["model_load"] = {
        "load_time_s": round(load_time, 3),
        "baseline_cpu_mb": round(baseline_cpu, 1),
        "baseline_gpu_mb": round(baseline_gpu, 1),
        "after_load_cpu_mb": round(after_load_cpu, 1),
        "after_load_gpu_mb": round(after_load_gpu, 1),
        "model_footprint_cpu_mb": round(after_load_cpu - baseline_cpu, 1),
        "model_footprint_gpu_mb": round(after_load_gpu - baseline_gpu, 1),
        "peak_gpu_mb_at_load": round(gpu_max_allocated_mb(), 1),
    }

    for struct_name, struct_path in structures.items():
        atoms = read(struct_path)
        atoms.calc = calc

        reset_peak()
        gc.collect()
        sync()
        before_step_cpu = cpu_ram_mb()
        before_step_gpu = gpu_allocated_mb()

        t_first = time.time()
        _ = atoms.get_potential_energy()
        _ = atoms.get_forces()
        sync()
        first_step_time = time.time() - t_first
        after_first_cpu = cpu_ram_mb()
        after_first_gpu = gpu_allocated_mb()
        peak_after_first = gpu_max_allocated_mb()

        reset_peak()
        dyn = Langevin(atoms, timestep=1.0 * units.fs, temperature_K=300, friction=0.01 / units.fs)
        t_run = time.time()
        dyn.run(10)
        sync()
        run_time_10 = time.time() - t_run
        peak_during_run = gpu_max_allocated_mb()
        after_run_cpu = cpu_ram_mb()
        after_run_gpu = gpu_allocated_mb()

        results["structures"][struct_name] = {
            "n_atoms": len(atoms),
            "first_step_ms": round(first_step_time * 1000, 1),
            "time_10steps_s": round(run_time_10, 3),
            "ms_per_step": round(run_time_10 / 10 * 1000, 1),
            "before_step_cpu_mb": round(before_step_cpu, 1),
            "before_step_gpu_mb": round(before_step_gpu, 1),
            "after_first_cpu_mb": round(after_first_cpu, 1),
            "after_first_gpu_mb": round(after_first_gpu, 1),
            "peak_gpu_first_step_mb": round(peak_after_first, 1),
            "peak_gpu_during_run_mb": round(peak_during_run, 1),
            "after_run_cpu_mb": round(after_run_cpu, 1),
            "after_run_gpu_mb": round(after_run_gpu, 1),
            "step_cpu_delta_mb": round(after_run_cpu - before_step_cpu, 1),
            "step_gpu_delta_mb": round(after_run_gpu - before_step_gpu, 1),
        }
        print(
            f"{model_name} {structure_set} {struct_name}: "
            f"{run_time_10:.2f}s/10 steps, peak {peak_during_run:.1f} MB"
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["mace_mh1", "nequip_oam_l", "uma"])
    parser.add_argument("--structure-set", required=True, choices=sorted(STRUCTURE_SETS))
    parser.add_argument("--out-root", default="results/memory_extended")
    parser.add_argument("--nequip-model")
    args = parser.parse_args()

    data = profile_model(args.model, args.structure_set, args.nequip_model)
    out_dir = BASE / args.out_root / args.structure_set
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.model}_memory.json"
    out_path.write_text(json.dumps(data, indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
