#!/usr/bin/env python
"""Generate slabs with H attached to top-surface non-bridging oxygen atoms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import surface, supercells
from ase.io import read, write


BASE = Path(__file__).resolve().parents[2]
WORK = BASE

MINERALS = {
    "forsterite": {
        "file": "Mg2SiO4_balanced.xyz",
        "miller": (0, 1, 0),
        "supercell": np.diag([1, 1, 1]),
    },
    "fayalite": {
        "file": "Fe2SiO4_balanced.xyz",
        "miller": (0, 1, 0),
        "supercell": np.diag([1, 1, 1]),
    },
    "ilmenite": {
        "file": "TiFeO3_balanced.xyz",
        "miller": (0, 0, 1),
        "supercell": np.diag([1, 1, 1]),
    },
    "anorthite": {
        "file": "CaAl2Si2O8_balanced.xyz",
        "miller": (0, 1, 0),
        "supercell": np.diag([1, 1, 1]),
    },
}

NETWORK_FORMER_CUTOFFS_A = {
    "Si": 2.0,
    "Al": 2.2,
}


def make_slab(bulk_path: Path, miller: tuple[int, int, int], supercell: np.ndarray):
    bulk = read(bulk_path)
    slab = surface(bulk, miller, layers=1, vacuum=0.0)
    slab = supercells.make_supercell(slab, supercell)
    slab.positions[:, 2] -= slab.positions[:, 2].min()
    slab.positions[:, 2] += 1.0
    cell = slab.cell.array.copy()
    cell[2, 2] = slab.positions[:, 2].max() + 30.0
    slab.set_cell(cell)
    slab.set_pbc([True, True, True])
    return slab


def network_former_neighbors(slab, o_idx: int) -> list[dict]:
    neighbors = []
    for atom in slab:
        cutoff = NETWORK_FORMER_CUTOFFS_A.get(atom.symbol)
        if cutoff is None:
            continue
        distance = float(slab.get_distance(o_idx, atom.index, mic=True))
        if 0.1 < distance <= cutoff:
            neighbors.append(
                {
                    "index": int(atom.index),
                    "symbol": atom.symbol,
                    "distance_a": distance,
                    "cutoff_a": cutoff,
                }
            )
    return neighbors


def hydroxylate_surface_nbo(
    slab,
    surface_depth: float,
    oh_length: float,
    max_network_neighbors: int,
):
    o_indices = [atom.index for atom in slab if atom.symbol == "O"]
    if not o_indices:
        raise ValueError("No oxygen atoms found in slab.")

    o_z = np.array([slab[i].position[2] for i in o_indices])
    max_o_z = float(o_z.max())
    surface_o = [i for i in o_indices if slab[i].position[2] >= max_o_z - surface_depth]
    neighbor_map = {i: network_former_neighbors(slab, i) for i in surface_o}
    selected = [
        i
        for i in surface_o
        if len(neighbor_map[i]) <= max_network_neighbors
    ]
    selected = sorted(selected, key=lambda i: (-slab[i].position[2], i))

    h_atoms = Atoms()
    pairs = []
    for o_idx in selected:
        h_pos = slab[o_idx].position + np.array([0.0, 0.0, oh_length])
        h_idx = len(slab) + len(h_atoms)
        h_atoms += Atoms("H", positions=[h_pos])
        pairs.append(
            {
                "o_index": int(o_idx),
                "h_index": int(h_idx),
                "o_z_a": float(slab[o_idx].position[2]),
                "network_former_coordination": len(neighbor_map[o_idx]),
                "network_former_neighbors": neighbor_map[o_idx],
                "initial_oh_distance_a": float(oh_length),
            }
        )

    return slab + h_atoms, {
        "surface_depth_a": surface_depth,
        "oh_length_a": oh_length,
        "site_rule": "top_surface_non_bridging_oxygen",
        "network_former_cutoffs_a": NETWORK_FORMER_CUTOFFS_A,
        "max_network_former_neighbors": max_network_neighbors,
        "max_o_z_a": max_o_z,
        "n_top_surface_o": len(surface_o),
        "n_surface_nbo": len(selected),
        "pairs": pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minerals", nargs="+", default=list(MINERALS), choices=MINERALS)
    parser.add_argument("--surface-depth", type=float, default=1.0)
    parser.add_argument("--oh-length", type=float, default=0.98)
    parser.add_argument("--max-network-neighbors", type=int, default=1)
    parser.add_argument("--input-dir", default=str(BASE / "structures" / "balanced_xyz"))
    parser.add_argument("--out-dir", default=str(BASE / "structures" / "hydroxylated_all"))
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_meta = {}

    for mineral in args.minerals:
        params = MINERALS[mineral]
        bulk_path = input_dir / params["file"]
        slab = make_slab(bulk_path, params["miller"], params["supercell"])
        hydrated, meta = hydroxylate_surface_nbo(
            slab,
            surface_depth=args.surface_depth,
            oh_length=args.oh_length,
            max_network_neighbors=args.max_network_neighbors,
        )
        out_path = out_dir / f"{mineral}_1layer_surface_NBO_H.xyz"
        write(out_path, hydrated, format="extxyz")
        meta.update(
            {
                "mineral": mineral,
                "bulk_structure": str(bulk_path),
                "structure": str(out_path),
                "natoms_before_h": len(slab),
                "natoms_after_h": len(hydrated),
            }
        )
        all_meta[out_path.stem] = meta
        print(
            f"{mineral}: wrote {out_path} with {meta['n_surface_nbo']} surface NBO-H groups "
            f"from {meta['n_top_surface_o']} top-surface O atoms "
            f"({len(slab)} -> {len(hydrated)} atoms)"
        )

    meta_path = out_dir / "oh_site_metadata.json"
    meta_path.write_text(json.dumps(all_meta, indent=2))
    print(f"saved {meta_path}")


if __name__ == "__main__":
    main()
