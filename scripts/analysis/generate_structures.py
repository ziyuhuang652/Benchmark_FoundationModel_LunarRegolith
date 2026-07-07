#!/usr/bin/env python
"""Generate lunar mineral crystal structures using experimental crystallographic data."""

import numpy as np
from ase.spacegroup import crystal
from ase.io import read, write
from ase.build import make_supercell
import urllib.request
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "structures")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_forsterite():
    """Mg2SiO4 - Pnma (#62) - Hazen (1976) Am. Mineral. 61, 1280-1293"""
    return crystal(
        symbols=['Mg', 'Mg', 'Si', 'O', 'O', 'O'],
        basis=[
            [0.0, 0.0, 0.0],
            [0.9926, 0.25, 0.2774],
            [0.4268, 0.25, 0.0942],
            [0.7661, 0.25, 0.0142],
            [0.2199, 0.25, 0.4489],
            [0.2774, 0.1626, 0.2282],
        ],
        spacegroup=62,
        cellpar=[4.7534, 10.1902, 5.9783, 90, 90, 90],
    )


def create_fayalite():
    """Fe2SiO4 - Pnma (#62) - Fujino et al. (1981) Acta Cryst. B37, 513-518"""
    return crystal(
        symbols=['Fe', 'Fe', 'Si', 'O', 'O', 'O'],
        basis=[
            [0.0, 0.0, 0.0],
            [0.9762, 0.25, 0.2810],
            [0.4269, 0.25, 0.0942],
            [0.7663, 0.25, 0.0148],
            [0.2193, 0.25, 0.4484],
            [0.2840, 0.1626, 0.2290],
        ],
        spacegroup=62,
        cellpar=[4.8195, 10.4788, 6.0873, 90, 90, 90],
    )


def create_ilmenite():
    """FeTiO3 - R-3 (#148) - Wechsler & Prewitt (1984) Am. Mineral. 69, 176-185"""
    return crystal(
        symbols=['Fe', 'Ti', 'O'],
        basis=[
            [0.0, 0.0, 0.1440],
            [0.0, 0.0, 0.3557],
            [0.3169, 0.0, 0.2440],
        ],
        spacegroup=148,
        cellpar=[5.0884, 5.0884, 14.0855, 90, 90, 120],
    )


def create_anorthite():
    """CaAl2Si2O8 - P-1 (#2) - COD:1000034 Wainwright & Starkey (1971)"""
    cif_path = os.path.join(OUTPUT_DIR, "anorthite_cod1000034.cif")
    if not os.path.exists(cif_path):
        url = "https://www.crystallography.net/cod/1000034.cif"
        print(f"Downloading anorthite CIF from COD...")
        urllib.request.urlretrieve(url, cif_path)
    return read(cif_path)


def make_balanced_supercell(atoms, target_length=15.0):
    """Create supercell with all dimensions >= target_length Angstroms."""
    cell = atoms.get_cell()
    lengths = np.linalg.norm(cell, axis=1)
    multipliers = np.maximum(2, np.ceil(target_length / lengths)).astype(int)
    P = np.diag(multipliers)
    supercell = make_supercell(atoms, P)
    return supercell, multipliers


def main():
    minerals = {
        'forsterite_Mg2SiO4': create_forsterite,
        'fayalite_Fe2SiO4': create_fayalite,
        'ilmenite_FeTiO3': create_ilmenite,
        'anorthite_CaAl2Si2O8': create_anorthite,
    }
    
    print("=" * 60)
    print("Lunar Mineral Structure Generation")
    print("=" * 60)
    
    for name, creator in minerals.items():
        print(f"\n{name}")
        print("-" * 40)
        
        unit_cell = creator()
        n_atoms_unit = len(unit_cell)
        cell = unit_cell.get_cell()
        lengths = np.linalg.norm(cell, axis=1)
        
        print(f"Unit cell: {n_atoms_unit} atoms")
        print(f"Dimensions: {lengths[0]:.3f} x {lengths[1]:.3f} x {lengths[2]:.3f} A")
        
        unit_file = os.path.join(OUTPUT_DIR, f"{name}_unit.xyz")
        write(unit_file, unit_cell)
        
        supercell, multipliers = make_balanced_supercell(unit_cell, target_length=15.0)
        n_atoms_super = len(supercell)
        super_cell = supercell.get_cell()
        super_lengths = np.linalg.norm(super_cell, axis=1)
        
        print(f"Supercell: {multipliers[0]}x{multipliers[1]}x{multipliers[2]} = {n_atoms_super} atoms")
        print(f"Dimensions: {super_lengths[0]:.3f} x {super_lengths[1]:.3f} x {super_lengths[2]:.3f} A")
        
        super_file = os.path.join(OUTPUT_DIR, f"{name}_supercell.xyz")
        write(super_file, supercell)
        
        cif_file = os.path.join(OUTPUT_DIR, f"{name}_supercell.cif")
        write(cif_file, supercell)
    
    print("\n" + "=" * 60)
    print("Structure generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
