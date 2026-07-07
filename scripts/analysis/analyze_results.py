#!/usr/bin/env python3
"""
Comprehensive analysis of MD results from four foundation models:
UPET, MACE, MatterSim, SevenNet on four lunar regolith minerals.

Generates plots for:
  1. Computing time comparison
  2. Temperature & energy evolution
  3. Key bond distances comparison
  4. Key bond angles comparison
  5. Radial distribution functions g(r)
"""

import json
import os
import re
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from itertools import combinations_with_replacement

from ase.io.trajectory import Trajectory
from ase.io import read as ase_read

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Paths & configuration
# ──────────────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parents[2]
RESULTS = BASE / "results"
OUTDIR = BASE / "analysis_plots"
OUTDIR.mkdir(exist_ok=True)

MODELS = ["sevennet", "mattersim", "upet", "mace"]
MODEL_LABELS = {"sevennet": "SevenNet", "mattersim": "MatterSim",
                "upet": "UPET", "mace": "MACE"}
MODEL_COLORS = {"sevennet": "#2196F3", "mattersim": "#4CAF50",
                "upet": "#FF9800", "mace": "#E91E63"}
MODEL_MARKERS = {"sevennet": "o", "mattersim": "s", "upet": "^", "mace": "D"}

STRUCTURES = ["Mg2SiO4_balanced", "Fe2SiO4_balanced",
              "TiFeO3_balanced", "CaAl2Si2O8_balanced"]
STRUCT_LABELS = {
    "Mg2SiO4_balanced":    "Mg₂SiO₄\n(Forsterite)",
    "Fe2SiO4_balanced":    "Fe₂SiO₄\n(Fayalite)",
    "TiFeO3_balanced":     "TiFeO₃\n(Ilmenite)",
    "CaAl2Si2O8_balanced": "CaAl₂Si₂O₈\n(Anorthite)",
}
STRUCT_LABELS_SHORT = {
    "Mg2SiO4_balanced":    "Mg₂SiO₄",
    "Fe2SiO4_balanced":    "Fe₂SiO₄",
    "TiFeO3_balanced":     "TiFeO₃",
    "CaAl2Si2O8_balanced": "CaAl₂Si₂O₈",
}

# Key bonds and angles to highlight (scientifically important)
HIGHLIGHT_BONDS = {
    "Mg2SiO4_balanced":    ["O-Si", "Mg-O"],
    "Fe2SiO4_balanced":    ["O-Si", "Fe-O"],
    "TiFeO3_balanced":     ["O-Ti", "Fe-O"],
    "CaAl2Si2O8_balanced": ["O-Si", "Al-O", "Ca-O"],
}
HIGHLIGHT_ANGLES = {
    "Mg2SiO4_balanced":    ["O-Si-O", "Mg-O-Si", "Mg-O-Mg"],
    "Fe2SiO4_balanced":    ["O-Si-O", "Fe-O-Si", "Fe-O-Fe"],
    "TiFeO3_balanced":     ["O-Ti-O", "Fe-O-Ti"],
    "CaAl2Si2O8_balanced": ["O-Si-O", "O-Al-O", "Al-O-Si"],
}

# g(r) pairs of interest per structure
GR_PAIRS = {
    "Mg2SiO4_balanced":    [("Si", "O"), ("Mg", "O"), ("O", "O")],
    "Fe2SiO4_balanced":    [("Si", "O"), ("Fe", "O"), ("O", "O")],
    "TiFeO3_balanced":     [("Ti", "O"), ("Fe", "O"), ("O", "O")],
    "CaAl2Si2O8_balanced": [("Si", "O"), ("Al", "O"), ("Ca", "O"), ("O", "O")],
}

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
})


# ──────────────────────────────────────────────────────────────────────────────
# Helper: load data
# ──────────────────────────────────────────────────────────────────────────────

def load_timing():
    """Return nested dict timing[model][struct] = (sim_time_s, num_atoms)."""
    timing = {}
    for model in MODELS:
        timing[model] = {}
        for struct in STRUCTURES:
            p = RESULTS / model / struct / "simulation_details.json"
            if p.exists():
                d = json.loads(p.read_text())
                timing[model][struct] = (d["simulation_time_s"], d["num_atoms"])
    return timing


def load_bond_analysis():
    """Return nested dict bonds[model][struct] = {distances:{...}, angles:{...}}."""
    bonds = {}
    for model in MODELS:
        bonds[model] = {}
        for struct in STRUCTURES:
            p = RESULTS / model / struct / "bond_analysis.json"
            if p.exists():
                bonds[model][struct] = json.loads(p.read_text())
    return bonds


def load_md_log(model, struct):
    """Parse md.log → arrays of time, Etot, Epot, Ekin, T."""
    p = RESULTS / model / struct / "md.log"
    if not p.exists():
        return None
    data = {"time": [], "Etot": [], "Epot": [], "Ekin": [], "T": []}
    with open(p) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("Time"):
                continue
            parts = line.split()
            if len(parts) == 5:
                try:
                    data["time"].append(float(parts[0]))
                    data["Etot"].append(float(parts[1]))
                    data["Epot"].append(float(parts[2]))
                    data["Ekin"].append(float(parts[3]))
                    data["T"].append(float(parts[4]))
                except ValueError:
                    pass
    return {k: np.array(v) for k, v in data.items()}


# ──────────────────────────────────────────────────────────────────────────────
# g(r) computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_gr(traj_path, species_a, species_b, rmax=6.0, nbins=200, stride=5):
    """
    Compute partial g(r) between species_a and species_b from an ASE trajectory.
    Uses the last half of the trajectory with periodic boundary conditions.
    """
    try:
        traj = Trajectory(str(traj_path))
    except Exception:
        return None, None

    nframes = len(traj)
    start = nframes // 2  # use last half for better statistics
    frames = list(range(start, nframes, stride))
    if len(frames) == 0:
        return None, None

    bins = np.linspace(0, rmax, nbins + 1)
    dr = bins[1] - bins[0]
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    hist = np.zeros(nbins)
    n_pairs = 0
    vol_avg = 0.0
    n_a_avg = 0.0
    n_b_avg = 0.0
    n_counted = 0

    for i in frames:
        try:
            atoms = traj[i]
        except Exception:
            continue
        cell = atoms.get_cell()
        if cell is None or np.linalg.det(cell) < 1.0:
            continue

        pos = atoms.get_positions()
        syms = np.array(atoms.get_chemical_symbols())
        idx_a = np.where(syms == species_a)[0]
        idx_b = np.where(syms == species_b)[0]

        if len(idx_a) == 0 or len(idx_b) == 0:
            return None, None

        vol_avg += atoms.get_volume()
        n_a_avg += len(idx_a)
        n_b_avg += len(idx_b)
        n_counted += 1

        cell_inv = np.linalg.inv(cell.T)
        pos_a = pos[idx_a]
        pos_b = pos[idx_b]

        # minimum image convention
        for pa in pos_a:
            dv = pos_b - pa
            # fractional coords
            df = dv @ cell_inv.T
            df -= np.round(df)
            dr_vec = df @ cell.T
            dist = np.sqrt((dr_vec**2).sum(axis=1))
            # exclude self (if same species)
            if species_a == species_b:
                dist = dist[dist > 1e-3]
            h, _ = np.histogram(dist[dist < rmax], bins=bins)
            hist += h

    if n_counted == 0 or vol_avg == 0:
        return None, None

    vol_avg /= n_counted
    n_a_avg /= n_counted
    n_b_avg /= n_counted

    rho_b = n_b_avg / vol_avg
    shell_vol = (4.0 / 3.0) * np.pi * (bins[1:]**3 - bins[:-1]**3)

    norm = n_a_avg * n_counted * rho_b * shell_vol
    # avoid double-counting for same-species
    if species_a == species_b:
        norm *= 0.5

    gr = hist / norm
    return bin_centers, gr


# ──────────────────────────────────────────────────────────────────────────────
# Plot 1: Computing time comparison
# ──────────────────────────────────────────────────────────────────────────────

def plot_computing_time(timing):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Panel (a): grouped bar chart per structure ──
    ax = axes[0]
    n_structs = len(STRUCTURES)
    n_models = len(MODELS)
    width = 0.18
    x = np.arange(n_structs)

    for j, model in enumerate(MODELS):
        times = [timing[model].get(s, (np.nan, 0))[0] for s in STRUCTURES]
        bars = ax.bar(x + j * width - (n_models - 1) * width / 2,
                      times, width=width,
                      label=MODEL_LABELS[model],
                      color=MODEL_COLORS[model], edgecolor="white", linewidth=0.5)
        for bar, t in zip(bars, times):
            if not np.isnan(t):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                        f"{t:.0f}s", ha="center", va="bottom", fontsize=7, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels([STRUCT_LABELS_SHORT[s] for s in STRUCTURES], fontsize=10)
    ax.set_ylabel("Simulation time (s)\nfor 1000 MD steps")
    ax.set_title("(a) Total computation time")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.25)
    ax.grid(axis="y", alpha=0.3)

    # ── Panel (b): time per step normalised by number of atoms ──
    ax2 = axes[1]
    for j, model in enumerate(MODELS):
        t_per_atom = []
        labels = []
        for s in STRUCTURES:
            if s in timing[model]:
                t, n = timing[model][s]
                t_per_atom.append(t / n)   # seconds / atom for 1000 steps
                labels.append(STRUCT_LABELS_SHORT[s])
        bars = ax2.bar(x + j * width - (n_models - 1) * width / 2,
                       t_per_atom, width=width,
                       label=MODEL_LABELS[model],
                       color=MODEL_COLORS[model], edgecolor="white", linewidth=0.5)

    ax2.set_xticks(x)
    ax2.set_xticklabels([STRUCT_LABELS_SHORT[s] for s in STRUCTURES], fontsize=10)
    ax2.set_ylabel("Time / atom (s/atom)\nfor 1000 MD steps")
    ax2.set_title("(b) Computation time normalised by system size")
    ax2.legend(loc="upper right", framealpha=0.9)
    ax2.set_ylim(0, ax2.get_ylim()[1] * 1.25)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Computing time comparison — four foundation models", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(OUTDIR / "01_computing_time.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 01_computing_time.png")


# ──────────────────────────────────────────────────────────────────────────────
# Plot 2: Temperature evolution
# ──────────────────────────────────────────────────────────────────────────────

def plot_temperature(logs):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=False)
    axes = axes.flatten()

    for i, struct in enumerate(STRUCTURES):
        ax = axes[i]
        for model in MODELS:
            d = logs.get((model, struct))
            if d is None:
                continue
            ax.plot(d["time"] * 1000, d["T"],   # ps → fs
                    color=MODEL_COLORS[model],
                    label=MODEL_LABELS[model],
                    linewidth=1.2, alpha=0.85)
        ax.axhline(300, color="gray", linestyle="--", linewidth=0.8, label="Target 300 K")
        ax.set_title(STRUCT_LABELS_SHORT[struct])
        ax.set_xlabel("Time (fs)")
        ax.set_ylabel("Temperature (K)")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.25)

    fig.suptitle("Temperature evolution during NVT-MD at 300 K", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTDIR / "02_temperature_evolution.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 02_temperature_evolution.png")


# ──────────────────────────────────────────────────────────────────────────────
# Plot 3: Potential energy evolution
# ──────────────────────────────────────────────────────────────────────────────

def plot_energy(logs):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()

    for i, struct in enumerate(STRUCTURES):
        ax = axes[i]
        for model in MODELS:
            d = logs.get((model, struct))
            if d is None:
                continue
            # normalise by num_atoms for comparability across models
            natoms = None
            sp = RESULTS / model / struct / "simulation_details.json"
            if sp.exists():
                natoms = json.loads(sp.read_text())["num_atoms"]
            epot = d["Epot"] / natoms if natoms else d["Epot"]
            ax.plot(d["time"] * 1000, epot,
                    color=MODEL_COLORS[model],
                    label=MODEL_LABELS[model],
                    linewidth=1.2, alpha=0.85)
        ax.set_title(STRUCT_LABELS_SHORT[struct])
        ax.set_xlabel("Time (fs)")
        ax.set_ylabel("Epot / atom (eV/atom)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    fig.suptitle("Potential energy evolution (per atom) during NVT-MD", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTDIR / "03_potential_energy.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 03_potential_energy.png")


# ──────────────────────────────────────────────────────────────────────────────
# Plot 4: Key bond distances comparison
# ──────────────────────────────────────────────────────────────────────────────

def _normalise_bond_key(key, bond_dict):
    """Try key and its reverse."""
    if key in bond_dict:
        return key
    rev = "-".join(reversed(key.split("-")))
    if rev in bond_dict:
        return rev
    return None


def plot_bond_distances(bonds):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, struct in enumerate(STRUCTURES):
        ax = axes[i]
        target_bonds = HIGHLIGHT_BONDS[struct]

        x_pos = np.arange(len(target_bonds))
        width = 0.18
        n_models = len(MODELS)

        for j, model in enumerate(MODELS):
            bd = bonds.get(model, {}).get(struct, {}).get("bond_distances", {})
            means, stds = [], []
            for bond in target_bonds:
                key = _normalise_bond_key(bond, bd)
                if key:
                    means.append(bd[key]["mean"])
                    stds.append(bd[key]["std"])
                else:
                    means.append(np.nan)
                    stds.append(0.0)

            offset = j * width - (n_models - 1) * width / 2
            bars = ax.bar(x_pos + offset, means, width=width,
                          yerr=stds, capsize=3,
                          label=MODEL_LABELS[model],
                          color=MODEL_COLORS[model],
                          edgecolor="white", linewidth=0.5,
                          error_kw={"elinewidth": 1, "ecolor": "black", "alpha": 0.6})

        ax.set_xticks(x_pos)
        ax.set_xticklabels(target_bonds, fontsize=10)
        ax.set_ylabel("Bond distance (Å)")
        ax.set_title(STRUCT_LABELS_SHORT[struct])
        ax.legend(fontsize=8, loc="best")
        ax.grid(axis="y", alpha=0.3)
        # reference lines for known values
        if struct in ("Mg2SiO4_balanced", "Fe2SiO4_balanced", "CaAl2Si2O8_balanced"):
            ax.axhline(1.63, color="red", linestyle=":", linewidth=0.8, alpha=0.6,
                       label="Si-O ideal ~1.63 Å")

    fig.suptitle("Key bond distances (mean ± std) from final structure", fontsize=13)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout()
    fig.savefig(OUTDIR / "04_bond_distances.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 04_bond_distances.png")


# ──────────────────────────────────────────────────────────────────────────────
# Plot 5: Key bond angles comparison
# ──────────────────────────────────────────────────────────────────────────────

def plot_bond_angles(bonds):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, struct in enumerate(STRUCTURES):
        ax = axes[i]
        target_angles = HIGHLIGHT_ANGLES[struct]

        x_pos = np.arange(len(target_angles))
        width = 0.18
        n_models = len(MODELS)

        for j, model in enumerate(MODELS):
            ba = bonds.get(model, {}).get(struct, {}).get("bond_angles", {})
            means, stds = [], []
            for angle in target_angles:
                key = _normalise_bond_key(angle, ba) if angle in ba else angle
                # try reverse
                if angle not in ba:
                    parts = angle.split("-")
                    rev = "-".join(reversed(parts))
                    key = rev if rev in ba else None
                else:
                    key = angle
                if key:
                    means.append(ba[key]["mean"])
                    stds.append(ba[key]["std"])
                else:
                    means.append(np.nan)
                    stds.append(0.0)

            offset = j * width - (n_models - 1) * width / 2
            ax.bar(x_pos + offset, means, width=width,
                   yerr=stds, capsize=3,
                   label=MODEL_LABELS[model],
                   color=MODEL_COLORS[model],
                   edgecolor="white", linewidth=0.5,
                   error_kw={"elinewidth": 1, "ecolor": "black", "alpha": 0.6})

        # reference line for tetrahedral angle
        ax.axhline(109.47, color="red", linestyle=":", linewidth=0.9, alpha=0.7,
                   label="Tetrahedral 109.47°")

        ax.set_xticks(x_pos)
        ax.set_xticklabels(target_angles, fontsize=9)
        ax.set_ylabel("Bond angle (°)")
        ax.set_title(STRUCT_LABELS_SHORT[struct])
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Key bond angles (mean ± std) from final structure", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTDIR / "05_bond_angles.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 05_bond_angles.png")


# ──────────────────────────────────────────────────────────────────────────────
# Plot 6: Radial distribution functions g(r)
# ──────────────────────────────────────────────────────────────────────────────

def plot_gr(struct):
    pairs = GR_PAIRS[struct]
    n_pairs = len(pairs)
    fig, axes = plt.subplots(1, n_pairs, figsize=(5 * n_pairs, 4), sharey=False)
    if n_pairs == 1:
        axes = [axes]

    print(f"    Computing g(r) for {STRUCT_LABELS_SHORT[struct]}...")

    for k, (sp_a, sp_b) in enumerate(pairs):
        ax = axes[k]
        for model in MODELS:
            traj_path = RESULTS / model / struct / "md.traj"
            if not traj_path.exists():
                continue
            r, gr = compute_gr(traj_path, sp_a, sp_b)
            if r is None:
                continue
            ax.plot(r, gr, color=MODEL_COLORS[model],
                    label=MODEL_LABELS[model],
                    linewidth=1.5, alpha=0.85)

        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)
        ax.set_xlabel("r (Å)")
        ax.set_ylabel("g(r)")
        ax.set_title(f"g(r): {sp_a}–{sp_b}")
        ax.legend(fontsize=8)
        ax.set_xlim(0.5, 6.0)
        ax.set_ylim(0, None)
        ax.grid(alpha=0.2)

    fig.suptitle(f"Radial distribution functions — {STRUCT_LABELS_SHORT[struct]}", fontsize=12)
    fig.tight_layout()
    fname = f"06_gr_{struct.split('_')[0]}.png"
    fig.savefig(OUTDIR / fname, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {fname}")


# ──────────────────────────────────────────────────────────────────────────────
# Plot 7: Model inter-comparison heatmap for key bond distances
# ──────────────────────────────────────────────────────────────────────────────

def plot_bond_heatmap(bonds):
    """
    For each structure and each highlight bond, show the spread across models.
    Rows = bond types, columns = models, colour = mean distance.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    for i, struct in enumerate(STRUCTURES):
        ax = axes[i]
        target_bonds = HIGHLIGHT_BONDS[struct]
        data = np.full((len(target_bonds), len(MODELS)), np.nan)

        for j, model in enumerate(MODELS):
            bd = bonds.get(model, {}).get(struct, {}).get("bond_distances", {})
            for ki, bond in enumerate(target_bonds):
                key = _normalise_bond_key(bond, bd)
                if key:
                    data[ki, j] = bd[key]["mean"]

        im = ax.imshow(data, aspect="auto", cmap="RdYlGn_r",
                       vmin=np.nanmin(data) - 0.05,
                       vmax=np.nanmax(data) + 0.05)
        plt.colorbar(im, ax=ax, label="Mean distance (Å)", shrink=0.85)

        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=10)
        ax.set_yticks(range(len(target_bonds)))
        ax.set_yticklabels(target_bonds, fontsize=10)
        ax.set_title(STRUCT_LABELS_SHORT[struct])

        # annotate with values
        for ki in range(len(target_bonds)):
            for j in range(len(MODELS)):
                v = data[ki, j]
                if not np.isnan(v):
                    ax.text(j, ki, f"{v:.3f}", ha="center", va="center",
                            fontsize=9, color="black")

    fig.suptitle("Mean bond distances (Å) heatmap — model comparison", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTDIR / "07_bond_distance_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 07_bond_distance_heatmap.png")


# ──────────────────────────────────────────────────────────────────────────────
# Plot 8: Summary panel — computing time vs. bond quality
# ──────────────────────────────────────────────────────────────────────────────

def plot_summary(timing, bonds):
    """
    Scatter plot: time (x) vs. Si-O bond distance deviation from ~1.63 Å (y),
    one point per model per structure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    sio_ref = 1.63  # reference Si-O bond length Å

    for model in MODELS:
        for struct in STRUCTURES:
            if struct not in timing[model]:
                continue
            t, _ = timing[model][struct]
            bd = bonds.get(model, {}).get(struct, {}).get("bond_distances", {})
            # get Si-O or O-Si
            key = _normalise_bond_key("O-Si", bd) or _normalise_bond_key("Si-O", bd)
            if key is None:
                # use first available bond for this struct
                # (for TiFeO3 there is no Si-O)
                continue
            dev = abs(bd[key]["mean"] - sio_ref)
            ax.scatter(t, dev, color=MODEL_COLORS[model],
                       marker=MODEL_MARKERS[model], s=90,
                       label=MODEL_LABELS[model], zorder=5)
            ax.annotate(STRUCT_LABELS_SHORT[struct],
                        (t, dev), textcoords="offset points",
                        xytext=(5, 4), fontsize=7.5, color="gray")

    # deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    ax.legend(seen.values(), seen.keys(), fontsize=9)

    ax.set_xlabel("Total simulation time (s) for 1000 steps")
    ax.set_ylabel("|Si-O mean distance − 1.63 Å| (Å)\n(lower = closer to reference)")
    ax.set_title("Speed vs. Si-O bond accuracy across models and structures")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTDIR / "08_speed_vs_accuracy.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 08_speed_vs_accuracy.png")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("Loading data ...")
    timing = load_timing()
    bonds = load_bond_analysis()

    logs = {}
    for model in MODELS:
        for struct in STRUCTURES:
            d = load_md_log(model, struct)
            if d is not None:
                logs[(model, struct)] = d

    print("\nGenerating plots ...")

    print(" [1/8] Computing time comparison ...")
    plot_computing_time(timing)

    print(" [2/8] Temperature evolution ...")
    plot_temperature(logs)

    print(" [3/8] Potential energy evolution ...")
    plot_energy(logs)

    print(" [4/8] Bond distances ...")
    plot_bond_distances(bonds)

    print(" [5/8] Bond angles ...")
    plot_bond_angles(bonds)

    print(" [6/8] Radial distribution functions ...")
    for struct in STRUCTURES:
        plot_gr(struct)

    print(" [7/8] Bond distance heatmap ...")
    plot_bond_heatmap(bonds)

    print(" [8/8] Speed vs. accuracy summary ...")
    plot_summary(timing, bonds)

    print(f"\nAll plots saved to: {OUTDIR}")


if __name__ == "__main__":
    main()
