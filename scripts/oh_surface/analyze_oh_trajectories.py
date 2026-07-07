#!/usr/bin/env python
"""Track initial O-H pairs through trajectories and plot distance distributions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase.io import read


WORK = Path(__file__).resolve().parents[2]


def load_metadata(path: Path) -> dict:
    return json.loads(path.read_text())


def distance_mic(atoms, i: int, j: int) -> float:
    return float(atoms.get_distance(i, j, mic=True))


def analyze_run(
    traj_path: Path,
    meta: dict,
    timestep_fs: float,
    detach_cutoff: float,
    last_n_frames: int | None,
):
    frames = read(traj_path, ":")
    frame_offset = 0
    if last_n_frames is not None and last_n_frames > 0 and len(frames) > last_n_frames:
        frame_offset = len(frames) - last_n_frames
        frames = frames[frame_offset:]
    rows = []
    pair_rows = []
    pairs = meta["pairs"]

    for local_frame_i, atoms in enumerate(frames):
        frame_i = frame_offset + local_frame_i
        time_fs = frame_i * timestep_fs
        for pair_id, pair in enumerate(pairs):
            d = distance_mic(atoms, pair["o_index"], pair["h_index"])
            rows.append(
                {
                    "frame": frame_i,
                    "time_fs": time_fs,
                    "pair_id": pair_id,
                    "o_index": pair["o_index"],
                    "h_index": pair["h_index"],
                    "network_former_coordination": pair.get("network_former_coordination"),
                    "oh_distance_a": d,
                    "detached": d > detach_cutoff,
                }
            )

    df = pd.DataFrame(rows)
    for pair_id, group in df.groupby("pair_id"):
        valid = group[~group["detached"]]
        pair_rows.append(
            {
                "pair_id": pair_id,
                "o_index": int(group["o_index"].iloc[0]),
                "h_index": int(group["h_index"].iloc[0]),
                "network_former_coordination": (
                    int(group["network_former_coordination"].iloc[0])
                    if "network_former_coordination" in group
                    and not pd.isna(group["network_former_coordination"].iloc[0])
                    else None
                ),
                "n_frames": int(len(group)),
                "n_valid_frames": int(len(valid)),
                "detached_fraction": float(group["detached"].mean()),
                "first_detached_frame": (
                    int(group.loc[group["detached"], "frame"].iloc[0])
                    if group["detached"].any()
                    else None
                ),
                "mean_oh_distance_valid_a": float(valid["oh_distance_a"].mean()) if len(valid) else np.nan,
                "std_oh_distance_valid_a": float(valid["oh_distance_a"].std()) if len(valid) > 1 else 0.0,
                "max_oh_distance_a": float(group["oh_distance_a"].max()),
            }
        )

    return df, pd.DataFrame(pair_rows)


def plot_distribution(df: pd.DataFrame, out_path: Path, title: str, detach_cutoff: float) -> None:
    valid = df[~df["detached"]]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    if not valid.empty:
        ax.hist(valid["oh_distance_a"], bins=40, density=True, color="#3b6f8f", alpha=0.82)
    ax.axvline(detach_cutoff, color="#8f3b3b", linestyle="--", linewidth=1.2, label="detach cutoff")
    ax.set_xlabel("Tracked O-H distance (A)")
    ax.set_ylabel("Probability density")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", nargs="?", default=str(WORK / "results"))
    parser.add_argument("--metadata", default=str(WORK / "structures" / "oh_site_metadata.json"))
    parser.add_argument("--analysis-dir", default=str(WORK / "analysis"))
    parser.add_argument("--figures-dir", default=str(WORK / "figures"))
    parser.add_argument("--detach-cutoff", type=float, default=1.25)
    parser.add_argument("--timestep-fs", type=float, default=1.0)
    parser.add_argument("--last-n-frames", type=int, default=1000)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    metadata = load_metadata(Path(args.metadata))
    analysis_dir = Path(args.analysis_dir)
    figures_dir = Path(args.figures_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    all_pair = []
    all_frame = []
    for traj in sorted(results_dir.glob("*/*/md.traj")):
        model = traj.parent.parent.name
        structure = traj.parent.name
        if structure not in metadata:
            print(f"skip {traj}: no metadata for {structure}")
            continue
        df, pair_df = analyze_run(
            traj,
            metadata[structure],
            timestep_fs=args.timestep_fs,
            detach_cutoff=args.detach_cutoff,
            last_n_frames=args.last_n_frames,
        )
        df.insert(0, "model", model)
        df.insert(1, "structure", structure)
        pair_df.insert(0, "model", model)
        pair_df.insert(1, "structure", structure)
        all_frame.append(df)
        all_pair.append(pair_df)

        run_csv = analysis_dir / f"{model}_{structure}_oh_timeseries.csv"
        pair_csv = analysis_dir / f"{model}_{structure}_oh_pair_summary.csv"
        df.to_csv(run_csv, index=False)
        pair_df.to_csv(pair_csv, index=False)
        plot_distribution(
            df,
            figures_dir / f"{model}_{structure}_oh_distance_distribution.png",
            f"{model}: {structure}",
            args.detach_cutoff,
        )
        print(
            f"{model}/{structure}: pairs={len(pair_df)} "
            f"detached_pairs={(pair_df['detached_fraction'] > 0).sum()} "
            f"valid_mean={df.loc[~df['detached'], 'oh_distance_a'].mean():.3f} A"
        )

    if not all_frame:
        print("No analyzable trajectories found.")
        return

    frame_df = pd.concat(all_frame, ignore_index=True)
    pair_df = pd.concat(all_pair, ignore_index=True)
    frame_df.to_csv(analysis_dir / "all_oh_timeseries.csv", index=False)
    pair_df.to_csv(analysis_dir / "all_oh_pair_summary.csv", index=False)

    summary = (
        pair_df.groupby(["model", "structure"])
        .agg(
            n_pairs=("pair_id", "count"),
            n_detached_pairs=("detached_fraction", lambda x: int((x > 0).sum())),
            mean_detached_fraction=("detached_fraction", "mean"),
            mean_oh_distance_valid_a=("mean_oh_distance_valid_a", "mean"),
            std_oh_distance_valid_a=("mean_oh_distance_valid_a", "std"),
            max_oh_distance_a=("max_oh_distance_a", "max"),
        )
        .reset_index()
    )
    summary.to_csv(analysis_dir / "oh_summary.csv", index=False)
    plot_summary(summary, figures_dir)
    plot_combined_distributions(frame_df, figures_dir)
    print(f"saved {analysis_dir / 'oh_summary.csv'}")


def plot_summary(summary: pd.DataFrame, figures_dir: Path) -> None:
    if summary.empty:
        return
    plot_df = summary.sort_values(["structure", "model"]).copy()
    plot_df["label"] = plot_df["structure"].str.replace("_1layer_surface_NBO_H", "", regex=False)
    plot_df["label"] = plot_df["label"] + "\n" + plot_df["model"]
    fig, ax = plt.subplots(figsize=(max(8.0, 0.45 * len(plot_df)), 4.2))
    ax.bar(np.arange(len(plot_df)), plot_df["mean_oh_distance_valid_a"], color="#3b6f8f")
    ax.set_xticks(np.arange(len(plot_df)))
    ax.set_xticklabels(plot_df["label"], rotation=45, ha="right")
    ax.set_ylabel("Mean tracked O-H distance, last 1000 frames (A)")
    ax.set_title("Surface NBO-H bond distances")
    fig.tight_layout()
    path = figures_dir / "summary_mean_oh_distance_last1000.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"saved {path}")

    fig, ax = plt.subplots(figsize=(max(8.0, 0.45 * len(plot_df)), 4.2))
    ax.bar(np.arange(len(plot_df)), plot_df["n_detached_pairs"], color="#8f3b3b")
    ax.set_xticks(np.arange(len(plot_df)))
    ax.set_xticklabels(plot_df["label"], rotation=45, ha="right")
    ax.set_ylabel("Number of detached initial O-H pairs")
    ax.set_title("Detached O-H pairs excluded from filtered statistics")
    fig.tight_layout()
    path = figures_dir / "summary_detached_pairs_last1000.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"saved {path}")


def plot_combined_distributions(frame_df: pd.DataFrame, figures_dir: Path) -> None:
    valid = frame_df[~frame_df["detached"]].copy()
    if valid.empty:
        return
    minerals = sorted(valid["structure"].unique())
    fig, axes = plt.subplots(len(minerals), 1, figsize=(7.0, 2.4 * len(minerals)), sharex=True)
    if len(minerals) == 1:
        axes = [axes]
    for ax, structure in zip(axes, minerals):
        sub = valid[valid["structure"] == structure]
        for model, model_df in sub.groupby("model"):
            ax.hist(
                model_df["oh_distance_a"],
                bins=45,
                density=True,
                histtype="step",
                linewidth=1.4,
                label=model,
            )
        ax.set_ylabel("Density")
        ax.set_title(structure.replace("_1layer_surface_NBO_H", ""))
        ax.legend(frameon=False, ncol=4, fontsize=8)
    axes[-1].set_xlabel("Tracked O-H distance (A)")
    fig.tight_layout()
    path = figures_dir / "combined_oh_distance_distributions_last1000.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
