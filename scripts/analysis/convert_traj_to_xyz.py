#!/usr/bin/env python
"""
Converts all 'md.traj' files found in the 'results' directory into
multi-frame 'trajectory.xyz' files.

This is useful for visualizing the entire simulation trajectory in software
that prefers the XYZ format, such as VMD or Ovito.
"""
import argparse
from pathlib import Path
from ase.io import read, write

def main():
    parser = argparse.ArgumentParser(
        description="Convert all md.traj files in the results directory to XYZ format."
    )
    parser.add_argument(
        "--results_dir",
        default="results",
        help="Path to the results directory to search for trajectories.",
    )
    args = parser.parse_args()

    results_path = Path(args.results_dir)
    if not results_path.is_dir():
        print(f"Error: Results directory '{results_path}' not found.")
        return

    print(f"Searching for 'md.traj' files in '{results_path}'...")

    traj_files = list(results_path.glob("**/*md.traj"))

    if not traj_files:
        print("No 'md.traj' files found.")
        return

    print(f"Found {len(traj_files)} trajectory files to convert.")

    for traj_file in traj_files:
        try:
            output_xyz_path = traj_file.parent / "trajectory.xyz"
            print(f"Converting '{traj_file}' -> '{output_xyz_path}'...")

            # Read all frames from the trajectory file
            frames = read(traj_file, index=":")

            # Write all frames to a single XYZ file
            write(output_xyz_path, frames, format="xyz")

            print(f"  Successfully converted {len(frames)} frames.")
        except Exception as e:
            print(f"  Error converting '{traj_file}': {e}")

    print("\nConversion process finished.")

if __name__ == "__main__":
    main()
