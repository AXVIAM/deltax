#!/usr/bin/env python3
"""
Fig 1: Example galaxies in DeltaX space (clean 2x2 panel).

Each panel shows one representative SPARC LTG galaxy with
DeltaX_obs(r) and DeltaX_pred(r).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_galaxy(ax, csv_path, name):
    df = pd.read_csv(csv_path)

    r = df["r_kpc"].to_numpy()
    dx_obs = df["DeltaX_obs"].to_numpy()
    dx_pred = df["DeltaX_pred"].to_numpy()

    ax.plot(r, dx_obs, "o", ms=4, alpha=0.6, color="black", markerfacecolor="none", label="Observed")
    ax.plot(r, dx_pred, lw=2, color="black", linestyle="--", label="Predicted")

    ax.set_title(name, fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.relim()
    ax.autoscale()


def main():
    parser = argparse.ArgumentParser(description="Plot example DeltaX curves (2x2)")
    parser.add_argument(
        "--radial-dir",
        required=True,
        help="Path to radial CSV folder (outputs/.../radial)",
    )
    parser.add_argument(
        "--galaxies",
        nargs=4,
        default=["CamB", "DDO154", "NGC2403", "NGC3198"],
        help="Exactly four galaxy names for the 2x2 grid",
    )
    parser.add_argument(
        "--out",
        default="fig1_deltax_examples.png",
        help="Output figure filename",
    )

    args = parser.parse_args()

    radial_dir = Path(args.radial_dir).expanduser()
    if not radial_dir.is_dir():
        raise FileNotFoundError(f"Radial directory not found: {radial_dir}")

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))

    axes = axes.flatten()

    for ax, gal in zip(axes, args.galaxies):
        csv_path = radial_dir / f"{gal}__full.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing radial CSV: {csv_path}")
        plot_galaxy(ax, csv_path, gal)

    # Shared labels
    fig.supxlabel("Radius r [kpc]")
    fig.supylabel(r"$\Delta X$")

    # Single legend (placed above panels, outside plot area)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.93),
    )

    # Suptitle
    fig.suptitle(
        r"Representative SPARC galaxies in $\Delta X$ space",
        fontsize=16,
        y=0.985,
    )

    # Layout: reserve top space for title + legend
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(args.out, dpi=300)
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()