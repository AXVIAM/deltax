#!/usr/bin/env python3
"""
Fig 2: Representative SPARC galaxies in velocity space.

Plots:
  - Observed rotation curve V_obs(r)
  - DeltaX-predicted rotation curve V_pred(r) = DeltaX_pred * V_lum(r)

Input:
  Radial CSVs produced by run_sparc_ltg_deltax.py

Output:
  A 2x2 panel figure for selected galaxies
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_velocity_panel(ax, df, galaxy_name):
    """Plot observed vs predicted velocity for one galaxy."""
    r = df["r_kpc"].values
    v_obs = df["Vobs"].values
    v_lum = df["Vlum"].values
    dx_pred = df["DeltaX_pred"].values

    v_pred = np.sqrt(1.0 + dx_pred) * v_lum

    ax.scatter(
        r,
        v_obs,
        s=30,
        facecolors="none",
        edgecolors="black",
        linewidths=1.2,
        label="Observed",
        zorder=3,
    )

    ax.plot(
        r,
        v_pred,
        lw=2.5,
        color="black",
        linestyle="--",
        label="DeltaX prediction",
        zorder=4,
    )

    ax.set_title(galaxy_name)
    ax.set_xlabel("Radius r [kpc]")
    ax.set_ylabel("Velocity [km/s]")
    ax.grid(True, alpha=0.3)


def main():
    parser = argparse.ArgumentParser(description="Fig 2: Velocity-space examples")
    parser.add_argument(
        "--radial-dir",
        required=True,
        help="Directory containing per-galaxy radial CSV files",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output figure path (PNG)",
    )
    parser.add_argument(
        "--galaxies",
        nargs="+",
        default=["CamB", "DDO154", "NGC2403", "NGC3198"],
        help="Galaxy names to plot",
    )

    args = parser.parse_args()

    radial_dir = Path(args.radial_dir)
    out_path = Path(args.out)

    plt.style.use('grayscale')

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=False, sharey=False)
    axes = axes.flatten()

    for ax, gal in zip(axes, args.galaxies):
        csv_path = radial_dir / f"{gal}__full.csv"

        if not csv_path.exists():
            raise FileNotFoundError(f"Missing radial file: {csv_path}")

        df = pd.read_csv(csv_path)
        plot_velocity_panel(ax, df, gal)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=2,
        frameon=False,
    )

    fig.suptitle(
        "Representative SPARC galaxies in velocity space",
        fontsize=15,
        y=0.985,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.88])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()