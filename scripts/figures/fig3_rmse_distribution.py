#!/usr/bin/env python3
"""
Fig 3: Population distribution of DeltaX shape RMSE across SPARC LTGs.

This script reads the per-galaxy summary CSV produced by run_sparc_ltg_deltax.py
and visualizes the distribution of rmse_DeltaX_shape across the full sample.

Outputs a histogram and an optional cumulative distribution function (CDF).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    ap = argparse.ArgumentParser(description="Fig 3: RMSE distribution")
    ap.add_argument(
        "--summary-csv",
        required=True,
        help="Path to summary_per_galaxy CSV file or directory containing such files"
    )
    ap.add_argument(
        "--out",
        required=True,
        help="Output figure filename (PNG or PDF)"
    )
    return ap.parse_args()


def main():
    args = parse_args()

    summary_path = Path(args.summary_csv)
    if summary_path.is_dir():
        matches = list(summary_path.glob("summary_per_galaxy*.csv"))
        if len(matches) == 0:
            raise FileNotFoundError(f"No files matching 'summary_per_galaxy*.csv' found in directory {summary_path}")
        elif len(matches) > 1:
            # Select the file with the longest filename
            selected = max(matches, key=lambda p: len(p.name))
            print(f"Multiple summary CSV files found in directory. Using the most specific one: {selected.name}")
            summary_path = selected
        else:
            summary_path = matches[0]
    elif not summary_path.is_file():
        raise FileNotFoundError(f"Summary CSV file not found: {summary_path}")

    df = pd.read_csv(summary_path)

    if "rmse_DeltaX_shape" not in df.columns:
        raise ValueError(
            "Expected column 'rmse_DeltaX_shape' not found in summary CSV."
        )

    rmse = df["rmse_DeltaX_shape"].to_numpy()
    rmse = rmse[np.isfinite(rmse)]

    if len(rmse) == 0:
        raise RuntimeError("No valid RMSE values found.")

    fig, ax = plt.subplots(figsize=(7, 5))

    rmse_sorted = np.sort(rmse)
    cdf = np.arange(1, len(rmse_sorted) + 1) / len(rmse_sorted)
    ax.plot(rmse_sorted, cdf, color="black", lw=2)
    ax.set_ylabel("Cumulative fraction")

    p50 = np.percentile(rmse, 50)
    p68 = np.percentile(rmse, 68)
    p90 = np.percentile(rmse, 90)

    ax.text(
        0.98,
        0.60,
        f"Median = {p50:.2f}",
        transform=ax.transAxes,
        fontsize=10,
        color="black",
        ha="right",
        va="center"
    )

    for p, label in [(p50, "50%"), (p68, "68%"), (p90, "90%")]:
        ax.axvline(p, color="black", linestyle="--", linewidth=1)
        ax.text(
            p,
            0.05,
            label,
            rotation=90,
            va="bottom",
            ha="right",
            fontsize=10,
            color="black"
        )

    ax.set_xlabel("RMSE (DeltaX shape)")
    ax.set_title("Cumulative distribution of DeltaX shape RMSE across SPARC LTGs")

    ax.set_ylim(0, 1)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    plt.close(fig)

    print(f"Wrote: {args.out}")
    print(f"Galaxies included: {len(rmse)}")
    print(f"Median RMSE: {np.median(rmse):.3f}")
    print(f"90th percentile RMSE: {np.percentile(rmse, 90):.3f}")


if __name__ == "__main__":
    main()
