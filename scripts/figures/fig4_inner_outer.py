#!/usr/bin/env python3
"""
Fig 4: Inner vs outer radius contribution.

Plots per-galaxy DeltaX shape RMSE measured on inner radii
versus outer radii.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--summary-csv",
        required=True,
        help="Path to summary_per_galaxy CSV with inner/outer RMSE columns",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="Output figure path (PNG)",
    )
    args = ap.parse_args()

    summary_path = Path(args.summary_csv)
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)

    df = pd.read_csv(summary_path)

    required_cols = [
        "rmse_DeltaX_shape_inner",
        "rmse_DeltaX_shape_outer",
    ]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    x = df["rmse_DeltaX_shape_inner"].to_numpy(float)
    y = df["rmse_DeltaX_shape_outer"].to_numpy(float)

    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]

    fig, ax = plt.subplots(figsize=(6.5, 6.0))

    ax.scatter(
        x,
        y,
        s=28,
        facecolor="0.3",
        edgecolor="black",
        linewidth=0.3,
        alpha=0.8,
    )

    # 1:1 reference line
    lim = max(x.max(), y.max()) * 1.05
    ax.plot([0, lim], [0, lim], linestyle="--", color="black", lw=1.0)
    ax.text(
        0.05 * lim,
        0.92 * lim,
        "Dashed line: equal inner/outer contribution",
        fontsize=9,
        ha="left",
        va="top",
        color="black",
    )

    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)

    ax.set_xlabel("Inner-region RMSE (DeltaX shape)")
    ax.set_ylabel("Outer-region RMSE (DeltaX shape)")

    ax.set_title("Inner vs outer radius contribution to DeltaX shape error")

    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)

    plt.tight_layout()
    outpath = Path(args.out)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=200)
    plt.close()

    print(f"Wrote: {outpath}")


if __name__ == "__main__":
    main()