#!/usr/bin/env python3
"""
Fig 5: Worst-case galaxies by DeltaX shape RMSE.

This script identifies the N galaxies with the largest DeltaX shape RMSE
and outputs:
  1) a CSV table for the paper / appendix
  2) optionally, a rendered table figure (PNG)

Usage:
  python scripts/fig5_worst_cases_table.py \
    --summary-csv path/to/summary_per_galaxy.csv \
    --out-csv figures/fig5_worst_cases.csv \
    --top-n 10 \
    [--out-fig figures/fig5_worst_cases.png]
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-csv", required=True, help="Per-galaxy summary CSV")
    ap.add_argument("--out-csv", required=True, help="Output CSV for worst cases")
    ap.add_argument("--top-n", type=int, default=10, help="Number of worst galaxies")
    ap.add_argument(
        "--out-fig",
        default=None,
        help="Optional PNG path to render table as a figure",
    )
    return ap.parse_args()


def render_table(df, out_fig):
    fig, ax = plt.subplots(figsize=(10, 0.6 * len(df) + 1.5))
    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)

    fig.tight_layout()
    fig.savefig(out_fig, dpi=300)
    plt.close(fig)


def main():
    args = parse_args()

    summary_path = Path(args.summary_csv)
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)

    df = pd.read_csv(summary_path)

    if "rmse_DeltaX_shape" not in df.columns:
        raise ValueError("Column 'rmse_DeltaX_shape' not found in summary CSV")

    cols_keep = [
        c
        for c in [
            "galaxy",
            "rmse_DeltaX_shape",
            "rmse_DeltaX_abs",
            "n_points",
            "r_max_kpc",
        ]
        if c in df.columns
    ]

    worst = (
        df.sort_values("rmse_DeltaX_shape", ascending=False)
        .head(args.top_n)[cols_keep]
        .reset_index(drop=True)
    )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    worst.to_csv(out_csv, index=False)

    print(f"Wrote worst-case table CSV: {out_csv}")

    if args.out_fig:
        out_fig = Path(args.out_fig)
        render_table(worst, out_fig)
        print(f"Wrote worst-case table figure: {out_fig}")


if __name__ == "__main__":
    main()
