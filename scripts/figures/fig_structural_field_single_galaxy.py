#!/usr/bin/env python3
"""
Structural field visualization for a single galaxy.

Creates a polar (2.5D) map where radius is physical radius (kpc)
and color encodes DeltaX_pred(r).

This is a radial structural map, not a 3D reconstruction.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radial-csv", required=True,
                    help="Radial CSV for a single galaxy (e.g. CamB__full.csv)")
    ap.add_argument("--out", required=True,
                    help="Output image path (PNG)")
    ap.add_argument("--n-phi", type=int, default=200,
                    help="Number of azimuth samples for visualization")
    args = ap.parse_args()

    radial_path = Path(args.radial_csv)
    if not radial_path.exists():
        raise FileNotFoundError(radial_path)

    df = pd.read_csv(radial_path)

    if "r_kpc" not in df.columns or "DeltaX_pred" not in df.columns:
        raise ValueError("CSV must contain r_kpc and DeltaX_pred columns")

    r = df["r_kpc"].to_numpy()
    dx = df["DeltaX_pred"].to_numpy()

    # Ensure sorted by radius
    order = np.argsort(r)
    r = r[order]
    dx = dx[order]

    # Build polar grid
    phi = np.linspace(0, 2 * np.pi, args.n_phi)
    R, Phi = np.meshgrid(r, phi, indexing="ij")
    DX = np.tile(dx[:, None], (1, args.n_phi))

    # Convert to Cartesian for plotting
    X = R * np.cos(Phi)
    Y = R * np.sin(Phi)

    # Plot
    plt.figure(figsize=(6, 6))
    pcm = plt.pcolormesh(
        X, Y, DX,
        shading="auto",
        cmap="viridis"
    )
    plt.colorbar(pcm, label=r"$\Delta X_{\rm pred}(r)$")

    galaxy_name = radial_path.stem.replace("__full", "")
    plt.title(f"Radial structural field: {galaxy_name}")

    plt.xlabel("x [kpc]")
    plt.ylabel("y [kpc]")
    plt.gca().set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    plt.close()

    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()