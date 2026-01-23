#!/usr/bin/env python3

import math
import numpy as np
import pandas as pd
from pathlib import Path

from run_sparc_ltg_deltax import (
    parse_rotmod_file,
    compute_deltax,
    rmse_shape,
    VARIANTS,
)

# -----------------------------
# Configuration
# -----------------------------
IN_DIR = Path("/Users/axviam/deltax_galaxy_engine/Rotmod_LTG")

# Grid ranges
B_VALUES = np.linspace(0.2, 1.4, 25)     # radial exponent
N_VALUES = np.linspace(0.1, 3.0, 25)     # feedback strength

OUT_CSV = Path("/Users/axviam/deltax_galaxy_engine/scan_onlyD_bN_results.csv")

# ONLY_D variant
ONLY_D = [v for v in VARIANTS if v.name == "only_D"][0]

# -----------------------------
# Scan
# -----------------------------
rows = []

files = sorted(IN_DIR.glob("*_rotmod.dat"))
assert len(files) > 0, "No SPARC rotmod files found."

print(f"Scanning {len(B_VALUES)} × {len(N_VALUES)} = {len(B_VALUES)*len(N_VALUES)} models")
print(f"Galaxies: {len(files)}")

for b in B_VALUES:
    for N in N_VALUES:
        rmses = []

        for fp in files:
            _, df = parse_rotmod_file(fp)
            if df.empty:
                continue

            r = df["r_kpc"].to_numpy(float)
            Vobs = df["Vobs"].to_numpy(float)
            Vgas = df["Vgas"].to_numpy(float)
            Vdisk = df["Vdisk"].to_numpy(float)
            Vbul = df["Vbul"].to_numpy(float)
            SBdisk = df["SBdisk"].to_numpy(float)

            res = compute_deltax(
                r=r,
                Vobs=Vobs,
                Vgas=Vgas,
                Vdisk=Vdisk,
                Vbul=Vbul,
                SBdisk=SBdisk,
                a=0.0,          # irrelevant for ONLY_D
                b=float(b),
                c=0.0,          # irrelevant for ONLY_D
                N=float(N),
                variant=ONLY_D,
            )

            mask = (
                np.isfinite(res["DeltaX_obs"])
                & np.isfinite(res["DeltaX_pred"])
                & (res["Vlum"] > 1e-6)
            )

            if np.count_nonzero(mask) < 3:
                continue

            rm = rmse_shape(
                res["DeltaX_pred"][mask],
                res["DeltaX_obs"][mask],
            )

            if np.isfinite(rm):
                rmses.append(float(rm))

        if len(rmses) == 0:
            continue

        rmses = np.asarray(rmses)

        rows.append({
            "b": float(b),
            "N": float(N),
            "median_rmse_shape": float(np.median(rmses)),
            "mean_rmse_shape": float(np.mean(rmses)),
            "p90_rmse_shape": float(np.percentile(rmses, 90)),
            "worst_rmse_shape": float(np.max(rmses)),
            "n_galaxies": int(len(rmses)),
        })

# -----------------------------
# Save
# -----------------------------
out_df = pd.DataFrame(rows)
out_df.to_csv(OUT_CSV, index=False)

print(f"\nWrote results to: {OUT_CSV}")
print("\nBest (lowest median shape RMSE):")
print(out_df.sort_values("median_rmse_shape").head(10))