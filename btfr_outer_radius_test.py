#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

G_KPC_KMS2_PER_MSUN = 4.30091e-6  # kpc (km/s)^2 / Msun


def clamp_min(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.maximum(x, eps)


def parse_rotmod_file(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = re.split(r"\s+", line)
            if len(parts) < 8:
                continue
            try:
                vals = [float(p) for p in parts[:8]]
            except ValueError:
                continue
            rows.append(vals)

    cols = ["r_kpc", "Vobs", "errV", "Vgas", "Vdisk", "Vbul", "SBdisk", "SBbul"]
    df = pd.DataFrame(rows, columns=cols)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["r_kpc", "Vobs", "Vdisk", "SBdisk"])
    df = df.sort_values("r_kpc").reset_index(drop=True)
    return df


def deltax_onlyD_predict(
    r_kpc: np.ndarray,
    Vobs: np.ndarray,
    Vgas: np.ndarray,
    Vdisk: np.ndarray,
    Vbul: np.ndarray,
    b: float,
    N: float,
    K: float = math.pi,
) -> tuple[np.ndarray, np.ndarray]:
    """
    D-only version consistent with your LTG script:
      - Vlum = sqrt(Vgas^2 + Vdisk^2 + Vbul^2)
      - DeltaX_obs = Vobs / Vlum
      - D = r; Dhat = D/median(D)
      - B = K * Dhat^b / (1 + ln(1 + Dhat))
      - DeltaX_pred = (-1 + sqrt(1 + 4 N B)) / (2 N)
    """
    Vlum = np.sqrt(clamp_min(Vgas**2 + Vdisk**2 + Vbul**2))
    DeltaX_obs = Vobs / clamp_min(Vlum)

    D = r_kpc
    D0 = float(np.median(D[np.isfinite(D)])) if np.any(np.isfinite(D)) else 1.0
    if not np.isfinite(D0) or D0 <= 0:
        D0 = 1.0
    Dhat = D / D0

    D_term = clamp_min(Dhat)
    mdi = clamp_min(D_term)  # only D remains
    denom = 1.0 + np.log1p(mdi)
    B = K * (D_term**b) / clamp_min(denom)

    if abs(N) < 1e-15:
        DeltaX_pred = B.copy()
    else:
        inside = clamp_min(1.0 + 4.0 * N * B)
        DeltaX_pred = (-1.0 + np.sqrt(inside)) / (2.0 * N)

    return DeltaX_obs, DeltaX_pred, Vlum


def pick_outer_index(r: np.ndarray, mode: str = "last3_median") -> int:
    n = r.size
    if n == 0:
        return -1
    if mode == "last":
        return n - 1
    # last3_median: choose the index whose r is closest to median of last 3 radii
    k = min(3, n)
    tail = r[-k:]
    target = float(np.median(tail))
    idx = int(np.argmin(np.abs(r - target)))
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, help="Folder containing *_rotmod.dat files")
    ap.add_argument("--out-csv", required=True, help="Where to write per-galaxy outer-radius summary CSV")
    ap.add_argument("--b", type=float, default=0.5)
    ap.add_argument("--N", type=float, default=1.3)
    ap.add_argument("--K", type=float, default=math.pi)
    ap.add_argument("--outer-mode", choices=["last", "last3_median"], default="last3_median")
    args = ap.parse_args()

    in_dir = Path(args.in_dir).expanduser()
    files = sorted(in_dir.glob("*_rotmod.dat"))
    if not files:
        raise FileNotFoundError(f"No *_rotmod.dat files found in {in_dir}")

    rows = []
    for fp in files:
        gal = fp.name.replace("_rotmod.dat", "")
        df = parse_rotmod_file(fp)
        if df.empty:
            continue

        r = df["r_kpc"].to_numpy(float)
        Vobs = df["Vobs"].to_numpy(float)
        Vgas = df["Vgas"].to_numpy(float)
        Vdisk = df["Vdisk"].to_numpy(float)
        Vbul = df["Vbul"].to_numpy(float)

        # compute mapping
        dx_obs, dx_pred, Vlum = deltax_onlyD_predict(
            r_kpc=r, Vobs=Vobs, Vgas=Vgas, Vdisk=Vdisk, Vbul=Vbul,
            b=float(args.b), N=float(args.N), K=float(args.K)
        )

        m = np.isfinite(r) & np.isfinite(Vobs) & np.isfinite(Vlum) & (Vlum > 1e-6) & np.isfinite(dx_pred)
        if np.count_nonzero(m) < 3:
            continue

        r_m = r[m]; Vobs_m = Vobs[m]; Vlum_m = Vlum[m]; dx_pred_m = dx_pred[m]
        i_out = pick_outer_index(r_m, mode=str(args.outer_mode))
        if i_out < 0:
            continue

        r_out = float(r_m[i_out])
        Vf_obs = float(Vobs_m[i_out])
        Vf_pred = float(dx_pred_m[i_out] * Vlum_m[i_out])

        # optional baryonic enclosed proxy at r_out
        Menc_lum = float((Vlum_m[i_out] ** 2) * r_out / G_KPC_KMS2_PER_MSUN)

        rows.append({
            "galaxy": gal,
            "r_out_kpc": r_out,
            "Vf_obs_kms": Vf_obs,
            "Vf_pred_kms": Vf_pred,
            "Vlum_out_kms": float(Vlum_m[i_out]),
            "DeltaX_pred_out": float(dx_pred_m[i_out]),
            "Menc_lum_proxy_Msun": Menc_lum,
            "n_points_used": int(np.count_nonzero(m)),
        })

    out = pd.DataFrame(rows).sort_values("Vf_obs_kms")
    out_path = Path(args.out_csv).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    # quick summary printed
    if len(out) >= 3:
        x = out["Vf_obs_kms"].to_numpy(float)
        y = out["Vf_pred_kms"].to_numpy(float)
        # linear fit y = a + b x
        A = np.vstack([np.ones_like(x), x]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        a0, b0 = float(coef[0]), float(coef[1])
        rmse = float(np.sqrt(np.mean((y - (a0 + b0 * x)) ** 2)))
        corr = float(np.corrcoef(x, y)[0, 1])
        print(f"Wrote: {out_path}")
        print(f"Galaxies used: {len(out)}")
        print(f"Fit: Vf_pred ≈ {a0:.3f} + {b0:.3f} * Vf_obs")
        print(f"RMSE(Vf): {rmse:.3f} km/s   corr: {corr:.3f}")
    else:
        print(f"Wrote: {out_path}")
        print(f"Galaxies used: {len(out)} (too few for summary stats)")


if __name__ == "__main__":
    main()