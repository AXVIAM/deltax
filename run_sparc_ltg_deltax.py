#!/usr/bin/env python3
"""
Run DeltaX mapping on SPARC LTG rotmod files.

Input format (per file):
# Distance = ...
# Rad   Vobs    errV    Vgas    Vdisk   Vbul    SBdisk  SBbul
# kpc   km/s    km/s    km/s    km/s    km/s    L/pc^2  L/pc^2
<data rows...>

Outputs:
- summary CSV: one row per galaxy with metrics
- per-galaxy radial CSVs: r, Vobs, Vlum, DeltaX_obs, DeltaX_pred, etc.
- (optional) ablation variants: drop M, D, I (and pairs) with same metrics

ORDER OF OPERATIONS (DO NOT REORDER):

1. Load SPARC rotation curve data (no smoothing, no interpolation)
2. Compute observed DeltaX from Vobs and Vlum
3. Construct normalized distance coordinate D_hat
4. Compute baryonic driver B(D_hat) with saturation
5. Solve DeltaX (1 + N DeltaX) = B(D_hat) analytically
6. Compare predicted vs observed DeltaX (shape and amplitude)
7. Aggregate per-galaxy and sample-level statistics
"""

from __future__ import annotations

from __future__ import annotations

import argparse
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


import numpy as np
import pandas as pd

# =============================
# Physical and Numerical Constants
# =============================


# -----------------------------
# Utilities
# -----------------------------
def describe_array(name: str, x: np.ndarray) -> Dict[str, float]:
    """Return basic stats for debugging."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {
            f"{name}_min": float("nan"),
            f"{name}_p1": float("nan"),
            f"{name}_p50": float("nan"),
            f"{name}_p99": float("nan"),
            f"{name}_max": float("nan"),
        }
    return {
        f"{name}_min": float(np.min(x)),
        f"{name}_p1": float(np.percentile(x, 1)),
        f"{name}_p50": float(np.percentile(x, 50)),
        f"{name}_p99": float(np.percentile(x, 99)),
        f"{name}_max": float(np.max(x)),
    }


 # =============================
# Physical and Numerical Constants (Defaults, Not Tuned Per Galaxy)
# =============================
G_KPC_KMS2_PER_MSUN = 4.30091e-6  # kpc (km/s)^2 / Msun

DEFAULT_A = math.pi / 5.0
DEFAULT_B = math.pi / 3.0
DEFAULT_C = math.pi / 4.0
DEFAULT_N = 7.0 * math.pi / 12.0
DEFAULT_K = math.pi


 # =============================
# Numerical Stabilization and Derivative Operators
# =============================
def clamp_min(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.maximum(x, eps)


def moving_average_3(y: np.ndarray) -> np.ndarray:
    """
    Light 3-point moving average, edge-handled by copying endpoints.
    """
    if y.size < 3:
        return y.copy()
    out = y.copy()
    out[1:-1] = (y[:-2] + y[1:-1] + y[2:]) / 3.0
    out[0] = y[0]
    out[-1] = y[-1]
    return out


def second_derivative_nonuniform(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Second derivative y''(x) on a non-uniform grid using a standard 3-point stencil.

    For interior points i:
      y''_i = 2 * [ (y_{i+1}-y_i)/h2 - (y_i-y_{i-1})/h1 ] / (h1 + h2)
    where h1 = x_i - x_{i-1}, h2 = x_{i+1} - x_i

    Endpoints are set by copying nearest interior estimate.
    """
    n = x.size
    if n < 3:
        return np.zeros_like(y)

    d2 = np.zeros_like(y, dtype=float)
    for i in range(1, n - 1):
        h1 = x[i] - x[i - 1]
        h2 = x[i + 1] - x[i]
        if h1 <= 0 or h2 <= 0:
            d2[i] = 0.0
            continue
        term1 = (y[i + 1] - y[i]) / h2
        term2 = (y[i] - y[i - 1]) / h1
        d2[i] = 2.0 * (term1 - term2) / (h1 + h2)

    d2[0] = d2[1]
    d2[-1] = d2[-2]
    return d2


 # =============================
# SPARC Data Ingestion (No Smoothing, No Interpolation)
# =============================
def parse_rotmod_file(path: Path) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Parse SPARC rotmod file into dataframe with expected columns.
    Returns (meta, df).
    """
    meta: Dict[str, float] = {}
    rows: List[List[float]] = []

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                # Example: "# Distance = 13.8 Mpc"
                m = re.search(r"Distance\s*=\s*([0-9.+-eE]+)", line)
                if m:
                    meta["distance_mpc"] = float(m.group(1))
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

    # Clean obvious issues
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["r_kpc", "Vobs", "Vdisk", "SBdisk"])
    df = df.sort_values("r_kpc").reset_index(drop=True)

    return meta, df


 # =============================
# Ablation Variants (Structural Tests, Not Model Fits)
# =============================
@dataclass
class Variant:
    name: str
    drop_M: bool = False
    drop_D: bool = False
    drop_I: bool = False


VARIANTS: List[Variant] = [
    Variant("full", False, False, False),
    Variant("drop_M", True, False, False),
    Variant("drop_D", False, True, False),
    Variant("drop_I", False, False, True),
    Variant("only_M", False, True, True),
    Variant("only_D", True, False, True),
    Variant("only_I", True, True, False),
    Variant("MD_only", False, False, True),
    Variant("MI_only", False, True, False),
    Variant("DI_only", True, False, False),
]


#
 # =============================
# Saturation / Moderation Term (Prevents Runaway Growth)
# =============================
# This term is an intrinsic part of B(D), not a post-fit correction.
def saturation_denom(x: np.ndarray, mode: str, sat_lambda: float = 1.0) -> np.ndarray:
    """Return the multiplicative denominator used to moderate the power-law term.

    All modes are >= 1.

    Modes:
      - 'log1p'   : 1 + log(1 + x)  (current default)
      - 'frac'    : 1 + x/(1 + x)   (soft saturation to 2)
      - 'tanhlog' : 1 + tanh(log(1 + x)) (soft saturation to 2)
      - 'none'    : 1

    Notes:
      - We use log1p for numerical stability.
      - We clamp x to be positive.
    """
    mode = (mode or "log1p").lower().strip()
    x = clamp_min(np.asarray(x, dtype=float))

    if mode in ("none", "no", "nol", "nologo", "off"):
        return np.ones_like(x)

    if mode in ("log1p", "log", "ln"):
        return 1.0 + sat_lambda * np.log1p(x)

    if mode in ("frac", "x_over_1px", "rational"):
        return 1.0 + (x / (1.0 + x))

    if mode in ("tanhlog", "tanh", "tanh_log"):
        return 1.0 + np.tanh(np.log1p(x))

    raise ValueError(f"Unknown saturation mode '{mode}'. Use one of: log1p, frac, tanhlog, none")

 # =============================
# Core DeltaX Equation Evaluation (Single-Galaxy, Deterministic)
# =============================
def compute_deltax(
    r: np.ndarray,
    Vobs: np.ndarray,
    Vgas: np.ndarray,
    Vdisk: np.ndarray,
    Vbul: np.ndarray,
    SBdisk: np.ndarray,
    a: float,
    b: float,
    c: float,
    N: float,
    variant: Variant,
    i_floor: float = 0.0,
    i_cap_p99: float = 0.0,
    K: float = DEFAULT_K,
    saturation: str = "log1p",
    sat_lambda: float = 1.0,
) -> Dict[str, np.ndarray]:
    """
    Compute DeltaX mapping arrays for one galaxy.
    """
    # ==================================================================
    # DELTAX CORE COMPUTATION (single galaxy, all radii)
    # This function is pure: no global state, no cross-galaxy coupling.
    # ==================================================================
    # ------------------------------------------------------------------
    # (2) OBSERVED RESPONSE (DATA ONLY)
    # Construct DeltaX_obs directly from SPARC observables.
    # No modeling assumptions enter here.
    # ------------------------------------------------------------------
    # --- (2) Observed Response: DeltaX_obs from SPARC observables ---
    Vlum = np.sqrt(clamp_min(Vgas**2 + Vdisk**2 + Vbul**2))
    DeltaX_obs = Vobs / clamp_min(Vlum)

    # ------------------------------------------------------------------
    # (3a) PHYSICAL FIELDS ON NATIVE RADIAL GRID
    # Computed per radius with no interpolation or resampling.
    # ------------------------------------------------------------------
    # --- (3a) Physical Fields on Native Radial Grid ---
    # M(r) = Vlum^2 * r / G
    M = (Vlum**2) * r / G_KPC_KMS2_PER_MSUN

    # D(r) = r
    D = r.copy()

    # ------------------------------------------------------------------
    # (3b) STRUCTURAL INVARIANT I(r)
    # Derived from disk surface brightness and velocity curvature.
    # Light local smoothing is applied ONLY for derivatives.
    # ------------------------------------------------------------------
    # --- (3b) Structural Invariant I(r) from Surface Brightness and Curvature ---
    # Structural invariant I(r):
    # p = max(SBdisk,0) / max_r SBdisk
    SBdisk_pos = np.maximum(SBdisk, 0.0)
    sb_max = float(np.max(SBdisk_pos)) if SBdisk_pos.size else 0.0
    if sb_max <= 0:
        p = np.zeros_like(SBdisk_pos)
    else:
        p = SBdisk_pos / sb_max

    # S = -p ln p, with clamp to avoid log(0)
    p_cl = clamp_min(p)
    S = -(p_cl * np.log(p_cl))

    # Smooth then second derivatives on native grid
    S_s = moving_average_3(S)
    Vlum_s = moving_average_3(Vlum)

    S_dd = second_derivative_nonuniform(r, S_s)
    Vlum_dd = second_derivative_nonuniform(r, Vlum_s)

    Ient = np.abs(S_dd)
    klum = np.abs(Vlum_dd)
    I = np.sqrt(clamp_min(Ient) * clamp_min(klum))

    # Optional stabilization for I(r) (off by default).
    # i_floor sets a hard minimum (useful if you later want a soft baseline).
    # i_cap_p99, if > 0, caps I(r) at (p99 * i_cap_p99) to suppress rare spikes.
    if i_floor > 0:
        I = np.maximum(I, float(i_floor))
    if i_cap_p99 > 0:
        cap = float(np.percentile(I[np.isfinite(I)], 99)) * float(i_cap_p99)
        if np.isfinite(cap) and cap > 0:
            I = np.minimum(I, cap)

    # ------------------------------------------------------------------
    # (3c) PER-GALAXY NORMALIZATION
    # Defines characteristic scales to form dimensionless variables.
    # No sample-level tuning or cross-galaxy coupling.
    # ------------------------------------------------------------------
    # --- Explicit normalization (per-galaxy characteristic scales) ---
    # These normalizations make the domain of the power-law explicit and controlled.
    # They correspond to implicit assumptions in the original pipeline.

    # --- (3c) Per-Galaxy Normalization (Dimensionless Variables) ---
    # Characteristic mass: median enclosed mass
    M0 = float(np.median(M[np.isfinite(M)])) if np.any(np.isfinite(M)) else 1.0

    # Characteristic radius: median radius
    D0 = float(np.median(D[np.isfinite(D)])) if np.any(np.isfinite(D)) else 1.0

    # Characteristic invariant scale: 99th percentile (robust to spikes)
    I0 = float(np.percentile(I[np.isfinite(I)], 99)) if np.any(np.isfinite(I)) else 1.0

    # Prevent division by zero
    if not np.isfinite(M0) or M0 <= 0:
        M0 = 1.0
    if not np.isfinite(D0) or D0 <= 0:
        D0 = 1.0
    if not np.isfinite(I0) or I0 <= 0:
        I0 = 1.0

    # Dimensionless normalized variables
    Mhat = M / M0
    Dhat = D / D0
    Ihat = I / I0

    # ------------------------------------------------------------------
    # (ABLATION LOGIC)
    # Setting an exponent to zero removes that variable exactly:
    #   X**0 = 1 → no scale, no shape, no influence.
    # Equation structure is preserved.
    # ------------------------------------------------------------------
    # --- Ablation Logic: Zero Exponent Removes Term Exactly ---
    # Apply ablations by setting dropped terms to 1
    M_term = np.ones_like(Mhat) if variant.drop_M else clamp_min(Mhat)
    D_term = np.ones_like(Dhat) if variant.drop_D else clamp_min(Dhat)
    I_term = np.ones_like(Ihat) if variant.drop_I else clamp_min(Ihat)

    # ------------------------------------------------------------------
    # (4) BARYONIC DRIVER B(r)
    # Power-law response with structural saturation.
    # Saturation is part of the equation, not a post-fit correction.
    # ------------------------------------------------------------------
    # --- (4) Baryonic Driver B(r) with Structural Saturation ---
    # B(r) = K * M^a D^b I^c / saturation(mdi)
    # Default saturation is 1 + log(1 + mdi), matching the previous implementation.
    mdi = clamp_min(M_term * D_term * I_term)
    denom = saturation_denom(mdi, mode=saturation, sat_lambda=sat_lambda)

    B = K * (M_term**a) * (D_term**b) * (I_term**c) / denom

    # ------------------------------------------------------------------
    # (5) ANALYTIC SOLUTION
    # Solve DeltaX (1 + N DeltaX) = B for DeltaX ≥ 0.
    # ------------------------------------------------------------------
    # --- (5) Analytic Solution of DeltaX (1 + N DeltaX) = B ---
    # Quadratic closed-form: DeltaX = (-1 + sqrt(1 + 4 N B)) / (2 N)
    # Handle N ~ 0 safely (though in your use N is not 0)
    if abs(N) < 1e-15:
        DeltaX_pred = B.copy()
    else:
        inside = 1.0 + 4.0 * N * B
        inside = clamp_min(inside)
        DeltaX_pred = (-1.0 + np.sqrt(inside)) / (2.0 * N)

    # --- (6) Return Radial Fields for Evaluation Only ---
    # Outputs are used only for comparison and aggregation.
    # No feedback into the model occurs.
    return {
        "r_kpc": r,
        "Vobs": Vobs,
        "Vlum": Vlum,
        "DeltaX_obs": DeltaX_obs,
        "DeltaX_pred": DeltaX_pred,
        "M": M,
        "D": D,
        "I": I,
        "B": B,
        "Ient": Ient,
        "klum": klum,
        "M0": M0,
        "D0": D0,
        "I0": I0,
    }
# =============================
# Evaluation Metrics (No Feedback into the Equation)
# =============================



def rmse(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    d = x - y
    return float(np.sqrt(np.mean(d * d)))

def rmse_shape(pred: np.ndarray, obs: np.ndarray) -> float:
    """
    Shape-only RMSE: allows a single optimal scalar rescaling of pred
    to best match obs, then computes RMSE.
    """
    if pred.size == 0:
        return float("nan")
    # Least-squares scalar alpha = (pred·obs)/(pred·pred)
    denom = np.dot(pred, pred)
    if denom <= 0 or not np.isfinite(denom):
        return float("nan")
    alpha = float(np.dot(pred, obs) / denom)
    res = alpha * pred - obs
    return float(np.sqrt(np.mean(res * res)))


# -----------------------------
# Curvature RMSE for grid scan universality (paper-style)
# -----------------------------
def curvature_rmse(x: np.ndarray, y_pred: np.ndarray, y_obs: np.ndarray) -> float:
    """Curvature RMSE between predicted and observed curves.

    Uses the second derivative on the native non-uniform radius grid after a light
    3-pt moving average (same smoother used elsewhere). Endpoints are handled by
    the derivative routine.

    This is intended to match the paper’s "curvature RMSE" concept.
    """
    if x.size < 3:
        return float("nan")
    yp = moving_average_3(np.asarray(y_pred, dtype=float))
    yo = moving_average_3(np.asarray(y_obs, dtype=float))
    kp = second_derivative_nonuniform(np.asarray(x, dtype=float), yp)
    ko = second_derivative_nonuniform(np.asarray(x, dtype=float), yo)
    m = np.isfinite(kp) & np.isfinite(ko)
    if np.count_nonzero(m) < 3:
        return float("nan")
    return rmse(kp[m], ko[m])


# -----------------------------
# Velocity-space curvature RMSE
# -----------------------------
def velocity_curvature_rmse(r: np.ndarray, V_pred: np.ndarray, V_obs: np.ndarray) -> float:
    """
    Velocity-space curvature RMSE.

    Computes RMSE between second derivatives of predicted and observed
    rotation curves V(r), after light 3-point smoothing, on the native
    non-uniform radius grid.
    """
    if r.size < 3:
        return float("nan")

    Vp = moving_average_3(np.asarray(V_pred, dtype=float))
    Vo = moving_average_3(np.asarray(V_obs, dtype=float))

    kp = second_derivative_nonuniform(np.asarray(r, dtype=float), Vp)
    ko = second_derivative_nonuniform(np.asarray(r, dtype=float), Vo)

    m = np.isfinite(kp) & np.isfinite(ko)
    if np.count_nonzero(m) < 3:
        return float("nan")

    return rmse(kp[m], ko[m])


# =============================
# Robustness Tests (Exponent Sensitivity, Not Model Training)
# =============================
def _objective_over_files(
    files: List[Path],
    a: float,
    b: float,
    c: float,
    N: float,
    K: float,
    metric: str = "shape",
    sample_limit: int = 0,
    seed: int = 0,
    saturation: str = "log1p",
) -> float:
    """Compute mean RMSE over the sample for the full variant.

    metric:
      - "shape": mean rmse_DeltaX_shape
      - "absolute": mean rmse_DeltaX_absolute
      - "combined": mean of (shape + absolute)
    """
    if sample_limit and sample_limit > 0:
        rnd = random.Random(seed)
        files = files.copy()
        rnd.shuffle(files)
        files = files[: int(sample_limit)]

    # Enforce D-only model: M and I removed
    a = 0.0
    c = 0.0
    vals: List[float] = []
    for fp in files:
        _, df = parse_rotmod_file(fp)
        if df.empty:
            continue
        r = df["r_kpc"].to_numpy(dtype=float)
        Vobs = df["Vobs"].to_numpy(dtype=float)
        Vgas = df["Vgas"].to_numpy(dtype=float)
        Vdisk = df["Vdisk"].to_numpy(dtype=float)
        Vbul = df["Vbul"].to_numpy(dtype=float)
        SBdisk = df["SBdisk"].to_numpy(dtype=float)

        res = compute_deltax(
            r=r,
            Vobs=Vobs,
            Vgas=Vgas,
            Vdisk=Vdisk,
            Vbul=Vbul,
            SBdisk=SBdisk,
            a=a,
            b=b,
            c=c,
            N=N,
            variant=next(v for v in VARIANTS if v.name == "only_D"),
            K=K,
            saturation=saturation,
        )

        mask = np.isfinite(res["DeltaX_obs"]) & np.isfinite(res["DeltaX_pred"]) & (res["Vlum"] > 1e-6)
        if np.count_nonzero(mask) < 3:
            continue
        pred = res["DeltaX_pred"][mask]
        obs = res["DeltaX_obs"][mask]
        rm_abs = rmse(pred, obs)
        rm_shp = rmse_shape(pred, obs)

        if metric == "absolute":
            val = rm_abs
        elif metric == "combined":
            val = rm_abs + rm_shp
        else:
            val = rm_shp

        if np.isfinite(val):
            vals.append(float(val))

    if not vals:
        return float("nan")
    return float(np.mean(vals))


def fit_exponents_random_search(
    in_dir: Path,
    iters: int = 200,
    metric: str = "shape",
    sample_limit: int = 0,
    seed: int = 0,
    min_a: float = 0.0,
    min_b: float = 0.0,
    min_c: float = 0.0,
    min_N: float = 0.0,
    a0: float = DEFAULT_A,
    b0: float = DEFAULT_B,
    c0: float = DEFAULT_C,
    N0: float = DEFAULT_N,
    K: float = DEFAULT_K,
    saturation: str = "log1p",
) -> Tuple[float, float, float, float, float]:
    """Simple, reproducible global search around the current parameters.

    Returns (best_score, a, b, c, N).

    This is intentionally dependency-free (no scipy) and meant as a robustness test.
    """
    files = sorted(in_dir.glob("*_rotmod.dat"))
    if not files:
        raise FileNotFoundError(f"No *_rotmod.dat files found in {in_dir}")

    rnd = random.Random(seed)

    # Start at current parameters
    best_a, best_b, best_c, best_N = float(a0), float(b0), float(c0), float(N0)
    best_score = _objective_over_files(files, best_a, best_b, best_c, best_N, K=K, metric=metric, sample_limit=sample_limit, seed=seed, saturation=saturation)

    # Multi-scale random perturbations
    # We perturb additively with decaying step sizes.
    step_scales = [0.50, 0.25, 0.10, 0.05, 0.02]

    for scale in step_scales:
        for _ in range(max(1, iters // len(step_scales))):
            # Propose parameters near current best.
            # We keep N positive to avoid sign/pathology issues.
            a = best_a + rnd.uniform(-scale, scale)
            b = best_b + rnd.uniform(-scale, scale)
            c = best_c + rnd.uniform(-scale, scale)
            N = best_N + rnd.uniform(-scale, scale)

            # Hard constraints to prevent degeneracy / collapse
            if N <= max(1e-6, float(min_N)):
                continue
            if not (max(0.0, float(min_a)) < a < 2.0 and max(0.0, float(min_b)) < b < 2.0 and max(0.0, float(min_c)) < c < 2.0):
                continue

            score = _objective_over_files(files, a, b, c, N, K=K, metric=metric, sample_limit=sample_limit, seed=seed, saturation=saturation)
            if np.isfinite(score) and (not np.isfinite(best_score) or score < best_score):
                best_score = float(score)
                best_a, best_b, best_c, best_N = float(a), float(b), float(c), float(N)

    return best_score, best_a, best_b, best_c, best_N


# =============================
# Universality Grid Scan (Population-Level Analysis)
# =============================
def parse_grid(spec: str) -> np.ndarray:
    """Parse grid spec min:max:n into linspace."""
    try:
        lo, hi, n = spec.split(":")
        lo = float(lo); hi = float(hi); n = int(n)
        if n < 2:
            return np.array([lo], dtype=float)
        return np.linspace(lo, hi, n)
    except Exception:
        raise ValueError(f"Invalid grid spec '{spec}', expected min:max:n")


def grid_scan_universality(
    in_dir: Path,
    grid_a: np.ndarray,
    grid_b: np.ndarray,
    grid_c: np.ndarray,
    grid_N: np.ndarray,
    rmse_thresh: float = 1.5,
    metric: str = "shape",
    agg: str = "median",
    K: float = DEFAULT_K,
    saturation: str = "log1p",
) -> pd.DataFrame:
    """
    Scan exponent grid and compute population-level statistics:
      - mean and median of the requested metric
      - std, fraction below threshold, worst-case
      - aggregation across galaxies can be median or mean (paper uses median)
    """
    files = sorted(in_dir.glob("*_rotmod.dat"))
    if not files:
        raise FileNotFoundError(f"No *_rotmod.dat files found in {in_dir}")

    rows = []
    for a in grid_a:
        for b in grid_b:
            for c in grid_c:
                for N in grid_N:
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
                            r=r, Vobs=Vobs, Vgas=Vgas, Vdisk=Vdisk, Vbul=Vbul, SBdisk=SBdisk,
                            a=float(a), b=float(b), c=float(c), N=float(N), variant=VARIANTS[0], K=K,
                            saturation=saturation,
                        )
                        mask = np.isfinite(res["DeltaX_obs"]) & np.isfinite(res["DeltaX_pred"]) & (res["Vlum"] > 1e-6)
                        if np.count_nonzero(mask) < 3:
                            continue
                        pred = res["DeltaX_pred"][mask]
                        obs = res["DeltaX_obs"][mask]

                        if metric == "absolute":
                            rm = rmse(pred, obs)
                        elif metric == "combined":
                            rm = rmse_shape(pred, obs) + rmse(pred, obs)
                        elif metric == "curvature":
                            rm = curvature_rmse(res["r_kpc"][mask], pred, obs)
                        elif metric == "velocity_curvature":
                            Vp = pred * res["Vlum"][mask]
                            Vo = res["Vobs"][mask]
                            rm = velocity_curvature_rmse(res["r_kpc"][mask], Vp, Vo)
                        else:
                            rm = rmse_shape(pred, obs)

                        if np.isfinite(rm):
                            rmses.append(float(rm))

                    if not rmses:
                        continue
                    rmses = np.asarray(rmses)
                    mean_v = float(np.mean(rmses))
                    median_v = float(np.median(rmses))
                    std_v = float(np.std(rmses))
                    worst_v = float(np.max(rmses))
                    frac_v = float(np.mean(rmses <= rmse_thresh))

                    score_agg = median_v if agg == "median" else mean_v

                    rows.append({
                        "a": float(a),
                        "b": float(b),
                        "c": float(c),
                        "N": float(N),
                        "grid_metric": str(metric),
                        "grid_agg": str(agg),
                        "score_agg": float(score_agg),
                        "mean_metric": float(mean_v),
                        "median_metric": float(median_v),
                        "std_metric": float(std_v),
                        "frac_below_thresh": float(frac_v),
                        "worst_metric": float(worst_v),
                        "n_galaxies": int(rmses.size),
                    })

    return pd.DataFrame(rows)


 # =============================
# End-to-End Pipeline Execution (Per-Galaxy, No Coupling)
# =============================
def run_directory(
    in_dir: Path,
    out_dir: Path,
    a: float,
    b: float,
    c: float,
    N: float,
    do_variants: bool,
    debug_galaxy: str = "",
    i_floor: float = 0.0,
    i_cap_p99: float = 0.0,
    K: float = DEFAULT_K,
    saturation: str = "log1p",
    sat_lambda: float = 1.0,
) -> None:
    # ==================================================================
    # SAMPLE-LEVEL EVALUATION (SPARC LTGs)
    # Applies the DeltaX equation independently to each galaxy.
    # ==================================================================
    out_dir.mkdir(parents=True, exist_ok=True)
    radial_dir = out_dir / "radial"
    radial_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*_rotmod.dat"))
    if not files:
        raise FileNotFoundError(f"No *_rotmod.dat files found in {in_dir}")

    summary_rows: List[Dict[str, object]] = []

    variants = VARIANTS if do_variants else [VARIANTS[0]]

    # NOTE: Each galaxy is processed independently.
    # No parameters are fit or adjusted per galaxy.
    for fp in files:
        galaxy = fp.name.replace("_rotmod.dat", "")
        meta, df = parse_rotmod_file(fp)

        # Arrays
        r = df["r_kpc"].to_numpy(dtype=float)
        Vobs = df["Vobs"].to_numpy(dtype=float)
        Vgas = df["Vgas"].to_numpy(dtype=float)
        Vdisk = df["Vdisk"].to_numpy(dtype=float)
        Vbul = df["Vbul"].to_numpy(dtype=float)
        SBdisk = df["SBdisk"].to_numpy(dtype=float)

        # Safety: drop points where Vlum would be ~0
        # (rare in practice; but avoids blowups in DeltaX_obs)
        # We'll compute Vlum in compute_deltax and then mask if needed.

        for var in variants:
            res = compute_deltax(
                r=r,
                Vobs=Vobs,
                Vgas=Vgas,
                Vdisk=Vdisk,
                Vbul=Vbul,
                SBdisk=SBdisk,
                a=a,
                b=b,
                c=c,
                N=N,
                variant=var,
                i_floor=i_floor,
                i_cap_p99=i_cap_p99,
                K=K,
                saturation=saturation,
                sat_lambda=float(sat_lambda),
            )

            if debug_galaxy and galaxy.lower() == debug_galaxy.lower() and var.name == "full":
                stats: Dict[str, float] = {}
                stats.update(describe_array("M", res["M"]))
                stats.update(describe_array("D", res["D"]))
                stats.update(describe_array("I", res["I"]))
                stats.update(describe_array("B", res["B"]))
                stats.update(describe_array("DeltaX_obs", res["DeltaX_obs"]))
                stats.update(describe_array("DeltaX_pred", res["DeltaX_pred"]))
                stats["M0"] = res["M0"]
                stats["D0"] = res["D0"]
                stats["I0"] = res["I0"]
                print(f"\n[DEBUG] {galaxy} (variant=full) stats:")
                for k in sorted(stats.keys()):
                    print(f"  {k}: {stats[k]:.6g}")
                print("")

            # Mask invalid points (e.g. tiny Vlum causing huge ratios)
            mask = np.isfinite(res["DeltaX_obs"]) & np.isfinite(res["DeltaX_pred"]) & (res["Vlum"] > 1e-6)
            if np.count_nonzero(mask) < 3:
                rmse_abs = float("nan")
                rmse_shp = float("nan")
                rmse_shp_inner = float("nan")
                rmse_shp_outer = float("nan")
            else:
                r_m = res["r_kpc"][mask]
                pred = res["DeltaX_pred"][mask]
                obs = res["DeltaX_obs"][mask]

                rmse_abs = rmse(pred, obs)
                rmse_shp = rmse_shape(pred, obs)

                # --- Inner vs outer radial split (median radius) ---
                r_mid = float(np.median(r_m))
                inner = r_m <= r_mid
                outer = r_m > r_mid

                rmse_shp_inner = rmse_shape(pred[inner], obs[inner]) if np.count_nonzero(inner) >= 3 else float("nan")
                rmse_shp_outer = rmse_shape(pred[outer], obs[outer]) if np.count_nonzero(outer) >= 3 else float("nan")

            # --- Velocity-space RMSE metrics ---
            if np.count_nonzero(mask) < 3:
                rmse_V_abs = float("nan")
                rmse_V_scaled = float("nan")
                V_scale_alpha = float("nan")
            else:
                Vpred = res["DeltaX_pred"][mask] * res["Vlum"][mask]
                Vobs_m = res["Vobs"][mask]

                rmse_V_abs = rmse(Vpred, Vobs_m)

                denom = float(np.dot(Vpred, Vpred))
                if denom > 0 and np.isfinite(denom):
                    V_scale_alpha = float(np.dot(Vobs_m, Vpred) / denom)
                else:
                    V_scale_alpha = 1.0

                Vpred_scaled = V_scale_alpha * Vpred
                rmse_V_scaled = rmse(Vpred_scaled, Vobs_m)

            # Per-galaxy summary row
            summary_rows.append({
                "galaxy": galaxy,
                "variant": var.name,
                "n_points": int(np.count_nonzero(mask)),
                "r_max_kpc": float(np.max(res["r_kpc"][mask])) if np.count_nonzero(mask) else float("nan"),
                "distance_mpc": meta.get("distance_mpc", float("nan")),
                "rmse_DeltaX_absolute": rmse_abs,
                "rmse_DeltaX_shape": rmse_shp,
                "rmse_DeltaX_shape_inner": rmse_shp_inner,
                "rmse_DeltaX_shape_outer": rmse_shp_outer,
                "rmse_V_absolute": rmse_V_abs,
                "rmse_V_scaled": rmse_V_scaled,
                "V_scale_alpha": V_scale_alpha,
                "median_DeltaX_obs": float(np.median(res["DeltaX_obs"][mask])) if np.count_nonzero(mask) else float("nan"),
                "median_DeltaX_pred": float(np.median(res["DeltaX_pred"][mask])) if np.count_nonzero(mask) else float("nan"),
            })

            # Write per-galaxy radial outputs for full model only (or for each variant if you want)
            # By default: write for each variant to support plotting comparisons easily.
            radial_out = radial_dir / f"{galaxy}__{var.name}.csv"
            out_df = pd.DataFrame({
                "r_kpc": res["r_kpc"],
                "Vobs": res["Vobs"],
                "Vlum": res["Vlum"],
                "DeltaX_obs": res["DeltaX_obs"],
                "DeltaX_pred": res["DeltaX_pred"],
                "M_Msun": res["M"],
                "D_kpc": res["D"],
                "I": res["I"],
                "B": res["B"],
                "Ient": res["Ient"],
                "klum": res["klum"],
                "Mhat": res["M"] / res["M0"],
                "Dhat": res["D"] / res["D0"],
                "Ihat": res["I"] / res["I0"],
            })
            out_df.to_csv(radial_out, index=False)

    # Save summary with parameter-tagged filenames
    sat_tag = str(saturation)
    lam_tag = f"lam{sat_lambda:.3f}".replace(".", "p")
    param_tag = f"{sat_tag}_{lam_tag}_a{a:.3f}_b{b:.3f}_c{c:.3f}_N{N:.3f}".replace(".", "p")

    summary = pd.DataFrame(summary_rows)
    summary_path = out_dir / f"summary_per_galaxy_{param_tag}.csv"
    summary.to_csv(summary_path, index=False)

    # Convenience: also save "full" subset ranked by shape-only rmse
    full = summary[summary["variant"] == "full"].copy()
    full = full.sort_values("rmse_DeltaX_shape", ascending=True)
    full_sorted_path = out_dir / f"summary_full_sorted_{param_tag}.csv"
    full.to_csv(full_sorted_path, index=False)

    print(f"Wrote: {summary_path}")
    print(f"Wrote: {full_sorted_path}")
    print(f"Wrote radial CSVs to: {radial_dir}")
    print(f"Galaxies processed: {len(files)}")
    print(f"Variants per galaxy: {len(variants)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=float, default=DEFAULT_K, help="Global amplitude constant K (default π)")
    ap.add_argument(
        "--saturation",
        type=str,
        default="log1p",
        choices=["log1p", "frac", "tanhlog", "none"],
        help=(
            "Saturation / moderation applied to the MDI product in B(r). "
            "log1p uses 1+log(1+x) (default). "
            "frac uses 1 + x/(1+x). "
            "tanhlog uses 1 + tanh(log(1+x)). "
            "none disables moderation (pure power-law)."
        ),
    )
    ap.add_argument(
        "--sat-lambda",
        type=float,
        default=1.0,
        help="Strength of saturation term (lambda). Default 1.0 reproduces current results."
    )
    ap.add_argument("--scan-K", type=str, default="", help="Scan K on a log-spaced grid specified as min:max:n (e.g. 0.2:10:80)")
    ap.add_argument("--in-dir", required=True, help="Directory containing *_rotmod.dat files")
    ap.add_argument("--out-dir", required=True, help="Output directory for CSVs")
    ap.add_argument("--a", type=float, default=DEFAULT_A)
    ap.add_argument("--b", type=float, default=DEFAULT_B)
    ap.add_argument("--c", type=float, default=DEFAULT_C)
    ap.add_argument("--N", type=float, default=DEFAULT_N)
    ap.add_argument("--variants", action="store_true", help="Run ablation variants in addition to full model")
    ap.add_argument("--debug-galaxy", default="", help="If set, print detailed stats for this galaxy (case-insensitive)")
    ap.add_argument("--i-floor", type=float, default=0.0, help="Optional minimum floor for I(r); default 0 disables")
    ap.add_argument("--i-cap-p99", type=float, default=0.0, help="Optional cap factor applied to p99(I); default 0 disables")
    ap.add_argument("--fit-exponents", action="store_true", help="Run a global random-search fit for (a,b,c,N) and print best decimal values")
    ap.add_argument("--fit-iters", type=int, default=200, help="Iterations for exponent fit (random search)")
    ap.add_argument("--fit-metric", choices=["shape", "absolute", "combined"], default="shape", help="Objective metric for exponent fit")
    ap.add_argument("--fit-sample", type=int, default=0, help="If >0, fit on a random subset of this many galaxies (speeds up)")
    ap.add_argument("--fit-seed", type=int, default=0, help="Seed for exponent fit reproducibility")
    ap.add_argument("--min-a", type=float, default=0.0, help="Minimum allowed a during exponent fit (prevents collapse); default 0")
    ap.add_argument("--min-b", type=float, default=0.0, help="Minimum allowed b during exponent fit; default 0")
    ap.add_argument("--min-c", type=float, default=0.0, help="Minimum allowed c during exponent fit (prevents collapse); default 0")
    ap.add_argument("--min-N", type=float, default=0.0, help="Minimum allowed N during exponent fit; default 0")
    # --- Grid scan / universality mode ---
    ap.add_argument("--grid-scan", action="store_true", help="Run grid scan over (a,b,c,N) and compute population universality statistics")
    ap.add_argument("--grid-a", type=str, default="0.2:1.2:6", help="Grid for a as min:max:nsteps (default 0.2:1.2:6)")
    ap.add_argument("--grid-b", type=str, default="0.4:1.4:6", help="Grid for b as min:max:nsteps (default 0.4:1.4:6)")
    ap.add_argument("--grid-c", type=str, default="0.2:1.2:6", help="Grid for c as min:max:nsteps (default 0.2:1.2:6)")
    ap.add_argument("--grid-N", type=str, default="0.5:2.5:6", help="Grid for N as min:max:nsteps (default 0.5:2.5:6)")
    ap.add_argument("--grid-thresh", type=float, default=1.5, help="RMSE_shape threshold for universality fraction (default 1.5)")
    ap.add_argument(
        "--grid-metric",
        choices=["shape", "absolute", "combined", "curvature", "velocity_curvature"],
        default="shape",
        help=(
            "Metric used per-galaxy in grid scan (default shape). "
            "shape=RMSE with scale freedom, "
            "absolute=raw RMSE, "
            "combined=shape+absolute, "
            "curvature=RMSE of 2nd-derivative between DeltaX_pred and DeltaX_obs, "
            "velocity_curvature=RMSE of 2nd-derivative between V_pred and V_obs"
        ),
    )
    ap.add_argument("--grid-agg", choices=["median", "mean"], default="median",
                    help="Aggregation across galaxies for grid score columns (default median, matching paper).")
    args = ap.parse_args()

    if bool(args.fit_exponents):
        score, a_best, b_best, c_best, N_best = fit_exponents_random_search(
            in_dir=Path(args.in_dir).expanduser(),
            iters=int(args.fit_iters),
            metric=str(args.fit_metric),
            sample_limit=int(args.fit_sample),
            seed=int(args.fit_seed),
            min_a=float(args.min_a),
            min_b=float(args.min_b),
            min_c=float(args.min_c),
            min_N=float(args.min_N),
            a0=float(args.a),
            b0=float(args.b),
            c0=float(args.c),
            N0=float(args.N),
            K=float(args.K),
            saturation=str(args.saturation),
        )
        print("\n[FIT MODE] D-only (M and I removed; fitting b and N only)")
        print("\n[FIT] Constraints:")
        print(f"  min_a={float(args.min_a):.6g}  min_b={float(args.min_b):.6g}  min_c={float(args.min_c):.6g}  min_N={float(args.min_N):.6g}")
        print("\n[FIT] Best parameters (decimal):")
        print(f"  objective ({args.fit_metric}) mean RMSE: {score:.6g}")
        print(f"  a = {a_best:.10f}")
        print(f"  b = {b_best:.10f}")
        print(f"  c = {c_best:.10f}")
        print(f"  N = {N_best:.10f}")
        print("\n[FIT] Compare to pi-based defaults:")
        print(f"  a0 = {float(args.a):.10f}")
        print(f"  b0 = {float(args.b):.10f}")
        print(f"  c0 = {float(args.c):.10f}")
        print(f"  N0 = {float(args.N):.10f}")
        return

    if bool(args.grid_scan):
        grid_a = parse_grid(args.grid_a)
        grid_b = parse_grid(args.grid_b)
        grid_c = parse_grid(args.grid_c)
        grid_N = parse_grid(args.grid_N)

        df = grid_scan_universality(
            in_dir=Path(args.in_dir).expanduser(),
            grid_a=grid_a,
            grid_b=grid_b,
            grid_c=grid_c,
            grid_N=grid_N,
            rmse_thresh=float(args.grid_thresh),
            metric=str(args.grid_metric),
            agg=str(args.grid_agg),
            K=float(args.K),
            saturation=str(args.saturation),
        )
        out = Path(args.out_dir).expanduser()
        out.mkdir(parents=True, exist_ok=True)
        out_csv = out / "grid_scan_universality.csv"
        df.to_csv(out_csv, index=False)
        print(f"Wrote grid scan results to: {out_csv}")
        return

    if args.scan_K:
        lo, hi, n = args.scan_K.split(":")
        Ks = np.logspace(np.log10(float(lo)), np.log10(float(hi)), int(n))
        files = sorted(Path(args.in_dir).expanduser().glob("*_rotmod.dat"))

        rows = []
        for Kval in Ks:
            score = _objective_over_files(
                files=files,
                a=float(args.a),
                b=float(args.b),
                c=float(args.c),
                N=float(args.N),
                K=float(Kval),
                metric=str(args.fit_metric),
                sample_limit=int(args.fit_sample),
                seed=int(args.fit_seed),
                saturation=str(args.saturation),
            )
            rows.append({"K": float(Kval), "score": float(score)})

        out = Path(args.out_dir).expanduser()
        out.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        df.to_csv(out / "scan_K_results.csv", index=False)
        print("Wrote scan_K_results.csv")
        return

    run_directory(
        in_dir=Path(args.in_dir).expanduser(),
        out_dir=Path(args.out_dir).expanduser(),
        a=float(args.a),
        b=float(args.b),
        c=float(args.c),
        N=float(args.N),
        do_variants=bool(args.variants),
        debug_galaxy=str(args.debug_galaxy),
        i_floor=float(args.i_floor),
        i_cap_p99=float(args.i_cap_p99),
        K=float(args.K),
        saturation=str(args.saturation),
        sat_lambda=float(args.sat_lambda),
    )


if __name__ == "__main__":
    main()
# =============================
# Command-Line Interface and Execution Modes
# =============================