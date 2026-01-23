# DeltaX: A Fixed-Parameter Mapping for Galaxy Rotation Curves

This repository contains the analysis code and derived data products used in the paper:

**Smolen, C. P. B.**, *DeltaX: A Fixed-Parameter Mapping for Galaxy Rotation Curves*  
(MNRAS, submitted)

The analysis applies a fixed-parameter empirical mapping to galaxy rotation curves from the SPARC database and evaluates its performance across the full sample. All parameters are fixed globally; no galaxy-by-galaxy fitting, smoothing, interpolation, or optimization is performed.

---

## Repository Structure

```
scripts/
  run_sparc_ltg_deltax.py        Main analysis script used in the paper
  figures/                      Scripts used to generate paper figures

outputs/
  ltg_onlyD_log1p/              Primary analysis outputs
    summary_per_galaxy_*.csv    Per-galaxy performance metrics
    summary_sorted_*.csv        Sample-level summaries
    radial/                     Radial profiles per galaxy

figures/
  fig1_deltax_examples.png
  fig2_velocity_examples.png
  fig3_rmse_cdf.png
  fig4_inner_outer.png

data/
  sparc/                        Input SPARC rotation-curve files (not included)
```

---

## Requirements

The analysis was run using Python 3.10+ with standard scientific packages, including:

- numpy
- pandas
- matplotlib
- scipy

Exact package versions are not critical; the results are stable under reasonable
environment differences.

---

## Data Sources

This work uses publicly available rotation-curve and baryonic mass model data from the SPARC database:

Lelli, F., McGaugh, S. S., Schombert, J. M. (2016), *AJ*, 152, 157

SPARC data are **not redistributed** in this repository. Users should obtain the data directly from the SPARC website and place the rotation-curve files in the expected directory structure under:

```
data/sparc/
```

---

## Running the Analysis

The primary analysis presented in the paper can be reproduced by running the following command from the repository root:

```
python scripts/run_sparc_ltg_deltax.py \
  --in-dir data/sparc \
  --out-dir outputs/ltg_onlyD_log1p \
  --a 0.0 \
  --b 0.5 \
  --c 0.0 \
  --N 1.3
```

This command applies the fixed-parameter $\Delta X$ mapping used in the paper (distance-only variant with logarithmic saturation) to all SPARC rotation-curve files found in `data/sparc/`.

The script produces the following outputs:

- **Per-galaxy summary file**

  ```
  outputs/ltg_onlyD_log1p/summary_per_galaxy_log1p_lam1p000_a0p000_b0p500_c0p000_N1p300.csv
  ```

  This file contains one row per galaxy, including:
  - number of usable radial points
  - shape-based and absolute RMSE metrics
  - inner and outer radial RMSE splits
  - velocity-space residual metrics
  - basic descriptive quantities used in the paper tables

- **Sorted per-galaxy summary**

  ```
  outputs/ltg_onlyD_log1p/summary_full_sorted_log1p_lam1p000_a0p000_b0p500_c0p000_N1p300.csv
  ```

  This file contains the same information as the per-galaxy summary, sorted by increasing shape-based RMSE. It is used to identify best- and worst-performing systems.

- **Per-galaxy radial profiles**

  ```
  outputs/ltg_onlyD_log1p/radial/<GALAXY>__full.csv
  ```

  These files contain radius-by-radius quantities for each galaxy, including:
  - $r$ (kpc)
  - $V_{\mathrm{obs}}$, $V_{\mathrm{lum}}$
  - $\Delta X_{\mathrm{obs}}$, $\Delta X_{\mathrm{pred}}$
  - intermediate fields used in the mapping

All quantities are evaluated independently at each radius. No information is shared between radial points prior to metric evaluation.

---

## Advanced Usage: Variants, Ablation Tests, and Robustness Checks

The main analysis script (`run_sparc_ltg_deltax.py`) supports a number of optional modes used in the paper to test robustness and perform ablation studies. All of these modes operate on the same SPARC input data and use the same core order of operations described in the paper.

### Running Term Ablation Tests

To reproduce the ablation tests discussed in Section~5.1 of the paper, run the script with the `--variants` flag. This instructs the script to evaluate multiple candidate dependency combinations (distance-only, mass-only, intensity-only, and combinations thereof) using fixed global exponents.

Example command:

```bash
python scripts/run_sparc_ltg_deltax.py \
  --in-dir data/sparc \
  --out-dir outputs/ablation_fixed_exponents \
  --a 0.5 \
  --b 0.5 \
  --c 0.5 \
  --N 1.3 \
  --variants
```

**What this does:**
- Evaluates multiple candidate variants (e.g., `only_D`, `only_M`, `MD_only`, `full`, etc.)
- Uses fixed exponents for all variants to isolate the contribution of each dependency
- Applies each variant uniformly across the full SPARC sample

**Outputs:**
- Per-galaxy metrics for each variant:
  ```
  outputs/ablation_fixed_exponents/summary_per_galaxy_*.csv
  ```
- Aggregated sample-level summaries by variant:
  ```
  outputs/ablation_fixed_exponents/summary_sorted_*.csv
  ```

These files are used to determine which dependencies are empirically necessary and which do not improve performance.

---

### Running Distance-Only (D-only) Analysis

The primary results presented in the paper use the distance-only form of the mapping. This corresponds to setting the mass and intensity exponents to zero.

Example command (used for main paper results):

```bash
python scripts/run_sparc_ltg_deltax.py \
  --in-dir data/sparc \
  --out-dir outputs/ltg_onlyD_log1p \
  --a 0.0 \
  --b 0.5 \
  --c 0.0 \
  --N 1.3
```

This produces:
- Per-galaxy summary metrics
- Sorted summary tables
- Per-galaxy radial profile CSV files

These outputs are used directly for Figures~1–4 and the appendix table in the paper.

---

### Notes on Exponents and Parameters

- The exponents `a`, `b`, and `c` control the relative weighting of candidate dependencies.
- Setting an exponent to zero removes that dependency from the mapping without altering the rest of the analysis.
- All parameters are applied globally and are never optimized on a per-galaxy basis.
- Parameter values used in the paper are motivated by robustness scans rather than fine tuning.

---

## Generating Figures

Each figure in the paper is generated by a corresponding script in:

```
scripts/figures/
```

Examples:

- **Figure 1 (ΔX profile examples)**  
  `fig1_deltax_examples.py`

- **Figure 3 (RMSE distribution)**  
  `fig3_rmse_distribution.py`


The exact image files used in the paper are provided in the `figures/` directory.

### Figure 1: $\Delta X$ profile examples (2×2 panel)

This figure plots $\Delta X_{\mathrm{obs}}(r)$ and $\Delta X_{\mathrm{pred}}(r)$ for four representative galaxies using the per-galaxy radial CSV outputs.

Run from the repository root:

```bash
python scripts/figures/fig1_deltax_examples.py \
  --radial-dir outputs/ltg_onlyD_log1p/radial \
  --galaxies CamB DDO154 NGC2403 NGC3198 \
  --out figures/fig1_deltax_examples.png
```

**Inputs:**
- Requires the per-galaxy radial files produced by the main analysis:
  `outputs/ltg_onlyD_log1p/radial/<GALAXY>__full.csv`

**Output:**
- Writes the PNG figure:
  `figures/fig1_deltax_examples.png`

---

### Figure 2: Velocity-space rotation curve examples

This figure shows observed and predicted rotation curves in velocity space, corresponding to the same representative galaxies shown in Figure~1. While Figure~1 compares the mapping directly in $\Delta X$ space, this figure illustrates how the fixed-parameter mapping translates back into physical rotation velocities.

Run from the repository root:

```bash
python scripts/figures/fig2_velocity_examples.py \
  --radial-dir outputs/ltg_onlyD_log1p/radial \
  --galaxies CamB DDO154 NGC2403 NGC3198 \
  --out figures/fig2_velocity_examples.png
```

**Inputs:**
- Requires the same per-galaxy radial CSV files produced by the main analysis:
  `outputs/ltg_onlyD_log1p/radial/<GALAXY>__full.csv`

**Output:**
- Writes the PNG figure:
  `figures/fig2_velocity_examples.png`

This figure provides an intuitive velocity-space complement to the $\Delta X$ profiles, demonstrating how agreement in $\Delta X(r)$ corresponds to agreement in the observed rotation curves.

---

### Figure 3: Distribution of shape-based RMSE across the SPARC sample

This figure summarizes the distribution of the shape-based $\mathrm{RMSE}_{\mathrm{shape}}$ metric across all galaxies in the sample. It provides a sample-level view of the mapping’s performance, highlighting the median behavior and the extent of the high-error tail.

Run from the repository root:

```bash
python scripts/figures/fig3_rmse_distribution.py \
  --summary-csv outputs/ltg_onlyD_log1p/summary_per_galaxy_log1p_lam1p000_a0p000_b0p500_c0p000_N1p300.csv \
  --out figures/fig3_rmse_cdf.png
```

**Inputs:**
- Requires the per-galaxy summary file produced by the main analysis:
  `outputs/ltg_onlyD_log1p/summary_per_galaxy_log1p_lam1p000_a0p000_b0p500_c0p000_N1p300.csv`

**Output:**
- Writes the PNG figure:
  `figures/fig3_rmse_cdf.png`

The resulting plot shows the cumulative distribution of $\mathrm{RMSE}_{\mathrm{shape}}$, with annotations indicating the median and upper quantiles used in the paper to characterize sample-wide performance.

---

### Figure 4: Inner vs. outer radial performance comparison

This figure compares the shape-based $\mathrm{RMSE}_{\mathrm{shape}}$ metric evaluated separately in the inner and outer radial regions of galaxies. For each galaxy, radial points are split at the median normalized radius, and performance is computed independently for the two regions. The figure demonstrates that neither inner nor outer radii dominate the overall performance of the mapping.

Run from the repository root:

```bash
python scripts/figures/fig4_inner_outer.py \
  --summary-csv outputs/ltg_onlyD_log1p/summary_per_galaxy_log1p_lam1p000_a0p000_b0p500_c0p000_N1p300.csv \
  --out figures/fig4_inner_outer.png
```

**Inputs:**
- Requires the per-galaxy summary CSV produced by the main analysis:
  `outputs/ltg_onlyD_log1p/summary_per_galaxy_log1p_lam1p000_a0p000_b0p500_c0p000_N1p300.csv`

**Output:**
- Writes the PNG figure:
  `figures/fig4_inner_outer.png`

The figure is rendered in grayscale to ensure clarity in black-and-white print formats.

---

### BTFR Outer-Radius Test

This script tests how the $\Delta X$ mapping relates to the Baryonic Tully--Fisher Relation (BTFR) when evaluated at a single, suitably large radius for each galaxy. Rather than fitting a global scaling relation directly, the script extracts outer-radius quantities from the rotation curves and examines whether BTFR-like behavior emerges as a projection of the fixed-parameter mapping.

The test is used to support the discussion in Section~6.2 of the paper, where the BTFR is interpreted as a limiting or single-radius projection of the more general radial behavior encoded by $\Delta X$.

Run from the repository root:

```bash
python scripts/btfr_outer_radius_test.py \
  --summary-csv outputs/ltg_onlyD_log1p/summary_per_galaxy_log1p_lam1p000_a0p000_b0p500_c0p000_N1p300.csv \
  --radial-dir outputs/ltg_onlyD_log1p/radial \
  --out outputs/btfr_outer_radius_test.csv
```

**Inputs:**
- Per-galaxy summary file produced by the main analysis:
  `outputs/ltg_onlyD_log1p/summary_per_galaxy_log1p_lam1p000_a0p000_b0p500_c0p000_N1p300.csv`
- Per-galaxy radial profile files:
  `outputs/ltg_onlyD_log1p/radial/<GALAXY>__full.csv`

**Output:**
- Writes a CSV file:
  ```
  outputs/btfr_outer_radius_test.csv
  ```

This file contains one row per galaxy with outer-radius quantities used to assess BTFR-like scaling behavior. It is intended for diagnostic and interpretive use rather than as a primary figure in the paper.

The script does not perform any fitting or optimization and uses the same fixed parameters as the main $\Delta X$ analysis.

---

## Parameter Scan: Distance-Only Exponent Test (`scan_onlyD_bn.py`)

This script performs a controlled parameter scan over the distance-only form of the $\Delta X$ mapping. It is included to demonstrate that the mapping’s performance is not the result of fine-tuning and that good agreement persists across a broad region of parameter space.

Specifically, the script explores a grid of values for the distance exponent $b$ and the coupling parameter $N$ while holding all other aspects of the analysis fixed. For each $(b, N)$ pair, the mapping is applied uniformly to the full SPARC sample and evaluated using the shape-based RMSE metric.

This test supports the robustness claims discussed in Section~5.3 of the paper, showing that the preferred parameter values lie within a wide basin of comparable performance rather than at a sharply tuned optimum.

### How to run

Run from the repository root:

```bash
python scripts/scan_onlyD_bn.py
```

The script does not require command-line arguments. Parameter ranges are defined internally and can be adjusted directly in the script if desired.

### Inputs

- SPARC rotation-curve data placed under:
  ```
  data/sparc/
  ```
- Uses the same data-loading and $\Delta X$ construction logic as the main analysis script.

### Output

The script writes a CSV file summarizing performance across the scanned parameter grid:

```
scan_onlyD_bN_results.csv
```

Each row corresponds to a single $(b, N)$ combination and includes:
- median shape-based RMSE across the sample
- mean RMSE
- upper-percentile RMSE
- worst-case RMSE
- number of galaxies included

The best-performing parameter combinations (lowest median shape RMSE) are also printed to the terminal for convenience.

### Notes

- No fitting or optimization is performed within the scan.
- All parameters are applied globally to the full sample for each grid point.
- The scan is intended as a diagnostic and robustness test rather than a primary analysis product.
- Results from this script motivate the fixed parameter choices adopted in the main paper.

---
---

## Outputs

Key derived outputs include:

- Per-galaxy performance metrics (`summary_per_galaxy_*.csv`)
- Sample-level summaries (`summary_sorted_*.csv`)
- Radial ΔX profiles for each galaxy (`radial/`)

These files correspond directly to the tables and figures presented in the paper.

---

## Notes on Reproducibility

All computations are performed **independently at each radius**. No information is shared between radial points prior to metric evaluation.

Small numerical differences may arise due to library versions or platform differences, but all qualitative and quantitative results reported in the paper are robust.

---

## License and Use

This repository is provided for transparency and reproducibility of the published analysis. Users are free to inspect, run, and adapt the code for research purposes, with appropriate citation of the associated paper and the SPARC dataset.
