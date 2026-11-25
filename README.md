# DS203: nation clustering - complete code submission

Start with **[Nation_Clustering.ipynb](Nation_Clustering.ipynb)**. It is executed and includes the figures, tables, numerical results, decision explanations and limitations. All implementation is editable in `analysis.py`.

This repository contains the DS203 nation clustering code. The report and video accompany the separate course submission.

## Quick start

1. Clone this repository and run the commands from its root.
2. Use Python **3.12**, the tested environment. Other Python versions/dependency versions have not been validated.
3. Before regenerating any output, verify the delivered file bytes with `python verify_delivery.py`. Generated PDFs include timestamps, so later byte changes are expected. Create and activate a virtual environment, then install the pinned dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip check
```

4. Either open `Nation_Clustering.ipynb` in your existing Jupyter installation and choose **Restart Kernel and Run All**, or regenerate all analysis outputs from a terminal:

```bash
python analysis.py
```

5. Run the offline validation checks:

```bash
python tests/check_results.py
```

To execute the notebook in a genuinely restarted kernel and recreate generated files from empty output folders, run:

```bash
python execute_notebook.py --clean-outputs
python tests/check_results.py
```

The numerical analysis needs **no network access, API key or missing bulk CSV**. The pinned requirements include IPython, ipykernel, nbclient and nbformat for the real-kernel execution command. A JupyterLab/Notebook user interface is optional; use your existing installation to view/edit the notebook. Runtime on a typical laptop should be short for 205 economies, but varies with the environment. The code caps numerical worker threads for this small problem.

## Repository contents

The notebook contains the figures and results. It links to three included evidence tables. Running the analysis regenerates the full set of plots, tables, and processed data listed below.

| Path | Purpose |
|---|---|
| `Nation_Clustering.ipynb` | Executed, annotated analysis; start here |
| `analysis.py` | Complete staged implementation and command-line entry point |
| `requirements.txt` | Exact tested analysis dependency versions |
| `execute_notebook.py` | Run the notebook in a fresh kernel using the calling interpreter; optionally recreate empty generated folders |
| `verify_delivery.py` | Verify delivered checksums before generating new timestamped PDFs |
| `fetch_data.py` | Optional download into a new snapshot directory; never needed for the frozen run |
| `tests/check_results.py` | Offline integrity, leakage and numerical checks |
| `data/raw/` | Official 2022 numerical snapshot, full downloaded JSON for the 11 indicators, country metadata |
| `data/provenance.json` | Source URLs, dates, file sizes and hashes |
| `data/quality_checks/` | Source response investigating the CAF life-expectancy anomaly |
| `data/processed/` | Observed country tables, imputation flags, standardized features, PCA scores and final country assignments |
| `outputs/figures/` | Required plots and model comparisons in both PNG and PDF |
| `outputs/tables/` | Missingness, exclusions, summaries, PCA parameters, profiles, country cases, comparisons and robustness CSVs |
| `outputs/results.json` | Canonical machine-readable main configuration, results and software versions |
| `DATA_QUALITY.md` | Anomalies, missingness, transformations, metadata dates and interpretation limits |
| `SHA256SUMS.txt` | Integrity manifest for the delivered files |

## Main decisions

**Data:** all 11 indicator IDs from the problem statement, one year (2022), and country/economy rows only. Aggregate entities are removed by official metadata. The main sample is cross-sectional; each economy has equal weight rather than population weight.

**Year:** compare 2021-2024 from the same WDI release using the fixed 217-economy roster. Choose the latest candidate with at least 90% of economies retained and at least half fully observed. This selects 2022 (205 retained, 110 complete cases); 2021 has better coverage (206/116), while 2023 has 202/107 and 2024 has 200/75. The 50% complete-case floor is a transparent design choice, and 2023 narrowly misses it. Tables `30` and `31` provide all coverage and missingness results. No cross-year filling is used.

**Metadata timing:** numerical indicators are for 2022. Income, region and lending labels are the official Country API snapshot retrieved on **23 November 2026**. They describe that snapshot, not historical 2022 classifications. They are external references only and never enter imputation, scaling, PCA or clustering.

**Missingness:** report first, then exclude economies with 6 or more of the 11 indicators missing. For the remaining economies, apply a global median imputer. The threshold is a modeling choice tested through a stricter threshold and complete cases. Individual imputation flags and per-feature observation counts are exported.

**Transformations:** use the natural logarithm of positive GDP per capita to address its strong right skew. Missing GDP is imputed on the logged scale. All other indicators use their original values before standardization. Preserve the original raw-GDP pipeline as an explicit diagnostic; do not choose a transformation just to improve scores or agreement.

**PCA:** standardize all clustering features, then retain the smallest number of principal components explaining at least 80% of variance. Export eigenvector coefficients, explained variance, imputation statistics and scaling parameters. The plotted two-PC view is not the full clustering space.

**Clustering:** K-means with seed 42 and 50 initializations tests k=2 through 8. Select the highest mean silhouette in that finite range; inspect the elbow, sizes and substantive meaning. Ward hierarchical clustering uses the identical representation and k, under the professor's permission to use hierarchical libraries.

**Comparisons:** compare clusters with income and region using counts plus explicitly normalized percentages. Compare the two partitions with crosstabs, ARI/NMI, profiles and country differences. These metrics are not accuracy against ground truth.

**Country cases:** export distance to centroid, within-cluster distance percentile, individual silhouette, observed measurements and imputation counts. Feature-driver summaries describe standardized original-feature deviations, not an exact contribution to PCA-space distance or a causal explanation.

**Source anomaly:** CAF life expectancy is reported as 18.818 in the frozen 2022 source. A point-series query returns the same value and a concerning temporal pattern. It is flagged and retained as reported in the main analysis; its exclusion is explicitly tested. No guessed correction is substituted. See `DATA_QUALITY.md`.

## Verified main output on the frozen data

| Quantity | Result |
|---|---:|
| Matched economies before missingness exclusions | 217 |
| Excluded / retained | 12 / 205 |
| Complete cases before imputation | 110 |
| Values imputed in retained sample | 197 of 2,255 |
| PCA components | 6 |
| Retained variance | 84.67% |
| First two PCs, used for visualization | 51.13% |
| Selected k | 2 |
| K-means group sizes | 60 / 145 |
| Ward group sizes | 58 / 147 |
| K-means / Ward silhouette | 0.3572 / 0.3400 |
| Between-method ARI / NMI | 0.5712 / 0.4212 |

These supersede the raw-GDP baseline **as the main configuration**. The raw-GDP diagnostic reproduces the teammate's 61/144 K-means split, 82.17% PCA retention and between-method ARI of 0.0801. Both results remain available, with their preprocessing explicitly distinguished.

The two-group main K-means partition is identical across ten tested random seeds. K-means also remains identical under the logged-GDP full-feature and 90%-PCA representations in these diagnostics; Ward varies more. The complete-case result concerns a smaller, selected sample. A stable split is evidence about these checks, not proof of uniquely natural country categories.

## Robustness analyses

The exported table includes the main configuration and nine diagnostics: raw GDP, no PCA truncation, 90%-variance PCA, at most three missing indicators, complete cases, k=3, GDP omission, exclusion of the source-quality flag, and retaining CAF while masking only its flagged life-expectancy field. Each refits its imputer, scaler and PCA. Each algorithm is compared with its own main partition on common country codes (`KMeans_ARI_vs_main`, `Ward_ARI_vs_main`, `n_common`). Raw GDP changes K-means little (ARI 0.980) but Ward substantially (ARI 0.073). Table `36` separates complete-case sample restriction from preprocessing refitting on the same 110 economies with k=2; table `37` supplies the country-keyed A/B/C assignments. Removing GDP is a diagnostic, not a replacement of the required 11-feature main model.

**Do not rank silhouettes across different transformed spaces or samples as if they measured the same geometry.** Evaluate persistent conclusions, sample changes and interpretation together.

## Integrity and meaningful checks

The analysis checks snapshot hashes, uniqueness, aggregate exclusion, finite values, positive observed GDP, sample alignment, standardization and the PCA retention threshold. The test script additionally replaces all income/region/lending labels and confirms that preprocessing and both fitted partitions remain unchanged. It joins exported and expected assignments by country code and verifies both model labels. Row shuffling passes; swapping two exported country codes in a temporary copy makes the actual test script fail. It also compares the main assignments with the preserved audited v1 benchmark and verifies the original raw-GDP baseline. The test script prints its validation results.

These checks address specific failure modes. They do not guarantee that every substantive interpretation is correct or that the source data is error-free.

## Optional source refresh

```bash
python fetch_data.py --output new_snapshot
```

The output directory must not already exist. WDI revisions and updated metadata can change values or results. Keep the delivered snapshot intact; inspect changes before adopting a new snapshot and updating the provenance. The download script validates that each response is complete and not silently truncated by pagination.

## Attribution and verification

The raw-GDP diagnostic reproduces the original teammate PCA baseline. The course clarification permits hierarchical-clustering libraries. GitHub Actions runs the checksum verification, command-line analysis and numerical checks, then reruns the notebook in a fresh Jupyter kernel.
