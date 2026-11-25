# DS203 Nation Clustering

This is our DS203 project, where we tried to group 205 countries/economies using 11 World Bank indicators. We used PCA first and then ran K-means and Ward (hierarchical) clustering on it. Everything is explained in the [notebook](Nation_Clustering.ipynb), and the actual pipeline code is in `analysis.py`.

The report and video are submitted separately for the course, so this repo just has the code (plus the data and outputs).

## How to run it

1. Clone the repo and run all the commands from the root folder.
2. Use Python 3.12, since that's what we tested on. We haven't checked other Python versions or other library versions, so they might work but we can't promise anything.
3. Before you regenerate anything, run `python verify_delivery.py` to check that the files match what we submitted. (The generated PDFs have timestamps in them, so after you regenerate they won't match byte-for-byte anymore, which is expected.)

   Then make a virtual environment, activate it, and install the pinned dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip check
```

4. Now you can either open `Nation_Clustering.ipynb` in Jupyter and do Restart Kernel and Run All, or just run the script from the terminal to regenerate all the outputs:

```bash
python analysis.py
```

5. Run the checks (these work offline):

```bash
python tests/check_results.py
```

If you want to run the notebook in a fresh kernel and rebuild all the generated files starting from empty output folders, use:

```bash
python execute_notebook.py --clean-outputs
python tests/check_results.py
```

Once the dependencies are installed you don't need internet, because the data is already saved in the repo.

## What's in the repo

The notebook has all the figures and results, and it links to three of the evidence tables. Running the analysis regenerates all the plots, tables and processed data listed below.

| Path | What it is |
|---|---|
| `Nation_Clustering.ipynb` | The executed notebook with explanations. Start here |
| `analysis.py` | The full pipeline, step by step, and can be run from the command line |
| `requirements.txt` | Exact library versions we tested with |
| `execute_notebook.py` | Runs the notebook in a fresh kernel using the same Python you call it with. Can also empty and rebuild the generated folders |
| `verify_delivery.py` | Checks the checksums of the submitted files (run it before generating new PDFs) |
| `fetch_data.py` | Optional, downloads data into a new snapshot folder. Not needed for the normal run |
| `tests/check_results.py` | Offline checks for data integrity, leakage and the numbers |
| `data/raw/` | The official 2022 data snapshot, the full downloaded JSON for all 11 indicators, and country metadata |
| `data/provenance.json` | Source URLs, dates, file sizes and hashes |
| `data/quality_checks/` | The source response we used to look into the CAF life expectancy issue |
| `data/processed/` | Country tables, imputation flags, standardized features, PCA scores and the final cluster for each country |
| `outputs/figures/` | All the required plots and model comparisons, in PNG and PDF |
| `outputs/tables/` | CSVs for missingness, exclusions, summaries, PCA parameters, profiles, country cases, comparisons and robustness |
| `outputs/results.json` | Main config, results and software versions in one machine-readable file |
| `DATA_QUALITY.md` | Notes on anomalies, missing data, transformations, metadata dates and limitations |
| `SHA256SUMS.txt` | Checksums for all the submitted files |

## Choices we made

**Data:** We used all 11 indicator IDs from the problem statement, only for the year 2022, and only actual countries/economies. Aggregate rows (like regional or income-group totals) are removed using the official metadata. It's a single-year analysis, and every economy counts equally (we didn't weight by population).

**Which year:** We compared 2021 to 2024, all from the same WDI release and the same fixed list of 217 economies. Our rule was to pick the latest year where at least 90% of the 217 economies are kept and at least half of the 217 have no missing values. That gave us 2022 (205 kept, 110 complete). 2021 actually had slightly better coverage (206 kept, 116 complete), 2023 had 202/107 and 2024 had 200/75. The 50% complete-case cutoff was our own choice, and 2023 just barely misses it. All the coverage and missingness numbers are in tables `30` and `31`. We didn't fill in missing values using other years.

**Metadata dates:** The indicator values are for 2022, but the income group, region and lending labels come from the World Bank Country API as downloaded on 23 November 2026. So they're the labels from that date, not what they were in 2022. We only use them afterwards to compare with the clusters. They're never used in imputation, scaling, PCA or clustering.

**Missing values:** We look at missingness first, then drop any economy that's missing 6 or more of the 11 indicators. For the rest, missing values are filled with the median (one global median imputer). The cutoff of 6 is a choice we made, so we also tried a stricter cutoff and complete cases only. The imputation flags for each value and the number of observed values per feature are exported too.

**Transformations:** GDP per capita is really right-skewed, so we take its natural log (all the observed values are positive). Missing GDP values are imputed after taking the log. All the other indicators are used as they are before standardizing. We also kept the original raw-GDP version as a diagnostic, and we didn't pick the transformation based on which one gave better scores or agreement.

**PCA:** All the features are standardized, and then we keep the smallest number of principal components that explain at least 80% of the variance. We export the eigenvector coefficients, explained variance, imputation stats and scaling parameters. Note that the 2D PC plot is only for visualization. The clustering uses all the retained components, not just the first two.

**Clustering:** We ran K-means (seed 42, 50 initializations) for k = 2 to 8 and picked the k with the highest mean silhouette score in that range. We also looked at the elbow plot, the cluster sizes and whether the groups actually make sense. Ward hierarchical clustering uses the exact same data and the same k (the professor allowed us to use libraries for hierarchical clustering).

**Comparisons:** We compare the clusters with income group and region using counts and also percentages (with the normalization clearly stated). To compare K-means and Ward with each other we used crosstabs, ARI/NMI, cluster profiles, and the countries that end up in different groups. These aren't accuracy scores, since there's no "correct" grouping to compare against.

**Country cases:** For each country we export its distance to the cluster centroid, the percentile of that distance within its cluster, its individual silhouette score, its actual measured values and how many of its values were imputed. The "feature driver" summaries show the standardized deviations of the original features. They aren't an exact breakdown of the distance in PCA space, and they don't explain cause and effect.

**Data issue:** In the 2022 data we saved, CAF (Central African Republic) life expectancy is listed as 18.818, which is suspicious. Querying just that one series gives the same value, and the trend over the years looks off too. We flagged it but kept it as reported in the main analysis, and we separately tested what happens when it's excluded. We didn't make up a corrected value. More details are in `DATA_QUALITY.md`.

## Main results (on the saved data)

| Quantity | Result |
|---|---:|
| Matched economies before dropping for missingness | 217 |
| Dropped / kept | 12 / 205 |
| Complete cases before imputation | 110 |
| Values imputed in the kept sample | 197 of 2,255 |
| PCA components | 6 |
| Variance retained | 84.67% |
| First two PCs (used for plots) | 51.13% |
| Chosen k | 2 |
| K-means group sizes | 60 / 145 |
| Ward group sizes | 58 / 147 |
| K-means / Ward silhouette | 0.3572 / 0.3400 |
| ARI / NMI between the two methods | 0.5712 / 0.4212 |

If we use GDP per capita without the log, we get back our teammate's original baseline: a 61/144 K-means split, 82.17% variance retained by PCA, and an ARI of 0.0801 between the two methods.

The main two-group K-means result is exactly the same for all ten random seeds we tried. K-means also gives the same result with logged GDP on the full features (no PCA truncation) and with 90%-variance PCA, but Ward changes more. The complete-case result is on a smaller, selected sample, so keep that in mind. Also, a stable split just means it held up in these checks. It doesn't prove that there are uniquely "natural" country categories.

## Robustness checks

Along with the main setup, the robustness table has nine diagnostics:

- raw GDP (no log)
- no PCA truncation
- 90%-variance PCA
- at most three missing indicators
- complete cases only
- k = 3
- leaving out GDP
- excluding the economy flagged for the data issue
- keeping CAF but masking only its flagged life expectancy value

Each one refits its own imputer, scaler and PCA. Each algorithm is compared with its own main result on the countries both runs have in common (columns `KMeans_ARI_vs_main`, `Ward_ARI_vs_main`, `n_common`). Raw GDP barely changes K-means (ARI 0.980) but changes Ward a lot (ARI 0.073). Table `36` separates the effect of restricting to complete cases from the effect of refitting the preprocessing on the same 110 economies with k = 2, and table `37` has the A/B/C assignments for each country. Leaving out GDP is only a diagnostic, not a replacement for the main 11-feature model.

**One thing to note:** silhouette scores from different transformed spaces or samples shouldn't be ranked against each other like they're measuring the same thing. It makes more sense to look at which conclusions hold up, how the sample changed, and the interpretation all together.

## Tests

The analysis itself checks the snapshot hashes, duplicates, that aggregates are removed, that all values are finite, that observed GDP is positive, that the samples line up, the standardization, and the 80% PCA threshold.

The test script also:

- replaces all the income/region/lending labels and checks that the preprocessing and both clusterings stay the same
- matches the exported and expected assignments by country code and checks both models' labels (shuffling the rows still passes, and swapping two country codes in a temporary copy makes the test fail, like it should)
- compares the main assignments with the saved audited v1 benchmark
- checks the original raw-GDP baseline

It prints the results of each check when you run it.

These tests only catch specific kinds of problems. They don't guarantee that every interpretation is right or that the source data has no errors.

## Downloading fresh data (optional)

```bash
python fetch_data.py --output new_snapshot
```

The output folder shouldn't already exist. WDI revisions and updated metadata can change the values or results, so keep the original snapshot as it is, and check the differences before switching to a new snapshot and updating the provenance. The script also checks that each response is complete and didn't get cut off by pagination.

## Credits and CI

The raw-GDP diagnostic reproduces our teammate's original PCA baseline. Using hierarchical clustering libraries was allowed as per the course clarification. GitHub Actions runs the checksum check, the analysis script and the numerical checks, and then reruns the notebook in a fresh Jupyter kernel.
