# Data quality and interpretation decisions

The raw snapshots are preserved. The main run uses the source values without undocumented corrections, clipping, deletion of valid extremes, or cross-year filling.

## Central African Republic life expectancy

The frozen WDI series `SP.DYN.LE00.IN` reports **18.818 years** for CAF in 2022. A separate point-series API check on 23 November 2026 returns:

| Year | Source-reported life expectancy, years |
|---|---:|
| 2020 | 50.596 |
| 2021 | 40.279 |
| 2022 | 18.818 |
| 2023 | 57.408 |
| 2024 | 57.671 |

This extreme temporal pattern requires scrutiny. The code flags it as a source-quality concern, without claiming that its true replacement is known. The primary analysis retains the reported value; the `Source-quality flag excluded` robustness run removes the flagged economy and refits all preprocessing and models. Refer to `outputs/tables/24_robustness.csv` for the effect on common-country assignments. The distribution and pairwise plots label the flagged observation. Do not describe 18.818 as an independently verified demographic fact, or attribute the dip to a cause without evidence.

Point-series source: https://api.worldbank.org/v2/country/CAF/indicator/SP.DYN.LE00.IN?date=2020:2024&format=json&per_page=20

The downloaded response is preserved in `data/quality_checks/caf_life_expectancy_2020_2024.json`. Only 2022 enters the main cross-sectional features; other years are a quality check.

## Missingness

- Exclude economies missing more than half of the 11 required indicators (6 or more missing); retain those with at least 6 observed indicators.
- Report missingness before exclusion and all excluded economies. The threshold is a transparent design choice, not a statistically proven optimum.
- Use global medians of retained economies, after the GDP log transform. Never calculate medians within income groups or regions.
- Retain flags identifying every imputed value. Observed-only cluster summaries include per-feature counts.
- Test a stricter threshold, complete cases, and the original teammate's raw-GDP configuration. The smaller complete-case sample can be selected and unrepresentative.

## Scale and transformations

GDP per capita is positive in the observed data and strongly right-skewed. The primary transform is the natural logarithm, followed by median imputation and z-score standardization. A log axis in descriptive plots is a display choice; the model's natural-log transformation is separately explicit. All other indicators retain their supplied measurement scale before standardization. Negative growth and inflation are allowed. Trade-to-GDP and gross enrollment can exceed 100%; gross capital formation can reflect negative inventory investment. Unusual values are investigated, not automatically clipped.

## Labels and time

The 2022 numerical indicators are compared with country metadata retrieved on 23 November 2026. Income groups and regions are reference labels from that snapshot, **not reconstructed 2022 classifications**. They do not enter the model. Dates matter especially for countries whose income groups or regions have changed. This comparison does not establish historical misclassification.

## Modeling limits

PCA retains at least 80% of standardized-feature variance. Large variance is not identical to substantive importance. Two plotted PCs omit variation used by the six-dimensional model. Cluster count is selected among k=2 through 8, not over every possible partition. Ward and K-means use the same representation and k, but their objectives differ. ARI/NMI quantify agreement, not accuracy. Silhouettes in different transformed spaces or samples must not be ranked as universal measures of quality. The study is descriptive, uses one year, weights economies equally, and does not establish causation.

## Sensitivity to the CAF life expectancy value

A second diagnostic retains CAF and masks only its life-expectancy field before using the same global median-imputation procedure and refitting scaling/PCA. It retains 205 economies and imputes 198 cells. Relative to each algorithm's own main partition, K-means ARI is 0.980 and Ward ARI is 0.618. Dropping CAF gives 204 common economies and ARIs 0.980 and 0.960 respectively. The two treatments answer different questions; the higher sensitivity of Ward to field masking is explicitly retained in the interpretation. Neither replaces the source observation in the primary fit.
