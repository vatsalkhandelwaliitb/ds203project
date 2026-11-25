"""DS203 nation clustering. Run `python analysis.py` for the complete offline analysis.

All models use numerical WDI indicators only. Categorical metadata is held out.
The main model uses log GDP per capita, global median imputation, StandardScaler,
PCA retaining >=80% variance, and K-means/Ward at the same selected cluster count.
Raw GDP and other plausible configurations are explicit sensitivity analyses.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import textwrap
from pathlib import Path

# Tiny country tables need one worker; also avoids ambiguous CPU discovery in containers.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import sklearn
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "outputs/figures"
TAB = ROOT / "outputs/tables"
PROC = ROOT / "data/processed"
INDICATORS = {
    "GDP_PC": "NY.GDP.PCAP.CD", "GDP_Growth": "NY.GDP.MKTP.KD.ZG",
    "Inflation": "FP.CPI.TOTL.ZG", "Unemployment": "SL.UEM.TOTL.ZS",
    "Trade": "NE.TRD.GNFS.ZS", "Capital_Formation": "NE.GDI.TOTL.ZS",
    "Life_Expectancy": "SP.DYN.LE00.IN", "Urban_Population": "SP.URB.TOTL.IN.ZS",
    "Internet_Users": "IT.NET.USER.ZS", "Electricity_Access": "EG.ELC.ACCS.ZS",
    "Secondary_Enrollment": "SE.SEC.ENRR",
}
FEATURES = list(INDICATORS)
HELD_OUT = ["IncomeGroup", "Region", "LendingType"]
LABELS = ["GDP/person", "GDP growth", "Inflation", "Unemployment", "Trade",
          "Capital formation", "Life expectancy", "Urban population", "Internet users",
          "Electricity access", "Secondary enrollment"]
INCOMES = ["Low income", "Lower middle income", "Upper middle income", "High income"]
COLORS = ["#d4513b", "#d99520", "#239b94", "#304c89"]
SHORT_INCOMES = ["Low", "Lower middle", "Upper middle", "High"]
CLUSTER_COLORS = ["#b94d45", "#237c86", "#a88622", "#6850a1", "#426749", "#a36488", "#4d6c9d", "#777777"]
CONFIG = {"year": 2022, "missing_fraction_max": 0.5, "log_gdp": True,
          "pca_variance_target": 0.8, "k_values": list(range(2, 9)),
          "random_state": 42, "n_init": 50, "hierarchical_linkage": "ward",
          "candidate_years": [2021, 2022, 2023, 2024],
          "year_min_retained_fraction": 0.90, "year_min_complete_fraction": 0.50}
_UNSET = object()


def table(df: pd.DataFrame | pd.Series, name: str, index: bool = True) -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / f"{name}.csv", index=index)


def savefig(name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIG / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close()


def load_data() -> dict:
    """Load frozen official data; verify integrity and country-level structure."""
    for p in (FIG, TAB, PROC):
        p.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook", font_scale=0.95)
    plt.rcParams.update({"figure.facecolor": "white", "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})
    provenance = json.loads((ROOT / "data/provenance.json").read_text())
    for name, info in provenance["files"].items():
        assert hashlib.sha256((ROOT / "data/raw" / name).read_bytes()).hexdigest() == info["sha256"], name
    raw = pd.read_csv(ROOT / f"data/raw/wdi_{CONFIG['year']}.csv", keep_default_na=True)
    raw = raw.loc[raw["Indicator Code"].isin(INDICATORS.values())].copy()
    assert not raw.duplicated(["Country Name", "Country Code", "Indicator Code"]).any()
    assert set(raw["Indicator Code"]) == set(INDICATORS.values())
    # Some API aggregate entries lack ISO3 codes. They will never match a valid economy.
    raw["Country Code"] = raw["Country Code"].fillna("")
    wide = raw.pivot(index=["Country Name", "Country Code"], columns="Indicator Code", values=str(CONFIG["year"])).reset_index()
    wide = wide.rename(columns={v: k for k, v in INDICATORS.items()})
    api = json.loads((ROOT / "data/raw/country_metadata.json").read_text())
    assert int(api[0]["pages"]) == 1 and len(api[1]) == int(api[0]["total"])
    meta_all = pd.DataFrame([{
        "Country Code": c["id"], "Region": c["region"]["value"].strip(),
        "IncomeGroup": c["incomeLevel"]["value"].strip(),
        "LendingType": c["lendingType"]["value"].strip(),
    } for c in api[1]])
    assert meta_all["Country Code"].is_unique
    meta = meta_all.loc[meta_all.Region != "Aggregates"].copy()
    assert not meta["Country Code"].eq("").any()
    reconcile = wide[["Country Name", "Country Code"]].copy()
    code_region = meta_all.set_index("Country Code").Region
    iso2_region = {c["iso2Code"]: c["region"]["value"] for c in api[1]}
    source = json.loads((ROOT / f"data/raw/wdi_{CONFIG['year']}_api.json").read_text())
    source_ids = {x["country"]["value"]: x["country"]["id"] for x in source[1]}
    reconcile["API_country_id"] = reconcile["Country Name"].map(source_ids)
    def classify(row):
        code = row["Country Code"]
        if code in set(meta["Country Code"]): return "eligible_economy"
        if code and code_region.get(code) == "Aggregates": return "coded_aggregate"
        if not code and iso2_region.get(row["API_country_id"]) == "Aggregates": return "uncoded_income_aggregate"
        return "UNMATCHED_NONBLANK_CODE" if code else "UNMATCHED_UNCODED_ENTITY"
    reconcile["status"] = reconcile.apply(classify, axis=1)
    table(reconcile, "28_entity_reconciliation", index=False)
    unknown = reconcile.loc[reconcile.status.str.startswith("UNMATCHED")]
    table(unknown, "28_unmatched_wdi_entities", index=False)
    if len(unknown):
        raise ValueError("Unmatched WDI entities require review; see 28_unmatched_wdi_entities.csv")
    eligible = wide.loc[wide["Country Code"].isin(meta["Country Code"])].copy()
    countries = eligible.merge(meta, on="Country Code", how="inner", validate="one_to_one")
    countries = countries.sort_values(["Country Name", "Country Code"]).reset_index(drop=True)
    countries[FEATURES] = countries[FEATURES].apply(pd.to_numeric, errors="raise")
    assert countries["Country Code"].is_unique
    assert not countries.Region.eq("Aggregates").any()
    assert len(FEATURES) == 11 and not set(FEATURES).intersection(HELD_OUT)
    assert not np.isinf(countries[FEATURES].to_numpy()).any()
    assert countries.GDP_PC.dropna().gt(0).all(), "Log GDP requires positive observations."
    assert countries[FEATURES].notna().any(axis=0).all()
    missing_n = countries[FEATURES].isna().sum(axis=1)
    row_audit = countries[["Country Code", "Country Name"] + HELD_OUT].copy()
    row_audit["missing_indicators"] = missing_n
    row_audit["missing_fraction"] = missing_n / len(FEATURES)
    row_audit["retained"] = row_audit.missing_fraction <= CONFIG["missing_fraction_max"]
    row_audit["reason"] = np.where(row_audit.retained, "Retained", f"Missing fraction exceeds {CONFIG['missing_fraction_max']:.0%}")
    quality = countries.loc[countries.Life_Expectancy < 40, ["Country Code", "Country Name", "Life_Expectancy"]].copy()
    quality["review_flag"] = "Source-reported life expectancy below 40 years: investigate; no invented correction"
    quality["main_treatment"] = "Retained as reported; flagged-economy exclusion tested separately"
    table(quality, "27_source_quality_flags", index=False)
    table(row_audit, "01_country_coverage_audit", index=False)
    table(row_audit.loc[~row_audit.retained], "02_excluded_economies", index=False)
    table(pd.DataFrame({"missing_count": countries[FEATURES].isna().sum(),
                        "missing_percent": countries[FEATURES].isna().mean() * 100}), "03_feature_missingness")
    retained = countries.loc[row_audit.retained].copy().reset_index(drop=True)
    countries.to_csv(PROC / f"country_table_observed_{CONFIG['year']}.csv", index=False)
    retained.to_csv(PROC / f"retained_observed_{CONFIG['year']}.csv", index=False)
    table(pd.DataFrame({"before": countries.IncomeGroup.value_counts(),
                        "retained": retained.IncomeGroup.value_counts()}).fillna(0).astype(int), "04_income_selection")
    unmatched = meta.loc[~meta["Country Code"].isin(eligible["Country Code"])]
    table(unmatched, "05_metadata_economies_without_wdi_rows", index=False)
    counts = {"wdi_entities": len(wide), "metadata_economies": len(meta), "matched_economies": len(countries),
              "excluded": len(countries) - len(retained), "retained": len(retained),
              "complete_cases": int(countries[FEATURES].notna().all(axis=1).sum()),
              "coded_aggregates": int(reconcile.status.eq("coded_aggregate").sum()),
              "uncoded_income_aggregates": int(reconcile.status.eq("uncoded_income_aggregate").sum())}
    assert counts["wdi_entities"] == counts["coded_aggregates"] + counts["uncoded_income_aggregates"] + counts["matched_economies"]
    table(pd.DataFrame([{"stage": k, "n": counts[k]} for k in ["wdi_entities", "coded_aggregates", "uncoded_income_aggregates", "matched_economies", "excluded", "retained"]]), "29_country_reconciliation_flow", index=False)
    print("DATA AUDIT:", counts)
    print("Income/region labels describe the metadata snapshot, not historical indicator-year classifications.")
    return {"countries": countries, "retained": retained, "coverage": row_audit,
            "counts": counts, "provenance": provenance, "quality_flags": quality,
            "reconciliation": reconcile, "config": dict(CONFIG), "metadata_roster": meta}


def candidate_years(s: dict) -> None:
    """Compare all years in one API release and one fixed official roster."""
    cfg = s["config"]
    api = json.loads((ROOT / "data/raw/wdi_candidate_years_2021_2024.json").read_text())
    assert int(api[0]["pages"]) == 1 and len(api[1]) == int(api[0]["total"])
    assert api[0].get("lastupdated") == s["provenance"]["wdi_lastupdated"], "Candidate and main release dates differ."
    rows = pd.DataFrame([{"Country Code": x["countryiso3code"], "indicator": x["indicator"]["id"],
                          "year": int(x["date"]), "value": x["value"]} for x in api[1]])
    codes = s["metadata_roster"]["Country Code"].tolist()
    records, per_feature = [], []
    for year in cfg["candidate_years"]:
        sub = rows.loc[(rows.year == year) & rows["Country Code"].isin(codes)]
        assert not sub.duplicated(["Country Code", "indicator"]).any()
        wide = sub.pivot(index="Country Code", columns="indicator", values="value").reindex(index=codes, columns=list(INDICATORS.values()))
        available = wide.notna().sum(axis=1)
        missing = wide.isna().mean()
        nret = int(((1 - available / len(FEATURES)) <= cfg["missing_fraction_max"] + 1e-12).sum())
        ncomplete = int((available == len(FEATURES)).sum())
        records.append({"year": year, "eligible_roster": len(codes), "complete_cases": ncomplete,
                        "at_least_six_observed": int((available >= 6).sum()), "retained_at_main_cutoff": nret,
                        "missing_cells_percent": float(missing.mean() * 100),
                        "worst_covered_indicator": {v: k for k, v in INDICATORS.items()}[missing.idxmax()],
                        "worst_indicator_missing_percent": float(missing.max() * 100),
                        "meets_coverage_policy": nret / len(codes) >= cfg["year_min_retained_fraction"] and ncomplete / len(codes) >= cfg["year_min_complete_fraction"]})
        per_feature.extend({"year": year, "indicator": {v: k for k, v in INDICATORS.items()}[ind],
                            "observed": int(wide[ind].notna().sum()), "missing_percent": float(missing[ind] * 100)} for ind in wide)
        if year == cfg["year"]:
            current = s["countries"].set_index("Country Code")[FEATURES].reindex(codes)
            candidate = wide.rename(columns={v: k for k, v in INDICATORS.items()})[FEATURES]
            assert np.allclose(current.to_numpy(), candidate.to_numpy(), equal_nan=True), "Main and candidate source releases differ. Review revisions."
    result = pd.DataFrame(records)
    selected = int(result.loc[result.meets_coverage_policy, "year"].max())
    assert selected == cfg["year"], "Coverage policy selects a different year; update main inputs/configuration and regenerate."
    result["selected"] = result.year.eq(selected)
    table(result, "30_candidate_year_coverage", index=False)
    table(pd.DataFrame(per_feature), "31_candidate_year_feature_coverage", index=False)
    s["year_comparison"] = result
    print("Coverage policy: latest candidate retaining at least", f"{cfg['year_min_retained_fraction']:.0%}", "of the fixed roster with at least", f"{cfg['year_min_complete_fraction']:.0%}", "fully observed.")
    print(result.to_string(index=False))
    print(f"Selected {selected}; 2021 has better coverage, but is older. The complete-case floor is a transparent choice, not an optimum.")


def eda(s: dict) -> None:
    """All requested EDA uses observed measurements before imputation."""
    d = s["countries"]
    desc = d[FEATURES].describe(percentiles=[.25, .5, .75]).T
    desc["skewness"] = d[FEATURES].skew()
    desc["missing_percent"] = d[FEATURES].isna().mean() * 100
    table(desc, "06_observed_summary_statistics")
    extrema = []
    for f in FEATURES:
        a = d.loc[d[f].notna(), ["Country Code", "Country Name", f]].sort_values(f)
        for kind, rows in [("lowest", a.head(3)), ("highest", a.tail(3))]:
            for _, row in rows.iterrows():
                extrema.append({"feature": f, "extreme": kind, "country": row["Country Name"], "value": row[f]})
    table(pd.DataFrame(extrema), "07_observed_extremes", index=False)
    missing = (d[FEATURES].isna().mean() * 100).sort_values()
    plt.figure(figsize=(10, 5.4)); missing.rename(index=dict(zip(FEATURES, LABELS))).plot.barh(color="#b8763b")
    plt.xlabel("Missing observations (%)"); plt.title("Missingness before exclusions: all matched economies")
    savefig("01_missingness")
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2))
    fs = ["GDP_PC", "Inflation", "Unemployment", "Life_Expectancy", "Internet_Users", "Electricity_Access"]
    units = ["GDP per person (current US$)", "Inflation (%)", "Unemployment (%)", "Life expectancy (years)", "Internet users (%)", "Electricity access (%)"]
    for ax, f, label in zip(axes.flat, fs, units):
        sns.histplot(d[f].dropna(), bins=25, ax=ax, color="#237c86")
        ax.set(xlabel=label, ylabel="Observed economies", title=f"n = {d[f].notna().sum()}")
        if f == "Life_Expectancy" and len(s["quality_flags"]):
            ax.text(.03, .92, "CAF 18.8: source value flagged", transform=ax.transAxes, fontsize=8)
    fig.suptitle(f"Observed {s['config']['year']} distributions; valid extremes retained", fontsize=15)
    savefig("02_required_distributions")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    g = s["retained"].GDP_PC.dropna()
    sns.histplot(g, bins=25, ax=axes[0], color="#b94d45")
    sns.histplot(np.log(g), bins=25, ax=axes[1], color="#237c86")
    axes[0].set(title=f"Raw GDP/person: skew {g.skew():.2f}", xlabel="Current US$")
    axes[1].set(title=f"Natural log: skew {np.log(g).skew():.2f}", xlabel="ln(GDP/person in current US$)")
    savefig("03_gdp_transform")
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5))
    for ax, f, label in zip(axes.flat, ["GDP_PC", "Internet_Users", "Life_Expectancy", "Electricity_Access"],
                            ["GDP/person (current US$, log axis)", "Internet users (%)", "Life expectancy (years)", "Electricity access (%)"]):
        sns.boxplot(data=d, x="IncomeGroup", y=f, hue="IncomeGroup", order=INCOMES, hue_order=INCOMES,
                    palette=COLORS, legend=False, ax=ax, fliersize=2)
        ax.set(xlabel="Metadata snapshot income group", ylabel=label)
        ax.set_xticks(range(4), SHORT_INCOMES)
        if f == "GDP_PC": ax.set_yscale("log")
    savefig("04_income_group_comparisons")
    corr = d[FEATURES].corr()
    counts = d[FEATURES].notna().astype(int).T @ d[FEATURES].notna().astype(int)
    table(corr, "08_observed_pairwise_correlations"); table(counts, "09_pairwise_observation_counts")
    strong = pd.DataFrame([{"feature_1": a, "feature_2": b, "pearson_r": corr.loc[a, b],
                            "n_observed_pairs": int(counts.loc[a, b])}
                           for i, a in enumerate(FEATURES) for b in FEATURES[i+1:]])
    strong = strong.sort_values("pearson_r", key=abs, ascending=False).reset_index(drop=True)
    table(strong, "32_observed_correlations_ranked", index=False)
    s["strong_correlations"] = strong
    print("Strongest observed pairs (pairwise sample sizes):\n", strong.head(6).round(3).to_string(index=False))
    print("Overlapping development indicators can give their shared signal more weight in Euclidean distance; standardization does not remove correlation.")
    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    sns.heatmap(corr, vmin=-1, vmax=1, center=0, cmap="vlag", annot=True, fmt=".2f",
                xticklabels=LABELS, yticklabels=LABELS, annot_kws={"fontsize": 7}, ax=ax)
    ax.set_title("Observed-data Pearson correlations; pairwise available observations")
    savefig("05_observed_correlations")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    pairs = [("GDP_PC", "Life_Expectancy"), ("GDP_PC", "Internet_Users"),
             ("Secondary_Enrollment", "Life_Expectancy"), ("GDP_PC", "Urban_Population")]
    for ax, (x, y) in zip(axes.flat, pairs):
        sns.scatterplot(data=d, x=x, y=y, hue="IncomeGroup", hue_order=INCOMES,
                        palette=COLORS, s=29, alpha=.8, ax=ax, legend=False)
        if x == "GDP_PC": ax.set_xscale("log")
        ax.set_xlabel("GDP/person (US$, log axis)" if x == "GDP_PC" else "Secondary enrollment (gross %)")
        ax.set_ylabel({"Life_Expectancy": "Life expectancy (years)", "Internet_Users": "Internet users (%)", "Urban_Population": "Urban population (%)"}[y])
        if y == "Life_Expectancy" and len(s["quality_flags"]):
            for _, point in d.loc[d.Life_Expectancy < 40].iterrows():
                ax.annotate(f"{point['Country Code']}: source flagged", (point[x], point[y]),
                            xytext=(7, 8), textcoords="offset points", fontsize=7)
    handles = [plt.Line2D([], [], marker="o", color=c, linestyle="", label=l) for c, l in zip(COLORS, SHORT_INCOMES)]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False)
    fig.subplots_adjust(top=.90)
    savefig("06_required_pairwise_relationships")
    print("EDA: six distributions, four income comparisons, four requested scatter plots, correlations and observation counts exported.")
    print("Trade/enrollment can exceed 100%; growth and inflation can be negative. Valid extremes are retained.")
    s["descriptive_statistics"] = desc


def representation(d: pd.DataFrame, *, config: dict | None = None, log_gdp=None, pca_target=_UNSET,
                   features: list[str] | None = None) -> dict:
    """Fit numerical preprocessing without reading any metadata/target columns."""
    fs = FEATURES if features is None else features
    cfg = dict(CONFIG if config is None else config)
    if log_gdp is not None: cfg["log_gdp"] = log_gdp
    if pca_target is not _UNSET: cfg["pca_variance_target"] = pca_target
    log_gdp, pca_target = cfg["log_gdp"], cfg["pca_variance_target"]
    assert not set(fs).intersection(HELD_OUT)
    transformed = d[fs].copy()
    if log_gdp and "GDP_PC" in fs:
        assert transformed.GDP_PC.dropna().gt(0).all()
        transformed["GDP_PC"] = np.log(transformed["GDP_PC"])
    imputer = SimpleImputer(strategy="median")
    imputed = imputer.fit_transform(transformed)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(imputed)
    model_corr = pd.DataFrame(scaled, columns=fs).corr()
    pca = PCA(svd_solver="full").fit(scaled)
    if pca_target is None:
        ncomp = len(fs); X = scaled
    else:
        ncomp = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), pca_target) + 1)
        X = pca.transform(scaled)[:, :ncomp]
    return {"features": fs, "transformed": transformed, "imputer": imputer, "imputed": imputed,
            "scaler": scaler, "scaled": scaled, "model_corr": model_corr, "pca": pca,
            "n_components": ncomp, "X": X, "config": cfg}


def preprocess(s: dict) -> None:
    d = s["retained"]
    s["rep"] = r = representation(d, config=s["config"])
    ev = r["pca"].explained_variance_ratio_
    var = pd.DataFrame({"PC": np.arange(1, 12), "explained_variance_ratio": ev,
                        "cumulative_variance": ev.cumsum()})
    table(var, "10_pca_variance", index=False)
    load = pd.DataFrame(r["pca"].components_.T, index=FEATURES, columns=[f"PC{i}" for i in range(1, 12)])
    table(load, "11_pca_direction_coefficients")
    table(r["model_corr"], "12_model_feature_correlations")
    prep = pd.DataFrame({"feature": FEATURES, "transform": ["ln(x)" if x == "GDP_PC" and r["config"]["log_gdp"] else "identity" for x in FEATURES],
                         "imputation_median_transformed_units": r["imputer"].statistics_,
                         "scaler_mean": r["scaler"].mean_, "scaler_scale": r["scaler"].scale_,
                         "imputed_count": d[FEATURES].isna().sum().to_numpy()})
    table(prep, "13_fitted_preprocessing_parameters", index=False)
    pd.DataFrame(d[FEATURES].isna().to_numpy(), columns=FEATURES, index=d["Country Code"]).to_csv(PROC / "imputation_flags.csv")
    pd.DataFrame(r["scaled"], columns=FEATURES, index=d["Country Code"]).to_csv(PROC / "standardized_model_features.csv")
    pd.DataFrame(r["X"], columns=[f"PC{i}" for i in range(1, r["n_components"] + 1)], index=d["Country Code"]).to_csv(PROC / "pca_scores.csv")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(var.PC, ev, color="#93bdc3", label="Individual variance")
    ax.plot(var.PC, ev.cumsum(), "o-", color="#237c86", label="Cumulative variance")
    ax.axhline(r["config"]["pca_variance_target"], linestyle="--", color="#b94d45", label=f"{r['config']['pca_variance_target']:.0%} retention rule")
    ax.set(xticks=np.arange(1, 12), xlabel="Principal component", ylabel="Fraction of variation", ylim=(0, 1.05))
    ax.legend(frameon=False, fontsize=9); savefig("07_pca_variance")
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(load.iloc[:, :r["n_components"]], center=0, cmap="vlag", annot=True, fmt=".2f",
                yticklabels=LABELS, ax=ax)
    ax.set_title(f"PCA direction coefficients; GDP/person is {'logged' if r['config']['log_gdp'] else 'raw'}")
    savefig("08_pca_coefficients")
    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    sns.heatmap(r["model_corr"], center=0, vmin=-1, vmax=1,
                cmap="vlag", annot=True, fmt=".2f", xticklabels=LABELS, yticklabels=LABELS,
                annot_kws={"fontsize": 7}, ax=ax)
    ax.set_title(f"Model-feature correlations: {'logged' if r['config']['log_gdp'] else 'raw'} GDP; median imputation")
    savefig("09_model_correlations")
    s["loadings"] = load
    print(f"PCA: {r['n_components']} components retain {ev[:r['n_components']].sum():.2%}; first two retain {ev[:2].sum():.2%}.")
    print(f"Preceding {r['n_components']-1} PCs retain {ev[:r['n_components']-1].sum():.2%}; this is below the configured threshold.")
    print("Coefficient signs are arbitrary. Large absolute coefficients identify variables defining each direction.")
    print(load.iloc[:, :3].round(3).to_string())


def canonical_labels(labels: np.ndarray, d: pd.DataFrame) -> np.ndarray:
    """Number clusters by increasing observed median GDP, solely for readable names."""
    med = pd.DataFrame({"label": labels, "GDP_PC": d.GDP_PC.to_numpy()}).groupby("label").GDP_PC.median().sort_values()
    mapping = {int(old): i + 1 for i, old in enumerate(med.index)}
    return np.array([mapping[int(x)] for x in labels])


def fit_clusters(d: pd.DataFrame, r: dict, *, fixed_k: int | None = None) -> dict:
    X = r["X"]
    cfg = r["config"]
    records = []; models = []
    for k in cfg["k_values"]:
        km = KMeans(n_clusters=k, random_state=cfg["random_state"], n_init=cfg["n_init"]).fit(X)
        models.append(km)
        records.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(X, km.labels_),
                        "smallest_cluster": int(np.bincount(km.labels_).min()),
                        "largest_cluster": int(np.bincount(km.labels_).max())})
    candidates = pd.DataFrame(records)
    best = int(candidates.silhouette.to_numpy().argmax())
    if fixed_k is not None:
        best = int(np.flatnonzero(candidates.k.to_numpy() == fixed_k)[0])
    km = models[best]; k = int(candidates.iloc[best].k)
    h = AgglomerativeClustering(n_clusters=k, linkage=cfg["hierarchical_linkage"]).fit(X)
    return {"k": k, "kmeans": km, "ward": h, "candidates": candidates,
            "km_labels": canonical_labels(km.labels_, d), "ward_labels": canonical_labels(h.labels_, d)}


def clustering(s: dict) -> None:
    d, r = s["retained"], s["rep"]
    s["fit"] = f = fit_clusters(d, r)
    table(f["candidates"], "14_k_selection", index=False)
    one = KMeans(n_clusters=1, random_state=r["config"]["random_state"], n_init=r["config"]["n_init"]).fit(r["X"])
    inertia = pd.concat([pd.DataFrame([{"k": 1, "inertia": one.inertia_}]), f["candidates"][["k", "inertia"]]], ignore_index=True)
    inertia["drop_from_previous"] = -inertia.inertia.diff()
    inertia["drop_percent_previous"] = -inertia.inertia.pct_change() * 100
    table(inertia, "14_inertia_including_k1", index=False)
    s["inertia"] = inertia
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(inertia.k, inertia.inertia, "o-", color="#237c86")
    axes[1].plot(f["candidates"].k, f["candidates"].silhouette, "o-", color="#b94d45")
    axes[0].set(xlabel="Number of clusters", ylabel="Within-cluster sum of squares", title="Elbow diagnostic")
    axes[1].set(xlabel="Number of clusters", ylabel="Mean silhouette", title=f"Selected k = {f['k']}")
    savefig("10_cluster_count_selection")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, labels, title in zip(axes, [f["km_labels"], f["ward_labels"]], ["K-means", "Ward hierarchical"]):
        for c in sorted(set(labels)):
            take = labels == c
            ax.scatter(r["X"][take, 0], r["X"][take, 1], s=27, color=CLUSTER_COLORS[c - 1], alpha=.8, label=f"Group {c} (n={take.sum()})")
        ev = r["pca"].explained_variance_ratio_
        ax.set(xlabel=f"PC1 ({ev[0]:.1%} of variance)", ylabel=f"PC2 ({ev[1]:.1%} of variance)", title=title)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"Display uses two PCs; both methods use all {r['n_components']} retained PCs", fontsize=12)
    savefig("11_pca_cluster_comparison")
    Z = linkage(r["X"], method=r["config"]["hierarchical_linkage"], metric="euclidean")
    fig, ax = plt.subplots(figsize=(12, 5))
    dendrogram(Z, truncate_mode="lastp", p=18, show_leaf_counts=True, ax=ax,
               leaf_label_func=lambda i: d.iloc[i]["Country Code"] if i < len(d) else f"({int(Z[i-len(d), 3])})",
               color_threshold=float((Z[-f["k"], 2] + Z[-f["k"] + 1, 2]) / 2) if f["k"] > 1 else 0)
    # For k=2 the cut lies between the final and penultimate merge heights.
    cut = float((Z[-f["k"], 2] + Z[-f["k"] + 1, 2]) / 2)
    ax.axhline(cut, color="#b94d45", linestyle="--", label=f"k={f['k']} cut: height {cut:.2f}")
    ax.set(xlabel="Collapsed branches: (economy count); individual leaves: country code", ylabel="Ward linkage height", title="Hierarchical structure: truncated overview")
    ax.legend(frameon=False); savefig("12_ward_dendrogram_overview")
    fig, ax = plt.subplots(figsize=(24, 8))
    dendrogram(Z, labels=d["Country Name"].tolist(), leaf_rotation=90, leaf_font_size=5, ax=ax, color_threshold=cut)
    ax.axhline(cut, color="#b94d45", linestyle="--")
    ax.set(ylabel="Ward linkage height", title="Full country dendrogram; see overview for readable summary")
    savefig("13_ward_dendrogram_full")
    s["linkage"] = Z
    s["ward_cut"] = cut
    print(f"Both methods use the same {r['n_components']}-PC coordinates and k={f['k']} to compare algorithms without changing inputs; Ward cut height={cut:.4f}.")
    print(f"K-means selected k={f['k']} over {r['config']['k_values']} by highest silhouette; inspect sizes and elbow as context.")
    print("Inertia including k=1:\n", inertia.round(3).to_string(index=False))
    print("Silhouette runner-up:\n", f["candidates"].sort_values("silhouette", ascending=False).iloc[[1]].round(4).to_string(index=False))
    print(f["candidates"].round(4).to_string(index=False))


def interpretation(s: dict) -> None:
    d, r, f = s["retained"].copy(), s["rep"], s["fit"]
    d["KMeans_Cluster"] = f["km_labels"]
    d["Ward_Cluster"] = f["ward_labels"]
    d["Imputed_Indicators"] = d[FEATURES].isna().sum(axis=1)
    d["Distance_to_Centroid"] = np.linalg.norm(r["X"] - f["kmeans"].cluster_centers_[f["kmeans"].labels_], axis=1)
    d["Within_Cluster_Distance_Percentile"] = d.groupby("KMeans_Cluster").Distance_to_Centroid.rank(pct=True) * 100
    d["Silhouette"] = silhouette_samples(r["X"], f["km_labels"])
    d.to_csv(PROC / "country_cluster_assignments.csv", index=False)
    s["assigned"] = d
    for name, column in [("kmeans", "KMeans_Cluster"), ("ward", "Ward_Cluster")]:
        for stat in ["mean", "median", "count"]:
            table(getattr(d.groupby(column)[FEATURES], stat)(), f"15_{name}_observed_profile_{stat}")
        for target, order in [("IncomeGroup", INCOMES), ("Region", None)]:
            ct = pd.crosstab(d[column], d[target])
            if order is not None: ct = ct.reindex(columns=order, fill_value=0)
            table(ct, f"16_{name}_{target}_counts")
            table(ct.div(ct.sum(axis=1), axis=0) * 100, f"16_{name}_{target}_row_percent")
            table(ct.div(ct.sum(axis=0), axis=1) * 100, f"16_{name}_{target}_column_percent")
    scaled = pd.DataFrame(r["scaled"], columns=FEATURES)
    scaled["Cluster"] = f["km_labels"]
    profile = scaled.groupby("Cluster")[FEATURES].mean()
    table(profile, "17_kmeans_mean_standardized_model_profiles")
    fig, ax = plt.subplots(figsize=(11.5, 3.8))
    sns.heatmap(profile, center=0, cmap="vlag", annot=True, fmt=".2f", xticklabels=LABELS, ax=ax)
    ax.set_title(f"K-means profiles: standardized model units; GDP is {'logged' if r['config']['log_gdp'] else 'raw'}")
    savefig("14_cluster_profiles")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8))
    for ax, col, title in zip(axes, ["KMeans_Cluster", "Ward_Cluster"], ["K-means", "Ward"]):
        ct = pd.crosstab(d[col], d.IncomeGroup).reindex(columns=INCOMES, fill_value=0)
        ct.columns = SHORT_INCOMES
        ct.plot.bar(stacked=True, color=COLORS, ax=ax, rot=0)
        ax.set(title=title, xlabel="Cluster", ylabel="Economies")
        ax.legend(fontsize=7, frameon=False, title="Income metadata", title_fontsize=8)
    savefig("15_clusters_vs_income")
    ct = pd.crosstab(d.KMeans_Cluster, d.Region, normalize="index") * 100
    fig, ax = plt.subplots(figsize=(11, 4.2))
    sns.heatmap(ct, cmap="Blues", annot=True, fmt=".1f", ax=ax)
    ax.set_title("Region composition within each K-means cluster (%)")
    ax.set_xticklabels([textwrap.fill(str(c), width=18) for c in ct.columns], fontsize=8, rotation=0)
    savefig("16_clusters_vs_regions")
    cross = pd.crosstab(d.KMeans_Cluster, d.Ward_Cluster)
    table(cross, "18_algorithm_comparison_counts")
    compare = pd.DataFrame([{"method": name, "k": f["k"], "silhouette": silhouette_score(r["X"], lab),
                             "sizes": "; ".join(f"{c}:{int((lab==c).sum())}" for c in sorted(set(lab)))}
                            for name, lab in [("K-means", f["km_labels"]), ("Ward", f["ward_labels"])]])
    table(compare, "19_algorithm_internal_comparison", index=False)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cross, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Same countries, preprocessing and cluster count")
    savefig("17_algorithm_cross_table")
    table(d.loc[d.KMeans_Cluster != d.Ward_Cluster, ["Country Code", "Country Name"] + HELD_OUT + ["KMeans_Cluster", "Ward_Cluster", "Imputed_Indicators"]],
          "20_algorithm_differences_ordered_by_GDP_labels", index=False)
    # The ordered labels above are descriptive, not an optimal matching of arbitrary IDs.
    top = d.sort_values("Distance_to_Centroid", ascending=False).head(12)
    table(top, "21_unusual_countries_full_observed_values", index=False)
    table(d.loc[d.Silhouette < 0].sort_values("Silhouette"), "22_boundary_countries_negative_silhouette", index=False)
    cases = top[["Country Name", "KMeans_Cluster", "Distance_to_Centroid", "Imputed_Indicators"]].copy()
    drivers = []
    for idx in top.index:
        observed = d.loc[idx, FEATURES].notna().to_numpy()
        center = scaled.loc[scaled.Cluster == d.loc[idx, "KMeans_Cluster"], FEATURES].mean().to_numpy()
        delta = r["scaled"][idx] - center
        order = np.argsort(-np.where(observed, np.abs(delta), -np.inf))[:3]
        drivers.append("; ".join(f"{FEATURES[j]}={d.loc[idx, FEATURES[j]]:.3g} (delta {delta[j]:+.2f} standardized units)" for j in order))
    cases["largest_observed_feature_deviations"] = drivers
    table(cases, "23_unusual_country_observed_drivers", index=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(top["Country Name"].iloc[::-1], top.Distance_to_Centroid.iloc[::-1], color="#237c86")
    ax.set(xlabel=f"Distance to K-means centroid in {r['n_components']}-PC space", title="Unusual profiles: large distance is distinct from ambiguous membership")
    savefig("18_unusual_countries")
    s["method_comparison"] = compare
    s["cross"] = cross
    s["ari"] = float(adjusted_rand_score(f["km_labels"], f["ward_labels"]))
    s["nmi"] = float(normalized_mutual_info_score(f["km_labels"], f["ward_labels"]))
    reference_scores = []
    measurement_coverage = []
    for method, col in [("K-means", "KMeans_Cluster"), ("Ward", "Ward_Cluster")]:
        for target in ["IncomeGroup", "Region"]:
            valid = d[target].notna() & d[target].astype(str).str.strip().ne("")
            reference_scores.append({"algorithm": method, "reference_label": target, "n": int(valid.sum()),
                                     "ARI": adjusted_rand_score(d.loc[valid, col], d.loc[valid, target]),
                                     "NMI": normalized_mutual_info_score(d.loc[valid, col], d.loc[valid, target]),
                                     "indicator_year": s["config"]["year"],
                                     "metadata_snapshot_date": s["provenance"]["snapshot_date"]})
        for group, sub in d.groupby(col):
            for feature in FEATURES:
                observed = int(sub[feature].notna().sum())
                measurement_coverage.append({"algorithm": method, "cluster": int(group), "cluster_size": len(sub),
                    "feature": feature, "observed_count": observed, "imputed_count": len(sub) - observed,
                    "imputed_percent": (len(sub) - observed) / len(sub) * 100,
                    "observed_only_median": sub[feature].median()})
    s["external_agreement"] = pd.DataFrame(reference_scores)
    s["cluster_measurement_coverage"] = pd.DataFrame(measurement_coverage)
    table(s["external_agreement"], "34_external_reference_agreement", index=False)
    table(s["cluster_measurement_coverage"], "33_cluster_observed_imputed_profiles", index=False)
    print(f"External comparison: {s['config']['year']} indicators vs {s['provenance']['snapshot_date']} metadata; agreement, not accuracy.")
    print(s["external_agreement"].round(4).to_string(index=False))
    disagreements = d.loc[d.KMeans_Cluster != d.Ward_Cluster].copy()
    # Inspect one economy from each direction of disagreement, rather than choosing only dramatic extremes.
    chosen = disagreements.groupby(["KMeans_Cluster", "Ward_Cluster"], sort=True).head(1).head(2).copy()
    details = []
    for _, row in chosen.iterrows():
        for feature in FEATURES:
            details.append({"Country Code": row["Country Code"], "Country Name": row["Country Name"],
                            "KMeans_Cluster": row.KMeans_Cluster, "Ward_Cluster": row.Ward_Cluster,
                            "KMeans_silhouette": row.Silhouette, "feature": feature,
                            "observed_value": row[feature], "imputed": bool(pd.isna(row[feature]))})
    s["disagreement_cases"] = pd.DataFrame(details)
    table(s["disagreement_cases"], "35_two_method_disagreement_cases", index=False)
    print("Algorithm disagreement and negative within-model silhouette answer different questions. Case details exported with per-field flags.")
    print("Original-unit cluster medians (observed values only; per-feature counts exported):")
    print(d.groupby("KMeans_Cluster")[FEATURES].median().round(2).to_string())
    print("K-means versus Ward:\n", cross.to_string())
    print(f"ARI={s['ari']:.4f}; NMI={s['nmi']:.4f}. Agreement is descriptive, not classification accuracy.")


def robustness(s: dict) -> None:
    main_cfg = s["config"]
    configs = [
        (f"Main: {'log' if main_cfg['log_gdp'] else 'raw'} GDP, PCA {main_cfg['pca_variance_target']:.0%}", {}, None),
        ("Raw GDP baseline", {"log_gdp": False}, None),
        ("Log GDP, no PCA truncation", {"pca_variance_target": None}, None),
        ("Log GDP, PCA 90%", {"pca_variance_target": .9}, None),
        ("Log GDP, at most 3 missing", {"missing_fraction_max": 3 / len(FEATURES)}, None),
        ("Log GDP, complete cases", {"missing_fraction_max": 0}, None),
        ("Log GDP, PCA 80%, k=3", {}, 3),
        ("GDP omitted diagnostic", {"log_gdp": False}, None),
        ("Source-quality flag excluded", {}, None),
        ("CAF life expectancy masked only", {}, None),
    ]
    results, assignments = [], []
    base = s["assigned"].set_index("Country Code")[["KMeans_Cluster", "Ward_Cluster"]]
    for name, overrides, fixed_k in configs:
        cfg = {**main_cfg, **overrides}
        mask = s["countries"][FEATURES].isna().mean(axis=1) <= cfg["missing_fraction_max"]
        d = s["countries"].loc[mask].reset_index(drop=True).copy()
        if name == "Source-quality flag excluded":
            d = d.loc[~d["Country Code"].isin(s["quality_flags"]["Country Code"])].reset_index(drop=True)
        if name == "CAF life expectancy masked only":
            d.loc[d["Country Code"].eq("CAF"), "Life_Expectancy"] = np.nan
        fs = [x for x in FEATURES if x != "GDP_PC"] if name == "GDP omitted diagnostic" else FEATURES
        r = representation(d, config=cfg, features=fs)
        f = fit_clusters(d, r, fixed_k=fixed_k)
        primary = base.reindex(d["Country Code"])
        ok = primary.notna().all(axis=1).to_numpy()
        km_ari = adjusted_rand_score(primary.loc[ok, "KMeans_Cluster"], f["km_labels"][ok])
        ward_ari = adjusted_rand_score(primary.loc[ok, "Ward_Cluster"], f["ward_labels"][ok])
        results.append({"configuration": name, "n": len(d), "n_common": int(ok.sum()),
                        "dimensions": r["X"].shape[1], "k": f["k"],
                        "silhouette_in_own_space": silhouette_score(r["X"], f["km_labels"]),
                        "KMeans_ARI_vs_main": km_ari, "Ward_ARI_vs_main": ward_ari,
                        "ARI_vs_main_common_countries": km_ari,
                        "ARI_KMeans_vs_Ward": adjusted_rand_score(f["km_labels"], f["ward_labels"]),
                        "imputed_cells": int(d[fs].isna().sum().sum()),
                        "KMeans_sizes": ";".join(str(int((f["km_labels"] == c).sum())) for c in sorted(set(f["km_labels"]))),
                        "Ward_sizes": ";".join(str(int((f["ward_labels"] == c).sum())) for c in sorted(set(f["ward_labels"])))})
        assignments.extend({"configuration": name, "Country Code": code, "KMeans_Cluster": int(k), "Ward_Cluster": int(h)}
                           for code, k, h in zip(d["Country Code"], f["km_labels"], f["ward_labels"]))
    s["robustness"] = pd.DataFrame(results)
    table(s["robustness"], "24_robustness", index=False)
    table(pd.DataFrame(assignments), "25_robustness_assignments", index=False)
    seeds = []
    for seed in range(10):
        cfg = {**main_cfg, "random_state": seed}
        km = KMeans(n_clusters=s["fit"]["k"], n_init=cfg["n_init"], random_state=cfg["random_state"]).fit(s["rep"]["X"])
        seeds.append({"seed": seed, "ARI_vs_main": adjusted_rand_score(s["fit"]["km_labels"], km.labels_),
                      "silhouette": silhouette_score(s["rep"]["X"], km.labels_)})
    s["seeds"] = pd.DataFrame(seeds)
    table(s["seeds"], "26_seed_stability", index=False)
    print("ROBUSTNESS: each algorithm is compared with its own main partition, joined by country code.")
    print(s["robustness"].round(4).to_string(index=False))
    print("Silhouettes from different spaces/samples are not directly ranked; the legacy ARI_vs_main_common_countries column aliases KMeans_ARI_vs_main.")
    complete_case_decomposition(s)


def complete_case_decomposition(s: dict) -> None:
    """A->B isolates sample restriction; B->C additionally refits preprocessing."""
    cc_mask = s["retained"][FEATURES].notna().all(axis=1).to_numpy()
    d = s["retained"].loc[cc_mask].reset_index(drop=True)
    assert len(d) == s["counts"]["complete_cases"]
    fixed_k = s["fit"]["k"]
    # B: same complete-case economies on their original full-sample PC coordinates.
    rb = {**s["rep"], "X": s["rep"]["X"][cc_mask]}
    fb = fit_clusters(d, rb, fixed_k=fixed_k)
    # C: same economies, but re-estimate medians, means, scales and PCA.
    rc = representation(d, config=s["config"])
    fc = fit_clusters(d, rc, fixed_k=fixed_k)
    result, labels = [], d[["Country Code", "Country Name"]].copy()
    for method, key in [("KMeans", "km_labels"), ("Ward", "ward_labels")]:
        A, B, C = s["fit"][key][cc_mask], fb[key], fc[key]
        result.append({"algorithm": method, "n": len(d), "k": fixed_k,
                       "A_to_B_ARI_sample_restriction": adjusted_rand_score(A, B),
                       "B_to_C_ARI_preprocessing_refit": adjusted_rand_score(B, C),
                       "A_to_C_ARI_total": adjusted_rand_score(A, C),
                       "main_PC_dimensions": rb["X"].shape[1], "refitted_PC_dimensions": rc["X"].shape[1]})
        labels[f"{method}_A_main_labels"] = A
        labels[f"{method}_B_refit_original_coordinates"] = B
        labels[f"{method}_C_refit_preprocessing"] = C
    s["complete_case_decomposition"] = pd.DataFrame(result)
    table(s["complete_case_decomposition"], "36_complete_case_decomposition", index=False)
    table(labels, "37_complete_case_ABC_assignments", index=False)
    print("Complete-case decomposition: fixed k, identical 110-country sample in A/B/C; ARI effects are not additive.")
    print(s["complete_case_decomposition"].round(4).to_string(index=False))


def validate_and_save(s: dict) -> dict:
    r, f, d = s["rep"], s["fit"], s["assigned"]
    assert len(d) == len(r["X"]) == len(f["km_labels"]) == len(f["ward_labels"])
    assert np.isfinite(r["X"]).all() and np.isfinite(r["scaled"]).all()
    assert np.allclose(r["scaled"].mean(axis=0), 0, atol=1e-12)
    assert np.allclose(r["scaled"].std(axis=0), 1, atol=1e-12)
    assert not set(r["features"]).intersection(HELD_OUT)
    assert d["Country Code"].is_unique and not d.Region.eq("Aggregates").any()
    ev = r["pca"].explained_variance_ratio_
    threshold = r["config"]["pca_variance_target"]
    if threshold is not None:
        assert ev[:r["n_components"]].sum() >= threshold
        assert ev[:r["n_components"]-1].sum() < threshold
    assert r["X"].shape[1] == r["n_components"]
    assert d["KMeans_Cluster"].nunique() == d["Ward_Cluster"].nunique() == f["k"]
    assert len(s["coverage"]) == s["counts"]["retained"] + s["counts"]["excluded"]
    summary = {"configuration": r["config"], "counts": s["counts"], "features": FEATURES,
               "metadata": s["provenance"]["metadata_status"],
               "indicator_year": s["config"]["year"], "metadata_snapshot_date": s["provenance"]["snapshot_date"],
               "n_components": r["n_components"], "retained_variance": float(ev[:r["n_components"]].sum()),
               "first_two_variance": float(ev[:2].sum()), "selected_k": f["k"],
               "kmeans_sizes": d.KMeans_Cluster.value_counts().sort_index().to_dict(),
               "ward_sizes": d.Ward_Cluster.value_counts().sort_index().to_dict(),
               "kmeans_silhouette": float(silhouette_score(r["X"], f["km_labels"])),
               "ward_silhouette": float(silhouette_score(r["X"], f["ward_labels"])),
               "ari_between_methods": s["ari"], "nmi_between_methods": s["nmi"],
               "ward_cut_height": s["ward_cut"],
               "imputed_values": int(d[FEATURES].isna().sum().sum()),
               "negative_silhouette_n": int((d.Silhouette < 0).sum()),
               "minimum_seed_ARI": float(s["seeds"].ARI_vs_main.min()),
               "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                            "scipy": scipy.__version__, "scikit-learn": sklearn.__version__,
                            "matplotlib": matplotlib.__version__, "seaborn": sns.__version__}}
    (ROOT / "outputs/results.json").write_text(json.dumps(summary, indent=2, default=lambda x: x.item()))
    s["summary"] = summary
    print("All structural checks passed. Results written to outputs/results.json.")
    print(json.dumps(summary, indent=2, default=lambda x: x.item()))
    return summary


def main() -> dict:
    # Small country tables do not benefit from many BLAS/OpenMP threads.
    with threadpool_limits(limits=1):
        state = load_data()
        candidate_years(state)
        eda(state)
        preprocess(state)
        clustering(state)
        interpretation(state)
        robustness(state)
        validate_and_save(state)
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-gdp-baseline", action="store_true", help="Reproduce the archived raw-GDP baseline without replacing the main result.")
    args = parser.parse_args()
    if args.raw_gdp_baseline:
        with threadpool_limits(limits=1):
            state = load_data()
            rep = representation(state["retained"], config={**CONFIG, "log_gdp": False})
            fit = fit_clusters(state["retained"], rep)
            print("Archived raw-GDP baseline:", {"n": len(state["retained"]), "k": fit["k"],
                  "KMeans_sizes": pd.Series(fit["km_labels"]).value_counts().sort_index().to_dict(),
                  "retained_variance": rep["pca"].explained_variance_ratio_[:rep["n_components"]].sum(),
                  "between_method_ARI": adjusted_rand_score(fit["km_labels"], fit["ward_labels"])})
    else:
        main()
