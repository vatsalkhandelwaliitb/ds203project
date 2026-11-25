"""Meaningful offline checks of frozen inputs, results and absence of label leakage."""
import json
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import analysis as project


def assert_assignments_match(expected, exported):
    """Compare country-label associations; row order and global ARI cannot verify identity."""
    key = "Country Code"
    cols = ["KMeans_Cluster", "Ward_Cluster"]
    assert expected[key].is_unique and exported[key].is_unique, "Duplicate country codes"
    assert set(expected[key]) == set(exported[key]), "Country-code roster mismatch"
    joined = expected[[key] + cols].merge(exported[[key] + cols], on=key, validate="one_to_one", suffixes=("_expected", "_exported"))
    for col in cols:
        bad = joined[f"{col}_expected"] != joined[f"{col}_exported"]
        assert not bad.any(), f"Country-code assignment mismatch for {col}: {joined.loc[bad, key].tolist()}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignments", type=Path, default=ROOT / "data/processed/country_cluster_assignments.csv")
    parser.add_argument("--skip-mutation", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        state = project.load_data()
        d = state["retained"]
        assert state["counts"]["matched_economies"] == 217
        assert len(d) == 205
        assert d[project.FEATURES].isna().sum().sum() == 197
        assert not set(project.FEATURES).intersection(project.HELD_OUT)
        assert state["counts"]["coded_aggregates"] == 44
        assert state["counts"]["uncoded_income_aggregates"] == 5
        a = project.representation(d)
        fake = d.copy()
        for col in project.HELD_OUT:
            fake[col] = "deliberately changed reference labels"
        b = project.representation(fake)
        assert np.allclose(a["X"], b["X"]), "Reference-label leakage into preprocessing."
        assert np.allclose(a["imputer"].statistics_, b["imputer"].statistics_)
        fit = project.fit_clusters(d, a)
        changed_fit = project.fit_clusters(fake, b)
        assert adjusted_rand_score(fit["km_labels"], changed_fit["km_labels"]) == 1
        assert adjusted_rand_score(fit["ward_labels"], changed_fit["ward_labels"]) == 1
        assert np.isfinite(a["X"]).all()
        assert np.allclose(a["scaled"].mean(0), 0, atol=1e-12)
        assert np.allclose(a["scaled"].std(0), 1, atol=1e-12)
        assert a["n_components"] == 6
        assert np.isclose(a["pca"].explained_variance_ratio_[:6].sum(), .8466687625015747)
        assert np.allclose(a["X"], a["pca"].transform(a["scaled"])[:, :6])
        assert sorted(np.bincount(fit["km_labels"])[1:].tolist()) == [60, 145]
        assert fit["k"] == 2
        assert np.isclose(adjusted_rand_score(fit["km_labels"], fit["ward_labels"]), .5712252539920316)
        # The original raw-GDP baseline remains reproducible after the extension.
        raw = project.representation(d, log_gdp=False)
        baseline = project.fit_clusters(d, raw)
        assert sorted(np.bincount(baseline["km_labels"])[1:].tolist()) == [61, 144]
        assert np.isclose(adjusted_rand_score(baseline["km_labels"], baseline["ward_labels"]), .08009046974182775)
        summary = json.loads((ROOT / "outputs/results.json").read_text())
        assignments = pd.read_csv(args.assignments)
        assert assignments["Country Code"].is_unique
        assert len(assignments) == summary["counts"]["retained"]
        expected = pd.DataFrame({"Country Code": d["Country Code"].to_numpy(),
                                 "KMeans_Cluster": fit["km_labels"], "Ward_Cluster": fit["ward_labels"]})
        assert_assignments_match(expected, assignments)
        assert_assignments_match(expected, assignments.sample(frac=1, random_state=8))
        benchmark = pd.read_csv(ROOT / "reference/benchmark_v1/country_cluster_assignments.csv")
        assert_assignments_match(benchmark, assignments)
        # Run this same test script on a real temporary corrupted export.
        if not args.skip_mutation:
            bad = assignments.copy()
            first = bad.index[0]
            other = bad.index[bad.KMeans_Cluster.ne(bad.loc[first, "KMeans_Cluster"])][0]
            bad.loc[[first, other], "Country Code"] = bad.loc[[other, first], "Country Code"].to_numpy()
            with tempfile.TemporaryDirectory(prefix="ds203-assignment-mutation-") as folder:
                path = Path(folder) / "corrupted_assignments.csv"
                bad.to_csv(path, index=False)
                run = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--assignments", str(path), "--skip-mutation"], capture_output=True, text=True)
                assert run.returncode != 0 and "Country-code assignment mismatch" in run.stderr, "Mutation test failed to detect swapped country codes"
            print("PASS: swapping two exported country codes in a temporary copy makes this test fail; row shuffling alone passes.")
        # All requested analyses have actual exported evidence.
        expected = ["02_required_distributions", "04_income_group_comparisons", "05_observed_correlations",
                    "06_required_pairwise_relationships", "07_pca_variance", "08_pca_coefficients",
                    "10_cluster_count_selection", "11_pca_cluster_comparison", "12_ward_dendrogram_overview",
                    "14_cluster_profiles", "15_clusters_vs_income", "16_clusters_vs_regions",
                    "17_algorithm_cross_table", "18_unusual_countries"]
        for name in expected:
            for extension in ["png", "pdf"]:
                p = ROOT / "outputs/figures" / f"{name}.{extension}"
                assert p.is_file() and p.stat().st_size > 1000, p
        for name in ["06_observed_summary_statistics", "09_pairwise_observation_counts",
                     "21_unusual_countries_full_observed_values", "24_robustness", "27_source_quality_flags",
                     "28_entity_reconciliation", "30_candidate_year_coverage", "33_cluster_observed_imputed_profiles",
                     "34_external_reference_agreement", "35_two_method_disagreement_cases", "36_complete_case_decomposition"]:
            assert (ROOT / "outputs/tables" / f"{name}.csv").is_file()
        years = pd.read_csv(ROOT / "outputs/tables/30_candidate_year_coverage.csv")
        assert years.loc[years.selected, "year"].tolist() == [project.CONFIG["year"]]
        assert years.loc[years.year.eq(2022), "complete_cases"].iloc[0] == 110
        agreement = pd.read_csv(ROOT / "outputs/tables/34_external_reference_agreement.csv")
        assert len(agreement) == 4 and agreement.n.eq(205).all()
        coverage = pd.read_csv(ROOT / "outputs/tables/33_cluster_observed_imputed_profiles.csv")
        for _, sub in coverage.groupby("algorithm"):
            assert sub.loc[sub.feature.eq("Secondary_Enrollment"), "imputed_count"].sum() == 68
        robust = pd.read_csv(ROOT / "outputs/tables/24_robustness.csv")
        raw_row = robust.loc[robust.configuration.eq("Raw GDP baseline")].iloc[0]
        assert np.isclose(raw_row.Ward_ARI_vs_main, .072536069590489, atol=1e-5)
        assert np.isclose(raw_row.KMeans_ARI_vs_main, .9799498789276595)
        masked = robust.loc[robust.configuration.eq("CAF life expectancy masked only")].iloc[0]
        assert masked.n == 205 and masked.imputed_cells == 198
        cfg_override = {**project.CONFIG, "random_state": 7, "n_init": 3, "pca_variance_target": .9}
        rr = project.representation(d, config=cfg_override)
        ff = project.fit_clusters(d, rr)
        assert ff["kmeans"].random_state == 7 and ff["kmeans"].n_init == 3
        assert ff["ward"].linkage == cfg_override["hierarchical_linkage"]
        assert rr["pca"].explained_variance_ratio_[:rr["n_components"]].sum() >= cfg_override["pca_variance_target"]
        print("PASS: provenance hashes, exact frozen-data counts, no label leakage, scaling/PCA,")
        print("main and original-baseline results, saved assignments, and required exported analyses.")


if __name__ == "__main__":
    main()
