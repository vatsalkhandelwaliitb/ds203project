"""Optional source refresh into a NEW directory. Existing analysis runs offline.

Usage: python fetch_data.py --output new_snapshot
Review revisions before replacing frozen inputs and their provenance manifest.
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

from analysis import INDICATORS


def fetch(url):
    req = Request(url, headers={"User-Agent": "DS203-academic-country-clustering/1.0"})
    with urlopen(req, timeout=90) as response:
        raw = response.read()
    parsed = json.loads(raw)
    if not isinstance(parsed, list) or len(parsed) != 2 or not isinstance(parsed[1], list):
        raise ValueError(f"Unexpected API response from {url}")
    if int(parsed[0]["pages"]) != 1 or len(parsed[1]) != int(parsed[0]["total"]):
        raise ValueError("Incomplete response: pagination must be handled before use.")
    return raw, parsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dest = args.output.resolve()
    dest.mkdir(parents=True, exist_ok=False)
    wdi_url = "https://api.worldbank.org/v2/country/all/indicator/" + ";".join(INDICATORS.values()) + "?source=2&date=2022&format=json&per_page=20000"
    meta_url = "https://api.worldbank.org/v2/country?format=json&per_page=400"
    raw, data = fetch(wdi_url)
    meta_raw, meta = fetch(meta_url)
    candidate_url = wdi_url.replace("date=2022&", "date=2021:2024&")
    candidate_raw, candidate = fetch(candidate_url)
    if candidate[0].get("lastupdated") != data[0].get("lastupdated"):
        raise ValueError("Main and candidate snapshots span different WDI releases; retrieve a consistent snapshot.")
    known_codes = {c["id"] for c in meta[1]}
    unmatched = sorted({x["countryiso3code"] for x in data[1] if x["countryiso3code"] and x["countryiso3code"] not in known_codes})
    if unmatched:
        (dest / "unmatched_nonblank_country_codes.json").write_text(json.dumps(unmatched, indent=2))
        raise ValueError(f"Unmatched nonblank country codes require review: {unmatched}")
    (dest / "wdi_2022_api.json").write_bytes(raw)
    (dest / "country_metadata.json").write_bytes(meta_raw)
    (dest / "wdi_candidate_years_2021_2024.json").write_bytes(candidate_raw)
    with (dest / "wdi_2022.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Country Name", "Country Code", "Indicator Code", "2022"])
        writer.writeheader()
        for x in data[1]:
            writer.writerow({"Country Name": x["country"]["value"], "Country Code": x["countryiso3code"],
                             "Indicator Code": x["indicator"]["id"], "2022": x["value"]})
    retrieved = dt.datetime.now(dt.timezone.utc).date().isoformat()
    manifest = {"indicator_year": 2022, "snapshot_date": retrieved,
                "wdi_lastupdated": data[0].get("lastupdated"), "wdi_url": wdi_url,
                "metadata_url": meta_url, "metadata_status": f"Retrieved from official Country API on {retrieved}",
                "candidate_years_url": candidate_url, "candidate_snapshot_lastupdated": candidate[0].get("lastupdated"),
                "label_interpretation": "Metadata snapshot labels, not reconstructed 2022 classifications; held out of model inputs.",
                "files": {}}
    for p in sorted(dest.iterdir()):
        manifest["files"][p.name] = {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
    (dest / "provenance.json").write_text(json.dumps(manifest, indent=2))
    print("New snapshot saved:", dest)
    print("Compare coverage, revisions and metadata dates before adopting this snapshot.")


if __name__ == "__main__":
    main()
