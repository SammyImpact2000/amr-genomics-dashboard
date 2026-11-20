#!/usr/bin/env python3
"""Fetch geographic and temporal surveillance metadata from BV-BRC /api/genome/.

Queries isolation_country, collection_year, collection_date, host_name, and
isolation_source for all unique genome_ids in data/processed/amr_label_matrix.csv
and saves the cleaned table to data/processed/genome_surveillance_metadata.csv.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests

API_BASE_URL = "https://www.bv-brc.org/api/genome/"
SELECTED_FIELDS = (
    "genome_id",
    "genome_name",
    "isolation_country",
    "geographic_group",
    "collection_year",
    "collection_date",
    "host_name",
    "isolation_source",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS_PATH = PROJECT_ROOT / "data" / "processed" / "amr_label_matrix.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "genome_surveillance_metadata.csv"


def fetch_genome_batch(
    session: requests.Session,
    genome_ids: list[str],
    timeout: int = 60,
    max_retries: int = 5,
) -> list[dict[str, Any]]:
    encoded_ids = ",".join(quote(gid, safe="") for gid in genome_ids)
    select_fields = ",".join(SELECTED_FIELDS)
    url = (
        f"{API_BASE_URL}?in(genome_id,({encoded_ids}))"
        f"&select({select_fields})"
        f"&limit({len(genome_ids) + 50},0)"
    )
    headers = {"Accept": "application/json"}

    for attempt in range(1, max_retries + 1):
        try:
            resp = session.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            if attempt == max_retries:
                raise RuntimeError(
                    f"Failed to fetch genome metadata batch (size={len(genome_ids)}): {exc}"
                ) from exc
            time.sleep(2 ** (attempt - 1))
    return []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch BV-BRC genome surveillance metadata (country, year, host)."
    )
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--batch-size", type=int, default=150)
    args = parser.parse_args()

    labels = pd.read_csv(args.labels, usecols=["genome_id"], dtype={"genome_id": str})
    genome_ids = sorted(labels["genome_id"].dropna().unique().tolist())

    all_records: list[dict[str, Any]] = []
    n_batches = (len(genome_ids) + args.batch_size - 1) // args.batch_size
    print(f"Fetching /api/genome/ metadata for {len(genome_ids):,} genomes in {n_batches} batches...")

    with requests.Session() as session:
        for i in range(n_batches):
            batch_ids = genome_ids[i * args.batch_size : (i + 1) * args.batch_size]
            records = fetch_genome_batch(session, batch_ids)
            all_records.extend(records)
            if (i + 1) % 10 == 0 or (i + 1) == n_batches:
                print(f"  Batch {i + 1}/{n_batches} -> {len(all_records):,} genomes retrieved")

    df = pd.DataFrame(all_records)
    for col in SELECTED_FIELDS:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[list(SELECTED_FIELDS)].drop_duplicates(subset=["genome_id"]).copy()
    df["genome_id"] = df["genome_id"].astype(str).str.strip()
    df["collection_year"] = pd.to_numeric(df["collection_year"], errors="coerce").astype("Int64")

    # Normalize common country name variants
    if "isolation_country" in df.columns:
        df["isolation_country"] = (
            df["isolation_country"]
            .astype("string")
            .str.strip()
            .replace({"": pd.NA, "NA": pd.NA, "Not applicable": pd.NA, "missing": pd.NA})
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print(f"\nSaved {len(df):,} genome surveillance records to: {args.output}")
    print("\nTop 10 Isolation Countries:")
    print(df["isolation_country"].value_counts().head(10).to_string())
    valid_years = df["collection_year"].dropna()
    if not valid_years.empty:
        print(
            f"\nCollection Year Range: {int(valid_years.min())} - {int(valid_years.max())} "
            f"(non-null: {len(valid_years):,} / {len(df):,})"
        )


if __name__ == "__main__":
    main()
