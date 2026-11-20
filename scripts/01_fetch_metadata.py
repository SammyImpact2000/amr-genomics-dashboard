#!/usr/bin/env python3
"""Fetch laboratory-tested AMR metadata from the public BV-BRC REST API.

Queries https://www.bv-brc.org/api/genome_amr/ for NCBI Taxon ID 573
(Klebsiella pneumoniae) records with laboratory-tested Resistant or Susceptible
phenotypes across six target antibiotics and saves the raw records to CSV.
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests

API_BASE_URL = "https://www.bv-brc.org/api/genome_amr/"
TAXON_ID = 573
EVIDENCE_VALUE = "Laboratory Method"
VALID_PHENOTYPES = ("Resistant", "Susceptible")

# Canonical antibiotic names requested for the classification dashboard.
TARGET_ANTIBIOTICS = (
    "meropenem",
    "ceftriaxone",
    "ciprofloxacin",
    "gentamicin",
    "ampicillin",
    "trimethoprim-sulfamethoxazole",
)

# BV-BRC stores trimethoprim-sulfamethoxazole with a forward slash.
# Query both spellings so no matching laboratory records are missed.
API_ANTIBIOTIC_ALIASES = (
    "meropenem",
    "ceftriaxone",
    "ciprofloxacin",
    "gentamicin",
    "ampicillin",
    "trimethoprim-sulfamethoxazole",
    "trimethoprim/sulfamethoxazole",
)

SELECTED_FIELDS = (
    "id",
    "genome_id",
    "genome_name",
    "taxon_id",
    "antibiotic",
    "resistant_phenotype",
    "evidence",
    "laboratory_typing_method",
    "laboratory_typing_platform",
    "testing_standard",
    "testing_standard_year",
    "measurement",
    "measurement_sign",
    "measurement_value",
    "measurement_unit",
    "pmid",
    "date_inserted",
    "date_modified",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "metadata" / "bv_brc_amr_taxon573.csv"

CONTENT_RANGE_RE = re.compile(r"items\s+(\d+)-(\d+)/(\d+)")


def build_rql_query(limit: int, offset: int) -> str:
    """Build an RQL query string for the BV-BRC genome_amr endpoint."""
    encoded_evidence = quote(EVIDENCE_VALUE, safe="")
    encoded_antibiotics = ",".join(quote(ab, safe="") for ab in API_ANTIBIOTIC_ALIASES)
    encoded_phenotypes = ",".join(quote(p, safe="") for p in VALID_PHENOTYPES)
    select_fields = ",".join(SELECTED_FIELDS)

    return (
        f"eq(taxon_id,{TAXON_ID})"
        f"&eq(evidence,{encoded_evidence})"
        f"&in(resistant_phenotype,({encoded_phenotypes}))"
        f"&in(antibiotic,({encoded_antibiotics}))"
        f"&select({select_fields})"
        f"&limit({limit},{offset})"
    )


def parse_total_from_content_range(header_value: str | None) -> int | None:
    """Extract total record count from BV-BRC's Content-Range header."""
    if not header_value:
        return None
    match = CONTENT_RANGE_RE.search(header_value)
    return int(match.group(3)) if match else None


def fetch_page(
    session: requests.Session,
    limit: int,
    offset: int,
    timeout: int = 60,
    max_retries: int = 3,
) -> tuple[list[dict[str, Any]], int | None]:
    """Fetch a single paginated batch from the BV-BRC API with retries."""
    url = f"{API_BASE_URL}?{build_rql_query(limit=limit, offset=offset)}"
    headers = {"Accept": "application/json"}

    for attempt in range(1, max_retries + 1):
        try:
            response = session.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            records = response.json()
            total = parse_total_from_content_range(response.headers.get("Content-Range"))
            return records, total
        except (requests.RequestException, ValueError) as exc:
            if attempt == max_retries:
                raise RuntimeError(
                    f"Failed to fetch BV-BRC page at offset={offset} after {max_retries} attempts: {exc}"
                ) from exc
            backoff_seconds = 2 ** (attempt - 1)
            print(
                f"Request error at offset={offset} (attempt {attempt}/{max_retries}): {exc}. "
                f"Retrying in {backoff_seconds}s..."
            )
            time.sleep(backoff_seconds)

    return [], None


def fetch_all_records(
    page_size: int = 5000,
    max_records: int | None = None,
    timeout: int = 60,
) -> pd.DataFrame:
    """Paginate through all matching BV-BRC AMR records and return a DataFrame."""
    all_records: list[dict[str, Any]] = []
    offset = 0
    total_expected: int | None = None

    with requests.Session() as session:
        while True:
            current_limit = page_size
            if max_records is not None:
                remaining = max_records - len(all_records)
                if remaining <= 0:
                    break
                current_limit = min(page_size, remaining)

            batch, total = fetch_page(
                session=session,
                limit=current_limit,
                offset=offset,
                timeout=timeout,
            )

            if total is not None and total_expected is None:
                total_expected = total
                print(f"BV-BRC reports {total_expected:,} matching laboratory AMR records.")

            if not batch:
                break

            all_records.extend(batch)
            offset += len(batch)

            progress = f"{len(all_records):,}"
            if total_expected is not None:
                progress += f" / {total_expected:,}"
            print(f"Fetched {progress} records...")

            if len(batch) < current_limit:
                break
            if total_expected is not None and offset >= total_expected:
                break

    if not all_records:
        return pd.DataFrame(columns=list(SELECTED_FIELDS))

    df = pd.DataFrame(all_records)

    # Flatten list fields (e.g., pmid) into semicolon-delimited strings for clean CSV output.
    if "pmid" in df.columns:
        df["pmid"] = df["pmid"].apply(
            lambda val: ";".join(str(x) for x in val) if isinstance(val, list) else val
        )

    # Ensure consistent column ordering even if optional fields are absent in a batch.
    ordered_cols = [col for col in SELECTED_FIELDS if col in df.columns]
    extra_cols = [col for col in df.columns if col not in ordered_cols]
    return df[ordered_cols + extra_cols]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch BV-BRC laboratory AMR metadata for NCBI Taxon ID 573."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output CSV file path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=5000,
        help="Number of records to request per API page (default: 5000).",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional cap on total records to fetch (useful for quick smoke tests).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="HTTP request timeout in seconds (default: 60).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Querying BV-BRC genome_amr API for taxon_id={TAXON_ID} ({EVIDENCE_VALUE})...")
    df = fetch_all_records(
        page_size=args.page_size,
        max_records=args.max_records,
        timeout=args.timeout,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print(f"\nSaved {len(df):,} raw records to: {args.output}")
    if not df.empty:
        print(f"Unique genomes: {df['genome_id'].nunique():,}")
        print("\nRaw record counts by antibiotic:")
        print(df["antibiotic"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
