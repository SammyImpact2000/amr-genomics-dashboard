#!/usr/bin/env python3
"""Fetch BV-BRC Specialty Genes (Antibiotic Resistance) for a balanced MVP cohort.

Selects a balanced 500-genome cohort from data/processed/amr_label_matrix.csv
(genomes with laboratory susceptibility labels for at least 4 of the 5 core
antibiotics), queries the public BV-BRC Specialty Genes API in batches for
property="Antibiotic Resistance" across curated databases (CARD, NDARO, PATRIC),
and pivots the detected genes into a binary feature matrix.
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

API_BASE_URL = "https://www.bv-brc.org/api/sp_gene/"
PROPERTY_VALUE = '"Antibiotic Resistance"'
CURATED_SOURCES = ("CARD", "NDARO", "PATRIC")

CORE_ANTIBIOTICS = [
    "meropenem",
    "ceftriaxone",
    "ciprofloxacin",
    "gentamicin",
    "trimethoprim-sulfamethoxazole",
]

SELECTED_FIELDS = (
    "genome_id",
    "gene",
    "function",
    "product",
    "source",
    "property",
    "evidence",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABEL_PATH = PROJECT_ROOT / "data" / "processed" / "amr_label_matrix.csv"
DEFAULT_RAW_OUT = PROJECT_ROOT / "data" / "raw" / "metadata" / "bv_brc_sp_genes_mvp500.csv"
DEFAULT_FEATURE_OUT = PROJECT_ROOT / "data" / "processed" / "gene_feature_matrix.csv"

CONTENT_RANGE_RE = re.compile(r"items\s+(\d+)-(\d+)/(\d+)")


def select_balanced_cohort(
    label_df: pd.DataFrame,
    cohort_size: int = 500,
    min_drugs: int = 4,
    random_state: int = 42,
) -> pd.DataFrame:
    """Select a balanced cohort of genomes tested on at least `min_drugs` core drugs.

    Stratifies across resistance burden (number of resistant core drugs, 0..5)
    so pan-susceptible, partially resistant, and multidrug-resistant isolates
    are all well represented across the 5 core antibiotics.
    """
    core_df = label_df[["genome_id", *CORE_ANTIBIOTICS]].copy()
    tested_count = core_df[CORE_ANTIBIOTICS].notna().sum(axis=1)
    eligible = core_df.loc[tested_count >= min_drugs].copy()

    if len(eligible) <= cohort_size:
        return eligible.sort_values("genome_id").reset_index(drop=True)

    # Stratify by number of resistant core drugs (0 through 5) for balanced
    # representation of susceptible and resistant phenotypes across all 5 drugs.
    eligible["_n_resistant"] = eligible[CORE_ANTIBIOTICS].fillna(0).sum(axis=1).astype(int)
    rng = np.random.default_rng(random_state)

    strata = sorted(eligible["_n_resistant"].unique())
    target_per_stratum = cohort_size // len(strata)

    selected_parts: list[pd.DataFrame] = []
    remaining_parts: list[pd.DataFrame] = []

    for stratum in strata:
        group = eligible.loc[eligible["_n_resistant"] == stratum]
        # Prefer complete 5-drug panels first when equally eligible, randomized within tier
        group = group.assign(
            _n_tested=group[CORE_ANTIBIOTICS].notna().sum(axis=1),
            _rand=rng.random(len(group)),
        ).sort_values(["_n_tested", "_rand"], ascending=[False, True])

        take = min(len(group), target_per_stratum)
        selected_parts.append(group.iloc[:take])
        if len(group) > take:
            remaining_parts.append(group.iloc[take:])

    selected = pd.concat(selected_parts, ignore_index=True)

    # Top up any remaining slots evenly from leftover strata
    shortfall = cohort_size - len(selected)
    if shortfall > 0 and remaining_parts:
        leftovers = pd.concat(remaining_parts, ignore_index=True)
        leftovers = leftovers.sort_values(["_n_tested", "_rand"], ascending=[False, True])
        selected = pd.concat([selected, leftovers.iloc[:shortfall]], ignore_index=True)

    selected = (
        selected.drop(columns=["_n_resistant", "_n_tested", "_rand"])
        .sort_values("genome_id")
        .reset_index(drop=True)
    )
    return selected


def build_sp_gene_query(genome_ids: list[str], limit: int, offset: int) -> str:
    """Build an RQL query string for the BV-BRC sp_gene endpoint."""
    encoded_genomes = ",".join(quote(gid, safe="") for gid in genome_ids)
    encoded_property = quote(PROPERTY_VALUE, safe="")
    encoded_sources = ",".join(quote(src, safe="") for src in CURATED_SOURCES)
    select_fields = ",".join(SELECTED_FIELDS)

    return (
        f"in(genome_id,({encoded_genomes}))"
        f"&eq(property,{encoded_property})"
        f"&in(source,({encoded_sources}))"
        f"&select({select_fields})"
        f"&limit({limit},{offset})"
    )


def parse_total(header_value: str | None) -> int | None:
    if not header_value:
        return None
    match = CONTENT_RANGE_RE.search(header_value)
    return int(match.group(3)) if match else None


def fetch_batch(
    session: requests.Session,
    genome_ids: list[str],
    page_size: int = 25000,
    timeout: int = 90,
    max_retries: int = 5,
) -> list[dict[str, Any]]:
    """Fetch all Antibiotic Resistance specialty genes for a batch of genome_ids."""
    records: list[dict[str, Any]] = []
    offset = 0
    total_expected: int | None = None
    headers = {"Accept": "application/json"}

    while True:
        url = f"{API_BASE_URL}?{build_sp_gene_query(genome_ids, limit=page_size, offset=offset)}"
        batch: list[dict[str, Any]] = []
        total: int | None = None

        for attempt in range(1, max_retries + 1):
            try:
                response = session.get(url, headers=headers, timeout=timeout)
                response.raise_for_status()
                batch = response.json()
                total = parse_total(response.headers.get("Content-Range"))
                break
            except (requests.RequestException, ValueError) as exc:
                if attempt == max_retries:
                    raise RuntimeError(
                        f"Failed to fetch sp_gene batch (size={len(genome_ids)}, offset={offset}) "
                        f"after {max_retries} attempts: {exc}"
                    ) from exc
                backoff = 2 ** (attempt - 1)
                print(
                    f"  Retry {attempt}/{max_retries} after error: {exc} "
                    f"(sleeping {backoff}s)..."
                )
                time.sleep(backoff)

        if total is not None and total_expected is None:
            total_expected = total

        if not batch:
            break

        records.extend(batch)
        offset += len(batch)

        if len(batch) < page_size:
            break
        if total_expected is not None and offset >= total_expected:
            break

    return records


def extract_gene_name(row: pd.Series) -> str:
    """Resolve a clean gene feature name from `gene` (NDARO/PATRIC) or `function` (CARD)."""
    gene_val = row.get("gene")
    if pd.notna(gene_val):
        cleaned = str(gene_val).strip()
        if cleaned:
            return cleaned

    func_val = row.get("function")
    if pd.notna(func_val):
        cleaned = str(func_val).strip()
        if cleaned:
            return cleaned

    return ""


def build_feature_matrix(
    raw_sp_df: pd.DataFrame,
    cohort_genome_ids: list[str],
) -> pd.DataFrame:
    """Pivot raw specialty gene annotations into a binary genome_id x gene matrix."""
    if raw_sp_df.empty:
        return pd.DataFrame(index=pd.Index(cohort_genome_ids, name="genome_id"))

    work = raw_sp_df.copy()
    work["genome_id"] = work["genome_id"].astype(str).str.strip()
    work["gene_feature"] = work.apply(extract_gene_name, axis=1)
    work = work[work["gene_feature"].ne("")].copy()

    # Deduplicate (genome_id, gene_feature) pairs -> binary presence (1)
    pairs = work[["genome_id", "gene_feature"]].drop_duplicates()
    pairs["present"] = 1

    matrix = pairs.pivot(index="genome_id", columns="gene_feature", values="present")
    matrix = matrix.reindex(index=cohort_genome_ids).fillna(0).astype(int)
    matrix.index.name = "genome_id"
    matrix = matrix.reindex(sorted(matrix.columns), axis=1)
    return matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch BV-BRC Specialty Genes (Antibiotic Resistance) for an MVP cohort."
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=DEFAULT_LABEL_PATH,
        help=f"Cleaned binary label matrix CSV (default: {DEFAULT_LABEL_PATH})",
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=DEFAULT_RAW_OUT,
        help=f"Raw specialty gene records CSV (default: {DEFAULT_RAW_OUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_FEATURE_OUT,
        help=f"Output binary gene feature matrix CSV (default: {DEFAULT_FEATURE_OUT})",
    )
    parser.add_argument(
        "--cohort-size",
        type=int,
        default=500,
        help="Number of unique genomes to select for the MVP cohort (default: 500).",
    )
    parser.add_argument(
        "--min-drugs",
        type=int,
        default=4,
        help="Minimum number of labeled core antibiotics per genome (default: 4).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=25,
        help="Number of genome_ids per BV-BRC sp_gene API request (default: 25).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible cohort selection (default: 42).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.labels.exists():
        raise FileNotFoundError(
            f"Label matrix not found at {args.labels}. Run scripts/02_clean_labels.py first."
        )

    label_df = pd.read_csv(args.labels, dtype={"genome_id": str})
    cohort_df = select_balanced_cohort(
        label_df,
        cohort_size=args.cohort_size,
        min_drugs=args.min_drugs,
        random_state=args.seed,
    )
    cohort_ids = cohort_df["genome_id"].tolist()

    print(
        f"Selected balanced MVP cohort of {len(cohort_ids):,} genomes "
        f"(>= {args.min_drugs} of 5 core drugs labeled)."
    )
    print("\nMVP Cohort Label Distribution:")
    for drug in CORE_ANTIBIOTICS:
        s = cohort_df[drug]
        r_cnt = int((s == 1).sum())
        s_cnt = int((s == 0).sum())
        tot = r_cnt + s_cnt
        pct_r = 100.0 * r_cnt / tot if tot else 0.0
        print(f"  {drug:30s}: tested={tot:3d} | R={r_cnt:3d} ({pct_r:5.1f}%) | S={s_cnt:3d}")

    print(
        f"\nQuerying BV-BRC sp_gene API in batches of {args.batch_size} "
        f"(property={PROPERTY_VALUE}, sources={CURATED_SOURCES})..."
    )

    all_records: list[dict[str, Any]] = []
    n_batches = (len(cohort_ids) + args.batch_size - 1) // args.batch_size

    with requests.Session() as session:
        for batch_idx in range(n_batches):
            start = batch_idx * args.batch_size
            end = min(start + args.batch_size, len(cohort_ids))
            batch_ids = cohort_ids[start:end]
            batch_records = fetch_batch(session=session, genome_ids=batch_ids)
            all_records.extend(batch_records)
            print(
                f"  Batch {batch_idx + 1:2d}/{n_batches}: "
                f"genomes {start + 1:3d}-{end:3d} -> {len(batch_records):,} hits "
                f"(total: {len(all_records):,})"
            )

    raw_sp_df = pd.DataFrame(all_records)
    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    raw_sp_df.to_csv(args.raw_output, index=False)

    feature_matrix = build_feature_matrix(raw_sp_df, cohort_ids)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    feature_matrix.to_csv(args.output)

    genomes_with_genes = int((feature_matrix.sum(axis=1) > 0).sum())
    print("\nFeature Extraction Summary")
    print("=" * 72)
    print(f"Raw specialty gene hits       : {len(raw_sp_df):,}")
    if not raw_sp_df.empty and "source" in raw_sp_df.columns:
        print("Hits by curated database      :")
        for src, cnt in raw_sp_df["source"].value_counts().items():
            print(f"  {src:10s}: {cnt:,}")
    print(f"Genomes in feature matrix     : {feature_matrix.shape[0]:,}")
    print(f"Genomes with >= 1 AMR gene    : {genomes_with_genes:,}")
    print(f"Unique AMR gene features      : {feature_matrix.shape[1]:,}")
    print(f"Raw hits saved to             : {args.raw_output}")
    print(f"Feature matrix saved to       : {args.output}")


if __name__ == "__main__":
    main()
