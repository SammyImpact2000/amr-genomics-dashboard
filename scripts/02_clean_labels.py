#!/usr/bin/env python3
"""Clean BV-BRC AMR metadata into a deduplicated binary susceptibility matrix.

Reads raw metadata fetched by scripts/01_fetch_metadata.py, removes exact
duplicate and conflicting (genome_id, antibiotic) labels, pivots the cleaned
labels into a binary CSV matrix (1 = Resistant, 0 = Susceptible), saves the
matrix and class-balance report, and prints a class-balance summary table.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TARGET_ANTIBIOTICS = [
    "meropenem",
    "ceftriaxone",
    "ciprofloxacin",
    "gentamicin",
    "ampicillin",
    "trimethoprim-sulfamethoxazole",
]

ANTIBIOTIC_NORMALIZATION = {
    "trimethoprim/sulfamethoxazole": "trimethoprim-sulfamethoxazole",
    "trimethoprim-sulfamethoxazole": "trimethoprim-sulfamethoxazole",
}

PHENOTYPE_TO_BINARY = {
    "Resistant": 1,
    "Susceptible": 0,
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "metadata" / "bv_brc_amr_taxon573.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "amr_label_matrix.csv"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "class_balance_summary.csv"


def normalize_antibiotic(name: object) -> object:
    """Normalize antibiotic names to canonical lowercase dashboard spellings."""
    if pd.isna(name):
        return np.nan
    cleaned = str(name).strip().lower()
    return ANTIBIOTIC_NORMALIZATION.get(cleaned, cleaned)


def clean_and_pivot_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Filter, deduplicate, resolve conflicts, and pivot AMR labels to a binary matrix."""
    required_cols = {"genome_id", "antibiotic", "resistant_phenotype"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")

    work = df.copy()
    initial_rows = len(work)

    # Filter strictly to laboratory-tested records if the evidence column is present.
    if "evidence" in work.columns:
        work = work[work["evidence"].astype(str).str.strip() == "Laboratory Method"]

    work["genome_id"] = work["genome_id"].astype(str).str.strip()
    work["antibiotic"] = work["antibiotic"].map(normalize_antibiotic)
    work["resistant_phenotype"] = work["resistant_phenotype"].astype(str).str.strip()

    work = work[
        work["genome_id"].ne("")
        & work["antibiotic"].isin(TARGET_ANTIBIOTICS)
        & work["resistant_phenotype"].isin(PHENOTYPE_TO_BINARY)
    ].copy()
    filtered_rows = len(work)

    # 1. Remove exact duplicate (genome_id, antibiotic, resistant_phenotype) rows.
    deduped = work.drop_duplicates(
        subset=["genome_id", "antibiotic", "resistant_phenotype"]
    ).copy()
    exact_duplicates_removed = filtered_rows - len(deduped)

    # 2. Identify and drop conflicting (genome_id, antibiotic) pairs where both
    #    Resistant and Susceptible labels exist for the same genome and drug.
    label_counts = deduped.groupby(["genome_id", "antibiotic"])["resistant_phenotype"].transform(
        "nunique"
    )
    conflict_mask = label_counts > 1
    conflicting_rows_removed = int(conflict_mask.sum())
    conflicting_pairs_removed = int(
        deduped.loc[conflict_mask, ["genome_id", "antibiotic"]].drop_duplicates().shape[0]
    )

    clean = deduped.loc[~conflict_mask].copy()
    clean["label"] = clean["resistant_phenotype"].map(PHENOTYPE_TO_BINARY).astype("Int64")

    # 3. Pivot into a binary matrix (rows = genome_id, columns = target antibiotics).
    matrix = clean.pivot(index="genome_id", columns="antibiotic", values="label")
    matrix = matrix.reindex(columns=TARGET_ANTIBIOTICS).astype("Int64")
    matrix.index.name = "genome_id"
    matrix = matrix.sort_index()

    stats = {
        "initial_rows": initial_rows,
        "filtered_rows": filtered_rows,
        "exact_duplicates_removed": exact_duplicates_removed,
        "conflicting_pairs_removed": conflicting_pairs_removed,
        "conflicting_rows_removed": conflicting_rows_removed,
        "clean_pairs_retained": len(clean),
        "unique_genomes": len(matrix),
    }
    return matrix, stats


def build_class_balance_summary(matrix: pd.DataFrame) -> pd.DataFrame:
    """Compute per-antibiotic class-balance statistics from the binary label matrix."""
    total_genomes = len(matrix)
    rows: list[dict[str, object]] = []

    for antibiotic in TARGET_ANTIBIOTICS:
        series = matrix[antibiotic]
        resistant_count = int((series == 1).sum())
        susceptible_count = int((series == 0).sum())
        tested_count = resistant_count + susceptible_count
        missing_count = total_genomes - tested_count

        if tested_count > 0:
            resistant_pct = round(100.0 * resistant_count / tested_count, 2)
            susceptible_pct = round(100.0 * susceptible_count / tested_count, 2)
        else:
            resistant_pct = 0.0
            susceptible_pct = 0.0

        minority = min(resistant_count, susceptible_count)
        majority = max(resistant_count, susceptible_count)
        imbalance_ratio = f"{majority / minority:.2f}:1" if minority > 0 else "N/A"

        rows.append(
            {
                "antibiotic": antibiotic,
                "tested": tested_count,
                "resistant_1": resistant_count,
                "susceptible_0": susceptible_count,
                "resistant_pct": resistant_pct,
                "susceptible_pct": susceptible_pct,
                "missing": missing_count,
                "imbalance_ratio": imbalance_ratio,
            }
        )

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deduplicate BV-BRC AMR labels and pivot into a binary CSV matrix."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Input raw metadata CSV path (default: {DEFAULT_INPUT_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output binary matrix CSV path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help=f"Output class-balance summary CSV path (default: {DEFAULT_REPORT_PATH})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(
            f"Raw metadata file not found at {args.input}. "
            "Run scripts/01_fetch_metadata.py first."
        )

    raw_df = pd.read_csv(args.input, dtype={"genome_id": str}, low_memory=False)
    matrix, stats = clean_and_pivot_labels(raw_df)
    summary_df = build_class_balance_summary(matrix)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(args.output)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.report, index=False)

    print("Label Cleaning Summary")
    print("=" * 72)
    print(f"Raw rows loaded               : {stats['initial_rows']:,}")
    print(f"Valid target-drug lab rows    : {stats['filtered_rows']:,}")
    print(f"Exact duplicate rows removed  : {stats['exact_duplicates_removed']:,}")
    print(
        f"Conflicting pairs removed     : {stats['conflicting_pairs_removed']:,} "
        f"({stats['conflicting_rows_removed']:,} rows)"
    )
    print(f"Clean (genome, drug) labels   : {stats['clean_pairs_retained']:,}")
    print(f"Unique genomes in matrix      : {stats['unique_genomes']:,}")
    print(f"\nBinary matrix saved to        : {args.output}")
    print(f"Class-balance report saved to : {args.report}")
    print("\nClass-Balance Summary Table (1 = Resistant, 0 = Susceptible)")
    print("=" * 96)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
