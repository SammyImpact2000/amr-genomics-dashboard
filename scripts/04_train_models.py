#!/usr/bin/env python3
"""Train explainable XGBoost AMR classifiers with SHAP gene attributions.

Merges data/processed/gene_feature_matrix.csv with data/processed/amr_label_matrix.csv,
trains one XGBClassifier per core antibiotic using a stratified 80/20 train/test split
(with scale_pos_weight for class imbalance), saves trained models and feature column
lists to models/, prints evaluation metrics and top 5 SHAP genes per drug, and writes
a full report to reports/model_evaluation.md.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

CORE_ANTIBIOTICS = [
    "meropenem",
    "ceftriaxone",
    "ciprofloxacin",
    "gentamicin",
    "trimethoprim-sulfamethoxazole",
]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "gene_feature_matrix.csv"
DEFAULT_LABELS_PATH = PROJECT_ROOT / "data" / "processed" / "amr_label_matrix.csv"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "model_evaluation.md"


def slugify_drug(drug: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", drug.lower()).strip("_")


def compute_shap_values(
    model: XGBClassifier,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> np.ndarray:
    """Compute per-sample SHAP values for the positive (Resistant) class."""
    try:
        explainer = shap.TreeExplainer(model)
        raw_shap = explainer.shap_values(X_test)
    except Exception:
        # Fallback for any XGBoost/SHAP version edge case with TreeExplainer
        explainer = shap.Explainer(model.predict_proba, X_train)
        raw_shap = explainer(X_test).values

    arr = np.asarray(raw_shap)
    if arr.ndim == 3:
        # Shape (n_samples, n_features, 2) -> take class 1 (Resistant)
        arr = arr[:, :, 1]
    return arr


def extract_top_shap_genes(
    shap_values: np.ndarray,
    X_test: pd.DataFrame,
    top_k: int = 5,
) -> list[dict[str, object]]:
    """Return the top_k genes ranked by mean(|SHAP|) with directional impact."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs)[::-1][:top_k]

    top_genes: list[dict[str, object]] = []
    for rank, idx in enumerate(top_indices, start=1):
        gene = X_test.columns[idx]
        gene_vals = X_test.iloc[:, idx].to_numpy()
        gene_shap = shap_values[:, idx]
        present_mask = gene_vals == 1
        present_count = int(present_mask.sum())
        mean_shap_when_present = (
            float(gene_shap[present_mask].mean()) if present_count > 0 else 0.0
        )
        direction = "Resistant (+)" if mean_shap_when_present >= 0 else "Susceptible (-)"

        top_genes.append(
            {
                "rank": rank,
                "gene": gene,
                "mean_abs_shap": float(mean_abs[idx]),
                "mean_shap_when_present": mean_shap_when_present,
                "direction": direction,
                "test_prevalence": f"{present_count}/{len(X_test)}",
            }
        )
    return top_genes


def build_markdown_report(
    metrics_df: pd.DataFrame,
    shap_by_drug: dict[str, list[dict[str, object]]],
    n_genomes: int,
    n_features: int,
) -> str:
    """Format model evaluation metrics and top SHAP features as Markdown."""
    lines: list[str] = [
        "# Phase 2 Model Evaluation & SHAP Explainability Report",
        "",
        f"- **Cohort size (merged genomes):** {n_genomes:,}",
        f"- **AMR gene features:** {n_features:,} (binary presence/absence from BV-BRC Specialty Genes: CARD, NDARO, PATRIC)",
        "- **Model:** `XGBClassifier` (`n_estimators=200`, `max_depth=4`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, class-balanced via `scale_pos_weight`)",
        "- **Validation:** 80/20 stratified train/test split (`random_state=42`)",
        "",
        "## Per-Drug Classification Performance (Held-Out 20% Test Set)",
        "",
        "| Antibiotic | N Total | Train / Test | Test R / S | `scale_pos_weight` | ROC-AUC | F1-Score | Precision | Recall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in metrics_df.iterrows():
        lines.append(
            f"| `{row['antibiotic']}` | {int(row['n_total'])} | "
            f"{int(row['n_train'])} / {int(row['n_test'])} | "
            f"{int(row['test_resistant'])} / {int(row['test_susceptible'])} | "
            f"{row['scale_pos_weight']:.2f} | "
            f"{row['roc_auc']:.4f} | {row['f1']:.4f} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} |"
        )

    lines.extend(["", "## Top 5 SHAP Feature-Importance Genes per Antibiotic", ""])

    for drug in CORE_ANTIBIOTICS:
        lines.append(f"### `{drug}`")
        lines.append("")
        lines.append(
            "| Rank | Resistance Gene | Mean \\|SHAP\\| | Mean SHAP (Gene=1) | Effect Direction | Test Presence |"
        )
        lines.append("|---:|---|---:|---:|---|---:|")
        for item in shap_by_drug.get(drug, []):
            lines.append(
                f"| {item['rank']} | `{item['gene']}` | {item['mean_abs_shap']:.4f} | "
                f"{item['mean_shap_when_present']:+.4f} | {item['direction']} | {item['test_prevalence']} |"
            )
        lines.append("")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train XGBoost AMR classifiers and compute SHAP feature importances."
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=DEFAULT_FEATURES_PATH,
        help=f"Binary gene feature matrix CSV (default: {DEFAULT_FEATURES_PATH})",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=DEFAULT_LABELS_PATH,
        help=f"Cleaned binary susceptibility label matrix CSV (default: {DEFAULT_LABELS_PATH})",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help=f"Directory to save joblib models and feature columns (default: {DEFAULT_MODELS_DIR})",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help=f"Output Markdown evaluation report path (default: {DEFAULT_REPORT_PATH})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/test splits and XGBoost (default: 42).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.features.exists():
        raise FileNotFoundError(
            f"Feature matrix not found at {args.features}. "
            "Run scripts/03_fetch_gene_features.py first."
        )
    if not args.labels.exists():
        raise FileNotFoundError(
            f"Label matrix not found at {args.labels}. "
            "Run scripts/02_clean_labels.py first."
        )

    feat_df = pd.read_csv(args.features, dtype={"genome_id": str})
    label_df = pd.read_csv(args.labels, dtype={"genome_id": str})

    feature_cols = [c for c in feat_df.columns if c != "genome_id"]
    merged = pd.merge(
        feat_df,
        label_df[["genome_id", *CORE_ANTIBIOTICS]],
        on="genome_id",
        how="inner",
    )

    print(
        f"Merged dataset: {len(merged):,} genomes x {len(feature_cols):,} AMR gene features."
    )

    args.models_dir.mkdir(parents=True, exist_ok=True)
    feature_cols_path = args.models_dir / "feature_columns.joblib"
    joblib.dump(feature_cols, feature_cols_path)

    metric_rows: list[dict[str, object]] = []
    shap_by_drug: dict[str, list[dict[str, object]]] = {}

    for drug in CORE_ANTIBIOTICS:
        subset = merged.dropna(subset=[drug]).copy()
        X = subset[feature_cols].astype(np.float32)
        y = subset[drug].astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            stratify=y,
            random_state=args.seed,
        )

        n_pos = int((y_train == 1).sum())
        n_neg = int((y_train == 0).sum())
        scale_pos_weight = float(n_neg / n_pos) if n_pos > 0 else 1.0

        model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=args.seed,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        roc_auc = float(roc_auc_score(y_test, y_proba))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        precision = float(precision_score(y_test, y_pred, zero_division=0))
        recall = float(recall_score(y_test, y_pred, zero_division=0))

        model_path = args.models_dir / f"xgb_{slugify_drug(drug)}.joblib"
        joblib.dump(
            {
                "antibiotic": drug,
                "model": model,
                "feature_columns": feature_cols,
                "scale_pos_weight": scale_pos_weight,
            },
            model_path,
        )

        shap_vals = compute_shap_values(model, X_train, X_test)
        top_genes = extract_top_shap_genes(shap_vals, X_test, top_k=5)
        shap_by_drug[drug] = top_genes

        metric_rows.append(
            {
                "antibiotic": drug,
                "n_total": len(subset),
                "n_train": len(X_train),
                "n_test": len(X_test),
                "test_resistant": int((y_test == 1).sum()),
                "test_susceptible": int((y_test == 0).sum()),
                "scale_pos_weight": round(scale_pos_weight, 3),
                "roc_auc": round(roc_auc, 4),
                "f1": round(f1, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
            }
        )

    metrics_df = pd.DataFrame(metric_rows)

    print("\nModel Evaluation Summary (80/20 Stratified Split)")
    print("=" * 108)
    print(metrics_df.to_string(index=False))

    print("\nTop 5 SHAP Feature-Importance Genes per Antibiotic")
    print("=" * 108)
    for drug in CORE_ANTIBIOTICS:
        print(f"\n[{drug}]")
        for item in shap_by_drug[drug]:
            print(
                f"  {item['rank']}. {item['gene']:35s} "
                f"mean|SHAP|={item['mean_abs_shap']:.4f}  "
                f"meanSHAP(1)={item['mean_shap_when_present']:+.4f}  "
                f"({item['direction']}, test={item['test_prevalence']})"
            )

    report_md = build_markdown_report(
        metrics_df=metrics_df,
        shap_by_drug=shap_by_drug,
        n_genomes=len(merged),
        n_features=len(feature_cols),
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report_md, encoding="utf-8")

    print(f"\nSaved models & feature list to: {args.models_dir}")
    print(f"Saved evaluation report to    : {args.report}")


if __name__ == "__main__":
    main()
