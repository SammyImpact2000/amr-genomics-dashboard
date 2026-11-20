#!/usr/bin/env python3
"""Interactive Klebsiella pneumoniae AMR Susceptibility Predictor & Surveillance Dashboard.

Powered by laboratory-verified BV-BRC susceptibility data, curated AMR specialty
gene features (CARD, NDARO, PATRIC), and explainable XGBoost + SHAP models.
"""

from __future__ import annotations

import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="K. pneumoniae AMR Genomics Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

LABEL_MATRIX_PATH = DATA_DIR / "amr_label_matrix.csv"
GENE_MATRIX_PATH = DATA_DIR / "gene_feature_matrix.csv"
SURVEILLANCE_PATH = DATA_DIR / "genome_surveillance_metadata.csv"
EVAL_REPORT_PATH = REPORTS_DIR / "model_evaluation.md"

CORE_ANTIBIOTICS = [
    "meropenem",
    "ceftriaxone",
    "ciprofloxacin",
    "gentamicin",
    "trimethoprim-sulfamethoxazole",
]

DRUG_DISPLAY = {
    "meropenem": ("Meropenem", "Carbapenem"),
    "ceftriaxone": ("Ceftriaxone", "3rd-Gen Cephalosporin"),
    "ciprofloxacin": ("Ciprofloxacin", "Fluoroquinolone"),
    "gentamicin": ("Gentamicin", "Aminoglycoside"),
    "trimethoprim-sulfamethoxazole": ("TMP-SMX", "Folate Pathway Inhibitor"),
}

KEY_SIMULATOR_GENES = [
    ("KPC-2", "Carbapenemase (Class A)"),
    ("OXA-48 family", "Carbapenemase (Class D)"),
    ("BRP(MBL)", "NDM-linked Bleomycin Resistance"),
    ("NDM-1", "Metallo-β-lactamase"),
    ("CTX-M-15", "ESBL (3rd-Gen Cephalosporin)"),
    ("TEM-1", "Broad-Spectrum β-lactamase"),
    ("SHV-11", "Chromosomal β-lactamase"),
    ("Escherichia coli parC conferring resistance to fluoroquinolones", "QRDR Topoisomerase IV"),
    ("Salmonella isangi gyrA conferring resistance to fluoroquinolones", "QRDR DNA Gyrase"),
    ("QnrS1", "Plasmid Quinolone Resistance"),
    ("AAC(3)-IIe", "Aminoglycoside Acetyltransferase"),
    ("AAC(3)-II,III,IV,VI,VIII,IX,X", "Aminoglycoside Acetyltransferase"),
    ("sul1", "Sulfonamide Target Bypass"),
    ("sul2", "Sulfonamide Target Bypass"),
    ("qacEdelta1", "Class 1 Integron Marker"),
    ("aadA2", "Integron Cassette"),
]

BIOLOGICAL_CALLOUTS = {
    "meropenem": (
        "Carbapenem resistance is strongly driven by **KPC-2**, **OXA-48 family**, and "
        "**BRP(MBL)** (co-carried on *bla*NDM metallo-β-lactamase plasmids), alongside the "
        "high-risk MDR clonal marker **parC**."
    ),
    "ceftriaxone": (
        "**CTX-M-15** is the dominant global extended-spectrum β-lactamase (ESBL) hydrolyzing "
        "third-generation cephalosporins in *K. pneumoniae*, reinforced by MDR lineage markers."
    ),
    "ciprofloxacin": (
        "Fluoroquinolone resistance maps directly to canonical Quinolone Resistance-Determining "
        "Region (QRDR) mutations in **parC** and **gyrA**, plus plasmid-mediated **QnrS1**."
    ),
    "gentamicin": (
        "Gentamicin inactivation is driven by **AAC(3)-IIe** and the **AAC(3)-II** family of "
        "aminoglycoside 3-N-acetyltransferases, the primary enzymatic mechanism in Enterobacterales."
    ),
    "trimethoprim-sulfamethoxazole": (
        "TMP-SMX resistance is anchored by sulfonamide-resistant dihydropteroate synthases "
        "**sul2** and **sul1**, together with **qacEdelta1** (the 3′-conserved segment of class 1 "
        "integrons that co-carry *dfrA* trimethoprim resistance cassettes)."
    ),
}

# ---------------------------------------------------------------------------
# Custom CSS (Light & Dark Theme Compatible)
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2.5rem;
    }
    .hero-banner {
        background: linear-gradient(135deg, rgba(15, 118, 110, 0.14), rgba(37, 99, 235, 0.12));
        border: 1px solid rgba(148, 163, 184, 0.28);
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.25rem;
    }
    .hero-title {
        font-size: 1.65rem;
        font-weight: 700;
        margin: 0 0 0.25rem 0;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        opacity: 0.85;
        margin: 0;
    }
    .kpi-card {
        background: rgba(148, 163, 184, 0.08);
        border: 1px solid rgba(148, 163, 184, 0.24);
        border-radius: 12px;
        padding: 0.95rem 1.1rem;
        height: 100%;
    }
    .kpi-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        opacity: 0.72;
        margin-bottom: 0.25rem;
        font-weight: 600;
    }
    .kpi-value {
        font-size: 1.65rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .kpi-sub {
        font-size: 0.8rem;
        opacity: 0.75;
        margin-top: 0.2rem;
    }
    .badge {
        display: inline-block;
        padding: 0.22rem 0.65rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.03em;
    }
    .badge-red {
        background: rgba(239, 68, 68, 0.16);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.42);
    }
    .badge-green {
        background: rgba(16, 185, 129, 0.16);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.42);
    }
    .badge-amber {
        background: rgba(245, 158, 11, 0.16);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.45);
    }
    .badge-neutral {
        background: rgba(148, 163, 184, 0.16);
        color: inherit;
        border: 1px solid rgba(148, 163, 184, 0.35);
    }
    .antibiogram-card {
        background: rgba(148, 163, 184, 0.07);
        border: 1px solid rgba(148, 163, 184, 0.24);
        border-radius: 12px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.65rem;
    }
    .prob-track {
        width: 100%;
        height: 8px;
        background: rgba(148, 163, 184, 0.22);
        border-radius: 999px;
        overflow: hidden;
        margin: 0.45rem 0;
    }
    .sidebar-stat {
        background: rgba(148, 163, 184, 0.09);
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 10px;
        padding: 0.7rem 0.85rem;
        margin-bottom: 0.55rem;
        font-size: 0.85rem;
    }
    .bio-callout {
        background: rgba(16, 185, 129, 0.09);
        border-left: 4px solid #10b981;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        margin-top: 0.6rem;
        font-size: 0.9rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached Data & Model Loaders
# ---------------------------------------------------------------------------
def slugify_drug(drug: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", drug.lower()).strip("_")


@st.cache_data(show_spinner=False)
def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels = pd.read_csv(LABEL_MATRIX_PATH, dtype={"genome_id": str})
    genes = pd.read_csv(GENE_MATRIX_PATH, dtype={"genome_id": str})
    if SURVEILLANCE_PATH.exists():
        surv = pd.read_csv(SURVEILLANCE_PATH, dtype={"genome_id": str})
        surv["collection_year"] = pd.to_numeric(surv["collection_year"], errors="coerce")
    else:
        surv = pd.DataFrame(
            columns=[
                "genome_id",
                "genome_name",
                "isolation_country",
                "collection_year",
                "host_name",
                "isolation_source",
            ]
        )
    return labels, genes, surv


@st.cache_resource(show_spinner=False)
def load_models_and_explainers() -> tuple[dict[str, dict], dict[str, shap.TreeExplainer], list[str]]:
    feature_cols: list[str] = joblib.load(MODELS_DIR / "feature_columns.joblib")
    models: dict[str, dict] = {}
    explainers: dict[str, shap.TreeExplainer] = {}
    for drug in CORE_ANTIBIOTICS:
        bundle = joblib.load(MODELS_DIR / f"xgb_{slugify_drug(drug)}.joblib")
        models[drug] = bundle
        explainers[drug] = shap.TreeExplainer(bundle["model"])
    return models, explainers, feature_cols


@st.cache_data(show_spinner=False)
def get_test_cohort_ids(labels: pd.DataFrame, genes: pd.DataFrame) -> set[str]:
    """Reproduce the exact held-out 20% stratified test split genome_ids (seed=42)."""
    merged = pd.merge(genes[["genome_id"]], labels[["genome_id", *CORE_ANTIBIOTICS]], on="genome_id")
    test_ids: set[str] = set()
    for drug in CORE_ANTIBIOTICS:
        sub = merged.dropna(subset=[drug])
        _, test_sub = train_test_split(
            sub["genome_id"],
            test_size=0.20,
            stratify=sub[drug].astype(int),
            random_state=42,
        )
        test_ids.update(test_sub.tolist())
    return test_ids


@st.cache_data(show_spinner=False)
def parse_model_evaluation_md() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Parse metrics table and per-drug top 5 SHAP tables from reports/model_evaluation.md."""
    text = EVAL_REPORT_PATH.read_text(encoding="utf-8")

    # Parse main metrics table
    metric_rows: list[dict[str, object]] = []
    metric_pattern = re.compile(
        r"^\|\s*`([^`]+)`\s*\|\s*(\d+)\s*\|\s*(\d+)\s*/\s*(\d+)\s*\|\s*(\d+)\s*/\s*(\d+)\s*\|"
        r"\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|$",
        re.MULTILINE,
    )
    for m in metric_pattern.finditer(text):
        metric_rows.append(
            {
                "antibiotic": m.group(1),
                "n_total": int(m.group(2)),
                "n_train": int(m.group(3)),
                "n_test": int(m.group(4)),
                "test_r": int(m.group(5)),
                "test_s": int(m.group(6)),
                "scale_pos_weight": float(m.group(7)),
                "ROC-AUC": float(m.group(8)),
                "F1-Score": float(m.group(9)),
                "Precision": float(m.group(10)),
                "Recall": float(m.group(11)),
            }
        )
    metrics_df = pd.DataFrame(metric_rows)

    # Parse per-drug top 5 SHAP tables
    shap_tables: dict[str, pd.DataFrame] = {}
    sections = re.split(r"^###\s+`([^`]+)`\s*$", text, flags=re.MULTILINE)
    row_pattern = re.compile(
        r"^\|\s*(\d+)\s*\|\s*`([^`]+)`\s*\|\s*([\d.]+)\s*\|\s*([+-]?[\d.]+)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|$",
        re.MULTILINE,
    )
    for i in range(1, len(sections), 2):
        drug = sections[i].strip()
        body = sections[i + 1]
        rows = []
        for rm in row_pattern.finditer(body):
            rows.append(
                {
                    "Rank": int(rm.group(1)),
                    "Resistance Gene": rm.group(2),
                    "Mean |SHAP|": float(rm.group(3)),
                    "Mean SHAP (Gene=1)": float(rm.group(4)),
                    "Effect Direction": rm.group(5).strip(),
                    "Test Presence": rm.group(6).strip(),
                }
            )
        shap_tables[drug] = pd.DataFrame(rows)

    return metrics_df, shap_tables


# ---------------------------------------------------------------------------
# Helper UI Functions
# ---------------------------------------------------------------------------
def classify_confidence(prob: float) -> tuple[str, str, str]:
    """Return (label, badge_css_class, hex_color) for a resistance probability."""
    if prob >= 0.65:
        return "Resistant", "badge-red", "#ef4444"
    if prob <= 0.35:
        return "Susceptible", "badge-green", "#10b981"
    label = "Borderline (R)" if prob >= 0.50 else "Borderline (S)"
    return label, "badge-amber", "#f59e0b"


def shorten_gene_name(name: str, max_len: int = 38) -> str:
    if len(name) <= max_len:
        return name
    return name[: max_len - 1] + "…"


def render_kpi(label: str, value: str, sub: str = "") -> None:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def compute_single_shap(
    explainer: shap.TreeExplainer,
    x_row: pd.DataFrame,
) -> tuple[float, np.ndarray]:
    sv = explainer.shap_values(x_row)
    arr = np.asarray(sv)
    if arr.ndim == 3:
        arr = arr[0, :, 1]
    elif arr.ndim == 2:
        arr = arr[0, :]
    ev = explainer.expected_value
    if isinstance(ev, (list, np.ndarray)):
        base_val = float(np.asarray(ev).ravel()[-1])
    else:
        base_val = float(ev)
    return base_val, arr


def build_sample_upload_csv() -> str:
    """Return a realistic sample CARD RGI / AMRFinderPlus-style CSV for testing uploads."""
    sample_rows = [
        ("SAMPLE_CRE_001", "KPC-2", "CARD", 100.0),
        ("SAMPLE_CRE_001", "CTX-M-15", "CARD", 99.8),
        ("SAMPLE_CRE_001", "Escherichia coli parC conferring resistance to fluoroquinolones", "CARD", 98.5),
        ("SAMPLE_CRE_001", "AAC(3)-IIe", "CARD", 100.0),
        ("SAMPLE_CRE_001", "sul1", "NDARO", 100.0),
        ("SAMPLE_CRE_001", "sul2", "NDARO", 99.6),
        ("SAMPLE_CRE_001", "qacEdelta1", "CARD", 100.0),
        ("SAMPLE_CRE_001", "TEM-1", "CARD", 100.0),
        ("SAMPLE_ESBL_002", "blaCTX-M-15", "AMRFinderPlus", 100.0),
        ("SAMPLE_ESBL_002", "QnrS1", "CARD", 99.5),
        ("SAMPLE_ESBL_002", "SHV-11", "CARD", 100.0),
        ("SAMPLE_ESBL_002", "sul1", "NDARO", 100.0),
    ]
    df = pd.DataFrame(
        sample_rows,
        columns=["isolate_id", "gene_symbol", "source_db", "pct_identity"],
    )
    return df.to_csv(index=False)


def detect_upload_columns(df: pd.DataFrame) -> tuple[str, str | None]:
    """Auto-detect gene column (CARD RGI, AMRFinderPlus, or generic) and optional isolate ID column."""
    lower_map = {str(c).strip().lower(): c for c in df.columns}

    gene_candidates = [
        "best_hit_aro",
        "gene symbol",
        "element symbol",
        "gene_symbol",
        "gene",
        "gene_name",
        "aro_name",
        "amr_gene",
        "resistance_gene",
        "feature",
    ]
    gene_col = next((lower_map[c] for c in gene_candidates if c in lower_map), df.columns[0])

    isolate_candidates = [
        "isolate_id",
        "genome_id",
        "sample_id",
        "sample",
        "isolate",
        "strain",
        "contig_isolate",
    ]
    isolate_col = next(
        (lower_map[c] for c in isolate_candidates if c in lower_map and lower_map[c] != gene_col),
        None,
    )
    return gene_col, isolate_col


def map_uploaded_genes(
    raw_genes: list[str],
    feature_cols: list[str],
) -> tuple[list[str], list[str]]:
    """Map uploaded gene symbols to model feature columns (case-insensitive + bla-prefix aware)."""
    lookup = {col.strip().lower(): col for col in feature_cols}
    matched: set[str] = set()
    unmatched: list[str] = []

    for raw in raw_genes:
        cleaned = str(raw).strip()
        if not cleaned or cleaned.lower() == "nan":
            continue
        key = cleaned.lower()
        if key in lookup:
            matched.add(lookup[key])
            continue
        # Handle AMRFinderPlus `bla` prefix (e.g., blaKPC-2 -> KPC-2, blaCTX-M-15 -> CTX-M-15)
        if key.startswith("bla") and key[3:] in lookup:
            matched.add(lookup[key[3:]])
            continue
        # Handle underscore/space normalization
        norm_key = re.sub(r"[_]+", " ", key)
        if norm_key in lookup:
            matched.add(lookup[norm_key])
            continue
        unmatched.append(cleaned)

    return sorted(matched), sorted(set(unmatched))


def build_antibiogram_export_df(
    isolate_id: str,
    pred_records: list[dict[str, object]],
    explainers: dict[str, shap.TreeExplainer],
    x_input: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Build exportable CSV antibiogram report including top 3 SHAP driving genes per drug."""
    export_rows: list[dict[str, object]] = []
    for rec in pred_records:
        drug = str(rec["drug"])
        _, shap_vals = compute_single_shap(explainers[drug], x_input)
        top_idx = np.argsort(np.abs(shap_vals))[::-1][:3]
        top3_str = "; ".join(
            f"{feature_cols[i]} ({shap_vals[i]:+.3f})" for i in top_idx
        )
        actual_val = rec["actual"]
        actual_str = (
            "N/A"
            if actual_val is None
            else ("Resistant" if actual_val == 1 else "Susceptible")
        )
        export_rows.append(
            {
                "isolate_id": isolate_id,
                "antibiotic": rec["display"],
                "drug_class": rec["drug_class"],
                "predicted_phenotype": rec["conf_label"],
                "resistance_probability_pct": round(float(rec["prob"]) * 100.0, 2),
                "actual_lab_phenotype": actual_str,
                "top_3_shap_genes": top3_str,
            }
        )
    return pd.DataFrame(export_rows)


# ---------------------------------------------------------------------------
# View 1: In-Silico Antibiogram & SHAP Explainer
# ---------------------------------------------------------------------------
def render_antibiogram_view(
    labels: pd.DataFrame,
    genes: pd.DataFrame,
    surv: pd.DataFrame,
    models: dict[str, dict],
    explainers: dict[str, shap.TreeExplainer],
    feature_cols: list[str],
) -> None:
    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-title">🧬 In-Silico Antibiogram & SHAP Gene Explainer</div>
            <p class="hero-subtitle">
                Predict phenotypic antimicrobial resistance from detected AMR genes and inspect
                per-gene SHAP log-odds contributions in real time.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    mode = st.radio(
        "Input Mode",
        options=[
            "Real Isolate (Laboratory Test Cohort)",
            "Custom Resistance Gene Simulator",
            "Upload Gene Call Table (CSV/TSV)",
        ],
        horizontal=True,
        label_visibility="collapsed",
    )

    actual_labels: dict[str, int | None] = {d: None for d in CORE_ANTIBIOTICS}
    isolate_header = ""
    current_isolate_id = "custom_profile"
    active_genes_list: list[str] = []
    x_input: pd.DataFrame | None = None

    if mode.startswith("Real Isolate"):
        test_ids = get_test_cohort_ids(labels, genes)
        cohort_filter = st.checkbox(
            f"Restrict to held-out 20% test isolates only ({len(test_ids)} genomes)",
            value=True,
        )

        cohort = genes[["genome_id"]].merge(surv, on="genome_id", how="left")
        cohort = cohort.merge(labels[["genome_id", *CORE_ANTIBIOTICS]], on="genome_id", how="left")
        if cohort_filter:
            cohort = cohort[cohort["genome_id"].isin(test_ids)].copy()

        cohort = cohort.sort_values("genome_id").reset_index(drop=True)

        def fmt_option(idx: int) -> str:
            r = cohort.iloc[idx]
            country = r["isolation_country"] if pd.notna(r.get("isolation_country")) else "Unknown Country"
            year = str(int(r["collection_year"])) if pd.notna(r.get("collection_year")) else "Year N/A"
            n_res = int(r[CORE_ANTIBIOTICS].fillna(0).sum())
            return f"{r['genome_id']}  |  {country} ({year})  |  Lab R: {n_res}/5"

        selected_idx = st.selectbox(
            "Select Isolate (`genome_id`)",
            options=list(range(len(cohort))),
            format_func=fmt_option,
        )
        sel_row = cohort.iloc[selected_idx]
        gid = sel_row["genome_id"]
        current_isolate_id = str(gid)

        gene_row = genes.loc[genes["genome_id"] == gid, feature_cols].iloc[0]
        x_input = pd.DataFrame([gene_row.values], columns=feature_cols, dtype=np.float32)
        active_genes_list = [g for g in feature_cols if gene_row[g] == 1]

        for d in CORE_ANTIBIOTICS:
            val = sel_row[d]
            actual_labels[d] = int(val) if pd.notna(val) else None

        country_str = sel_row["isolation_country"] if pd.notna(sel_row.get("isolation_country")) else "Unknown"
        year_str = str(int(sel_row["collection_year"])) if pd.notna(sel_row.get("collection_year")) else "N/A"
        host_str = sel_row["host_name"] if pd.notna(sel_row.get("host_name")) else "N/A"
        src_str = sel_row["isolation_source"] if pd.notna(sel_row.get("isolation_source")) else "N/A"
        gname = sel_row["genome_name"] if pd.notna(sel_row.get("genome_name")) else gid
        isolate_header = f"**{gname}** (`{gid}`)"

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            render_kpi("Country", str(country_str), "BV-BRC Isolation Location")
        with c2:
            render_kpi("Collection Year", year_str, f"Host: {host_str}")
        with c3:
            render_kpi("Detected AMR Genes", f"{len(active_genes_list)}", f"Source: {str(src_str)[:26]}")
        with c4:
            n_lab_r = sum(1 for v in actual_labels.values() if v == 1)
            render_kpi("Lab Resistant Drugs", f"{n_lab_r} / 5", "Laboratory Method Ground Truth")

        with st.expander(f"View all {len(active_genes_list)} detected AMR genes in `{gid}`"):
            st.write(", ".join(f"`{g}`" for g in active_genes_list))

    elif mode.startswith("Custom Resistance"):
        # Custom Resistance Gene Simulator
        st.markdown("#### Simulate a Custom Resistance Gene Profile")
        available_key_genes = [
            (g, desc) for g, desc in KEY_SIMULATOR_GENES if g in feature_cols
        ]

        # Preset buttons
        p1, p2, p3, p4 = st.columns(4)
        if "sim_genes" not in st.session_state:
            st.session_state["sim_genes"] = [
                g for g in ["CTX-M-15", "KPC-2"] if g in feature_cols
            ]

        if p1.button("Pan-Susceptible Baseline", width="stretch"):
            st.session_state["sim_genes"] = []
        if p2.button("ESBL (CTX-M-15)", width="stretch"):
            st.session_state["sim_genes"] = [
                g for g in ["CTX-M-15", "TEM-1"] if g in feature_cols
            ]
        if p3.button("CRE (KPC-2 + QRDR)", width="stretch"):
            st.session_state["sim_genes"] = [
                g
                for g in [
                    "KPC-2",
                    "CTX-M-15",
                    "Escherichia coli parC conferring resistance to fluoroquinolones",
                    "sul1",
                ]
                if g in feature_cols
            ]
        if p4.button("XDR Clinical Profile", width="stretch"):
            st.session_state["sim_genes"] = [
                g
                for g in [
                    "KPC-2",
                    "CTX-M-15",
                    "Escherichia coli parC conferring resistance to fluoroquinolones",
                    "Salmonella isangi gyrA conferring resistance to fluoroquinolones",
                    "AAC(3)-IIe",
                    "sul1",
                    "sul2",
                    "qacEdelta1",
                ]
                if g in feature_cols
            ]

        st.caption("Toggle high-impact clinical resistance markers:")
        cols = st.columns(4)
        toggled_key_genes: list[str] = []
        current_set = set(st.session_state["sim_genes"])
        for idx, (gene_name, role_desc) in enumerate(available_key_genes):
            with cols[idx % 4]:
                checked = st.checkbox(
                    shorten_gene_name(gene_name, 28),
                    value=(gene_name in current_set),
                    help=f"{gene_name} — {role_desc}",
                    key=f"chk_{gene_name}",
                )
                if checked:
                    toggled_key_genes.append(gene_name)

        extra_options = [g for g in feature_cols if g not in {x[0] for x in available_key_genes}]
        extra_selected = st.multiselect(
            "Add any additional genes from the 702 curated AMR features:",
            options=extra_options,
            default=[g for g in st.session_state["sim_genes"] if g in extra_options],
        )

        active_genes_list = sorted(set(toggled_key_genes + extra_selected))
        st.session_state["sim_genes"] = active_genes_list

        vec = np.zeros(len(feature_cols), dtype=np.float32)
        col_to_idx = {c: i for i, c in enumerate(feature_cols)}
        for g in active_genes_list:
            vec[col_to_idx[g]] = 1.0
        x_input = pd.DataFrame([vec], columns=feature_cols, dtype=np.float32)
        current_isolate_id = "custom_simulated_profile"
        isolate_header = f"**Custom Simulated Isolate** ({len(active_genes_list)} active AMR genes)"

    else:
        # Mode 3: Upload Gene Call Table (CSV/TSV)
        st.markdown("#### Upload Gene Call Table (CARD RGI, AMRFinderPlus, or Gene List CSV/TSV)")
        up_col, dl_col = st.columns([2.2, 1.0])
        with up_col:
            uploaded_file = st.file_uploader(
                "Upload CSV or TSV containing detected resistance genes",
                type=["csv", "tsv", "txt"],
                help="Supports CARD RGI (`Best_Hit_ARO`), AMRFinderPlus (`Gene symbol` / `Element symbol`), or any CSV/TSV with a gene column (and optional `isolate_id` for batch mode).",
            )
        with dl_col:
            st.markdown("<div style='height:1.7rem;'></div>", unsafe_allow_html=True)
            st.download_button(
                label="⬇️ Download Sample Input CSV",
                data=build_sample_upload_csv(),
                file_name="sample_amr_gene_calls.csv",
                mime="text/csv",
                width="stretch",
            )

        if uploaded_file is None:
            st.info(
                "Upload a CARD RGI, AMRFinderPlus, or custom gene-call CSV/TSV above — or click "
                "**Download Sample Input CSV** to test the upload & batch prediction workflow."
            )
            return

        sep = "\t" if uploaded_file.name.lower().endswith((".tsv", ".txt")) else ","
        try:
            up_df = pd.read_csv(uploaded_file, sep=sep)
        except Exception as exc:
            st.error(f"Could not parse uploaded file: {exc}")
            return

        if up_df.empty:
            st.warning("Uploaded file contains no rows.")
            return

        auto_gene_col, auto_iso_col = detect_upload_columns(up_df)
        c_sel1, c_sel2 = st.columns(2)
        with c_sel1:
            gene_col = st.selectbox(
                "Gene Name Column",
                options=list(up_df.columns),
                index=list(up_df.columns).index(auto_gene_col),
            )
        with c_sel2:
            iso_options = ["(Single Isolate — All Rows)"] + [
                c for c in up_df.columns if c != gene_col
            ]
            default_iso_idx = (
                iso_options.index(auto_iso_col) if auto_iso_col in iso_options else 0
            )
            iso_choice = st.selectbox(
                "Isolate ID Column (optional, for batch tables)",
                options=iso_options,
                index=default_iso_idx,
            )

        col_to_idx = {c: i for i, c in enumerate(feature_cols)}

        if iso_choice != "(Single Isolate — All Rows)":
            isolate_groups = {
                str(iso): grp[gene_col].dropna().astype(str).tolist()
                for iso, grp in up_df.groupby(iso_choice, sort=True)
            }
            isolate_ids = list(isolate_groups.keys())

            if len(isolate_ids) > 1:
                # Batch summary table across all uploaded isolates
                batch_rows = []
                for iso_id in isolate_ids:
                    matched_g, _ = map_uploaded_genes(isolate_groups[iso_id], feature_cols)
                    vec_b = np.zeros(len(feature_cols), dtype=np.float32)
                    for g in matched_g:
                        vec_b[col_to_idx[g]] = 1.0
                    xb = pd.DataFrame([vec_b], columns=feature_cols, dtype=np.float32)
                    row_summary: dict[str, object] = {
                        "Isolate ID": iso_id,
                        "Matched Genes": len(matched_g),
                    }
                    for d in CORE_ANTIBIOTICS:
                        p = float(models[d]["model"].predict_proba(xb)[0, 1])
                        lbl, _, _ = classify_confidence(p)
                        row_summary[DRUG_DISPLAY[d][0]] = f"{lbl} ({p * 100:.0f}%)"
                    batch_rows.append(row_summary)

                st.markdown(f"##### Batch Summary ({len(isolate_ids)} Uploaded Isolates)")
                st.dataframe(pd.DataFrame(batch_rows), width="stretch", hide_index=True)

            current_isolate_id = st.selectbox(
                "Inspect isolate in Antibiogram & SHAP view:",
                options=isolate_ids,
            )
            raw_gene_list = isolate_groups[current_isolate_id]
        else:
            current_isolate_id = Path(uploaded_file.name).stem
            raw_gene_list = up_df[gene_col].dropna().astype(str).tolist()

        matched_genes, unmatched_genes = map_uploaded_genes(raw_gene_list, feature_cols)
        active_genes_list = matched_genes

        m1, m2, m3 = st.columns(3)
        with m1:
            render_kpi("Uploaded Gene Calls", f"{len(set(raw_gene_list))}", f"Isolate: {current_isolate_id}")
        with m2:
            render_kpi("Matched Model Features", f"{len(matched_genes)}", "Case-insensitive + bla-prefix aware")
        with m3:
            render_kpi("Unrecognized Genes", f"{len(unmatched_genes)}", "Not in 702 trained features")

        if matched_genes:
            with st.expander(f"Matched resistance genes ({len(matched_genes)})"):
                st.write(", ".join(f"`{g}`" for g in matched_genes))
        if unmatched_genes:
            with st.expander(f"Unmatched gene calls ({len(unmatched_genes)})"):
                st.write(", ".join(f"`{g}`" for g in unmatched_genes))

        vec = np.zeros(len(feature_cols), dtype=np.float32)
        for g in matched_genes:
            vec[col_to_idx[g]] = 1.0
        x_input = pd.DataFrame([vec], columns=feature_cols, dtype=np.float32)
        isolate_header = f"**Uploaded Isolate `{current_isolate_id}`** ({len(matched_genes)} matched AMR genes)"

    if x_input is None:
        return

    st.markdown("---")
    st.markdown(f"### Predicted In-Silico Antibiogram — {isolate_header}")

    # Run predictions for all 5 drugs
    pred_records: list[dict[str, object]] = []
    for drug in CORE_ANTIBIOTICS:
        model = models[drug]["model"]
        prob = float(model.predict_proba(x_input)[0, 1])
        pred_bin = 1 if prob >= 0.50 else 0
        conf_label, badge_cls, color = classify_confidence(prob)
        pred_records.append(
            {
                "drug": drug,
                "display": DRUG_DISPLAY[drug][0],
                "drug_class": DRUG_DISPLAY[drug][1],
                "prob": prob,
                "pred_bin": pred_bin,
                "conf_label": conf_label,
                "badge_cls": badge_cls,
                "color": color,
                "actual": actual_labels[drug],
            }
        )

    # Render 5 crisp antibiogram cards in a row
    card_cols = st.columns(5)
    for col, rec in zip(card_cols, pred_records):
        pct = rec["prob"] * 100.0
        if rec["actual"] is not None:
            actual_str = "Resistant" if rec["actual"] == 1 else "Susceptible"
            match_icon = "✅" if rec["actual"] == rec["pred_bin"] else "⚠️"
            lab_html = (
                f"<div style='font-size:0.78rem; margin-top:0.4rem; opacity:0.85;'>"
                f"Lab: <b>{actual_str}</b> {match_icon}</div>"
            )
        else:
            mode_tag = "Uploaded Profile" if mode.startswith("Upload") else "Simulated Profile"
            lab_html = f"<div style='font-size:0.78rem; margin-top:0.4rem; opacity:0.6;'>{mode_tag}</div>"

        with col:
            st.markdown(
                f"""
                <div class="antibiogram-card">
                    <div style="font-weight:700; font-size:1.0rem;">{rec['display']}</div>
                    <div style="font-size:0.74rem; opacity:0.7; margin-bottom:0.45rem;">{rec['drug_class']}</div>
                    <span class="badge {rec['badge_cls']}">{rec['conf_label']}</span>
                    <div style="font-size:1.4rem; font-weight:700; margin-top:0.45rem;">{pct:.1f}%</div>
                    <div class="prob-track">
                        <div style="width:{pct:.1f}%; height:100%; background:{rec['color']};"></div>
                    </div>
                    {lab_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Structured table view + Downloadable CSV Report
    table_rows = []
    for rec in pred_records:
        actual_text = (
            "—"
            if rec["actual"] is None
            else ("Resistant (1)" if rec["actual"] == 1 else "Susceptible (0)")
        )
        table_rows.append(
            {
                "Antibiotic": rec["display"],
                "Drug Class": rec["drug_class"],
                "Predicted Phenotype": rec["conf_label"],
                "Resistance Probability": f"{rec['prob'] * 100:.1f}%",
                "Actual Lab Phenotype": actual_text,
            }
        )
    st.dataframe(pd.DataFrame(table_rows), width="stretch", hide_index=True)

    export_df = build_antibiogram_export_df(
        isolate_id=current_isolate_id,
        pred_records=pred_records,
        explainers=explainers,
        x_input=x_input,
        feature_cols=feature_cols,
    )
    safe_iso_slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", current_isolate_id).strip("_") or "isolate"
    st.download_button(
        label="⬇️ Download Antibiogram Report (CSV)",
        data=export_df.to_csv(index=False),
        file_name=f"antibiogram_report_{safe_iso_slug}.csv",
        mime="text/csv",
    )

    # Interactive SHAP Explainer Section
    st.markdown("### Local SHAP Explanation (Gene-Level Attribution)")
    selected_drug = st.selectbox(
        "Select antibiotic to explain:",
        options=CORE_ANTIBIOTICS,
        format_func=lambda d: f"{DRUG_DISPLAY[d][0]} ({DRUG_DISPLAY[d][1]})",
    )

    base_val, shap_vals = compute_single_shap(explainers[selected_drug], x_input)
    pred_prob = float(models[selected_drug]["model"].predict_proba(x_input)[0, 1])

    shap_df = pd.DataFrame(
        {
            "gene": feature_cols,
            "present": x_input.iloc[0].to_numpy().astype(int),
            "shap_value": shap_vals,
            "abs_shap": np.abs(shap_vals),
        }
    ).sort_values("abs_shap", ascending=False)

    top_n = st.slider("Number of top contributing genes to display", 5, 20, 12)
    top_df = shap_df.head(top_n).iloc[::-1].copy()
    top_df["label"] = top_df.apply(
        lambda r: f"{shorten_gene_name(r['gene'], 42)} [{'Present' if r['present'] == 1 else 'Absent'}]",
        axis=1,
    )
    top_df["color"] = np.where(top_df["shap_value"] >= 0, "#ef4444", "#10b981")

    st.markdown(
        f"**SHAP Feature Contributions for {DRUG_DISPLAY[selected_drug][0]}** "
        f"(Predicted Resistance Probability: **{pred_prob * 100:.1f}%** | "
        f"Base Log-Odds: `{base_val:+.2f}`)"
    )

    min_val = min(float(top_df["shap_value"].min()), -0.1)
    max_val = max(float(top_df["shap_value"].max()), 0.1)

    fig = go.Figure(
        go.Bar(
            x=top_df["shap_value"],
            y=top_df["label"],
            orientation="h",
            marker_color=top_df["color"],
            text=[f"{v:+.3f}" for v in top_df["shap_value"]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Status: %{customdata[1]}<br>"
                "SHAP Contribution: %{x:+.4f} log-odds<extra></extra>"
            ),
            customdata=np.column_stack(
                [
                    top_df["gene"],
                    np.where(top_df["present"] == 1, "Present (1)", "Absent (0)"),
                ]
            ),
        )
    )
    fig.add_vline(x=0, line_width=1.2, line_color="rgba(148,163,184,0.6)")
    fig.update_layout(
        xaxis=dict(
            range=[min_val * 1.25, max_val * 1.25],
            title="SHAP Value (Log-Odds Impact: + pushes Resistant, − pushes Susceptible)",
        ),
        yaxis_title="",
        height=max(390, top_n * 30 + 95),
        margin=dict(l=220, r=70, t=70, b=45),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


# ---------------------------------------------------------------------------
# View 2: AMR Surveillance Dashboard (Population Epidemiology)
# ---------------------------------------------------------------------------
def render_surveillance_view(labels: pd.DataFrame, surv: pd.DataFrame) -> None:
    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-title">🌍 AMR Surveillance Dashboard (Population Epidemiology)</div>
            <p class="hero-subtitle">
                Laboratory-confirmed resistance prevalence, multidrug co-resistance patterns, and
                global temporal & geographic trends across BV-BRC <i>Klebsiella pneumoniae</i> isolates.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = labels.merge(surv, on="genome_id", how="left")
    n_isolates = len(df)
    n_countries = int(df["isolation_country"].dropna().nunique()) if "isolation_country" in df else 0
    valid_years = df["collection_year"].dropna()
    year_span = (
        f"{int(valid_years.min())}–{int(valid_years.max())}" if not valid_years.empty else "N/A"
    )
    mdr_mask = df[CORE_ANTIBIOTICS].fillna(0).sum(axis=1) >= 3
    mdr_pct = 100.0 * mdr_mask.mean()

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi("Lab-Verified Isolates", f"{n_isolates:,}", "BV-BRC Taxon 573 (Laboratory Method)")
    with k2:
        render_kpi("Countries Represented", f"{n_countries:,}", "Unique isolation countries")
    with k3:
        render_kpi("Collection Year Range", year_span, f"{len(valid_years):,} isolates with year")
    with k4:
        render_kpi("MDR Rate (≥3 Drugs)", f"{mdr_pct:.1f}%", f"{int(mdr_mask.sum()):,} multidrug-resistant isolates")

    st.markdown("---")
    left_col, right_col = st.columns([1.05, 0.95])

    # 1. Resistance Prevalence Bar Chart
    with left_col:
        prev_rows = []
        for drug in CORE_ANTIBIOTICS:
            s = df[drug].dropna()
            r_cnt = int((s == 1).sum())
            s_cnt = int((s == 0).sum())
            tot = r_cnt + s_cnt
            prev_rows.append(
                {
                    "Antibiotic": DRUG_DISPLAY[drug][0],
                    "Resistant (%)": 100.0 * r_cnt / tot if tot else 0.0,
                    "Susceptible (%)": 100.0 * s_cnt / tot if tot else 0.0,
                    "Resistant N": r_cnt,
                    "Susceptible N": s_cnt,
                    "Tested N": tot,
                }
            )
        prev_df = pd.DataFrame(prev_rows)

        fig_prev = go.Figure()
        fig_prev.add_trace(
            go.Bar(
                name="Resistant (1)",
                x=prev_df["Antibiotic"],
                y=prev_df["Resistant (%)"],
                marker_color="#ef4444",
                text=[f"{v:.1f}%" for v in prev_df["Resistant (%)"]],
                textposition="inside",
                customdata=np.column_stack([prev_df["Resistant N"], prev_df["Tested N"]]),
                hovertemplate="<b>%{x}</b><br>Resistant: %{y:.1f}% (%{customdata[0]:,} / %{customdata[1]:,})<extra></extra>",
            )
        )
        fig_prev.add_trace(
            go.Bar(
                name="Susceptible (0)",
                x=prev_df["Antibiotic"],
                y=prev_df["Susceptible (%)"],
                marker_color="#10b981",
                text=[f"{v:.1f}%" for v in prev_df["Susceptible (%)"]],
                textposition="inside",
                customdata=np.column_stack([prev_df["Susceptible N"], prev_df["Tested N"]]),
                hovertemplate="<b>%{x}</b><br>Susceptible: %{y:.1f}% (%{customdata[0]:,} / %{customdata[1]:,})<extra></extra>",
            )
        )
        fig_prev.update_layout(
            barmode="stack",
            title="Resistance Prevalence Across Core Antibiotics",
            yaxis_title="Percentage of Tested Isolates (%)",
            yaxis_range=[0, 100],
            height=420,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_prev, width="stretch")

    # 2. MDR Co-occurrence Heatmap
    with right_col:
        drug_names = [DRUG_DISPLAY[d][0] for d in CORE_ANTIBIOTICS]
        co_mat = np.zeros((len(CORE_ANTIBIOTICS), len(CORE_ANTIBIOTICS)))
        n_mat = np.zeros((len(CORE_ANTIBIOTICS), len(CORE_ANTIBIOTICS)), dtype=int)
        for i, d1 in enumerate(CORE_ANTIBIOTICS):
            for j, d2 in enumerate(CORE_ANTIBIOTICS):
                if i == j:
                    n_r = int((df[d1] == 1).sum())
                    n_mat[i, j] = n_r
                    co_mat[i, j] = 100.0 if n_r > 0 else 0.0
                else:
                    pair = df[[d1, d2]].dropna()
                    r1 = pair[pair[d1] == 1]
                    n_mat[i, j] = len(r1)
                    co_mat[i, j] = float(100.0 * (r1[d2] == 1).mean()) if len(r1) > 0 else 0.0

        fig_heat = go.Figure(
            data=go.Heatmap(
                z=co_mat,
                x=drug_names,
                y=drug_names,
                colorscale="YlOrRd",
                zmin=0,
                zmax=100,
                text=[[f"{co_mat[i, j]:.1f}%" for j in range(5)] for i in range(5)],
                texttemplate="%{text}",
                customdata=n_mat,
                hovertemplate=(
                    "Given Resistant to <b>%{y}</b>:<br>"
                    "Co-resistant to <b>%{x}</b>: <b>%{z:.1f}%</b><br>"
                    "(n = %{customdata:,} isolates)<extra></extra>"
                ),
            )
        )
        fig_heat.update_layout(
            title="Conditional Co-Resistance Heatmap: P(Col R | Row R)",
            xaxis_title="Co-Resistant Drug",
            yaxis_title="Index Resistant Drug",
            yaxis_autorange="reversed",
            height=420,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_heat, width="stretch")

    # 3. Temporal & Geographic Trends
    st.markdown("---")
    st.markdown("### Temporal & Geographic Surveillance Trends")

    ctrl1, ctrl2 = st.columns([1.2, 1.0])
    with ctrl1:
        drug_filter = st.selectbox(
            "Filter by Antibiotic",
            options=["All 5 Core Drugs", *CORE_ANTIBIOTICS],
            format_func=lambda x: x if x == "All 5 Core Drugs" else f"{DRUG_DISPLAY[x][0]} ({DRUG_DISPLAY[x][1]})",
        )
    with ctrl2:
        min_n_group = st.slider(
            "Minimum tested isolates per year / country",
            min_value=5,
            max_value=50,
            value=15,
            step=5,
        )

    t_col, g_col = st.columns(2)

    # Temporal Chart
    with t_col:
        yr_df = df.dropna(subset=["collection_year"]).copy()
        yr_df["collection_year"] = yr_df["collection_year"].astype(int)
        yr_df = yr_df[(yr_df["collection_year"] >= 2000) & (yr_df["collection_year"] <= 2026)]

        drugs_to_plot = CORE_ANTIBIOTICS if drug_filter == "All 5 Core Drugs" else [drug_filter]
        fig_time = go.Figure()
        palette = ["#ef4444", "#f59e0b", "#3b82f6", "#10b981", "#8b5cf6"]

        for idx, d in enumerate(drugs_to_plot):
            sub = yr_df.dropna(subset=[d])
            agg = (
                sub.groupby("collection_year")[d]
                .agg(tested="count", resistant="sum")
                .reset_index()
            )
            agg = agg[agg["tested"] >= min_n_group]
            agg["rate"] = 100.0 * agg["resistant"] / agg["tested"]
            fig_time.add_trace(
                go.Scatter(
                    x=agg["collection_year"],
                    y=agg["rate"],
                    mode="lines+markers",
                    name=DRUG_DISPLAY[d][0],
                    line=dict(width=2.5, color=palette[idx % len(palette)]),
                    customdata=np.column_stack([agg["resistant"], agg["tested"]]),
                    hovertemplate=(
                        f"<b>{DRUG_DISPLAY[d][0]}</b> (%{{x}})<br>"
                        "Resistance Rate: %{y:.1f}%<br>"
                        "Resistant: %{customdata[0]:.0f} / %{customdata[1]:.0f}<extra></extra>"
                    ),
                )
            )

        fig_time.update_layout(
            title="Resistance Rate Trend Over Collection Years",
            xaxis_title="Collection Year",
            yaxis_title="Resistance Rate (%)",
            yaxis_range=[0, 105],
            height=430,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_time, width="stretch")

    # Geographic Chart
    with g_col:
        geo_drug = "meropenem" if drug_filter == "All 5 Core Drugs" else drug_filter
        geo_sub = df.dropna(subset=["isolation_country", geo_drug]).copy()
        geo_agg = (
            geo_sub.groupby("isolation_country")[geo_drug]
            .agg(tested="count", resistant="sum")
            .reset_index()
        )
        geo_agg = geo_agg[geo_agg["tested"] >= min_n_group]
        geo_agg["rate"] = 100.0 * geo_agg["resistant"] / geo_agg["tested"]
        geo_top = geo_agg.sort_values("tested", ascending=False).head(12).iloc[::-1]

        fig_geo = px.bar(
            geo_top,
            x="tested",
            y="isolation_country",
            orientation="h",
            color="rate",
            color_continuous_scale="RdYlGn_r",
            range_color=[0, 100],
            text=[f"{r:.1f}% R (n={n:,})" for r, n in zip(geo_top["rate"], geo_top["tested"])],
            labels={
                "tested": "Tested Isolates",
                "isolation_country": "",
                "rate": "Resistance %",
            },
            title=f"Top Isolation Countries — {DRUG_DISPLAY[geo_drug][0]} Resistance Rate",
        )
        max_count = int(geo_top["tested"].max()) if not geo_top.empty else 100
        fig_geo.update_traces(textposition="outside", cliponaxis=False)
        fig_geo.update_layout(
            xaxis=dict(range=[0, max_count * 1.45]),
            height=430,
            margin=dict(r=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_geo, width="stretch")


# ---------------------------------------------------------------------------
# View 3: Model Performance & Biological Validation
# ---------------------------------------------------------------------------
def render_performance_view(n_features: int) -> None:
    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-title">📊 Model Performance & Biological Validation</div>
            <p class="hero-subtitle">
                Held-out 20% stratified test metrics for all 5 XGBoost classifiers and global
                SHAP biological validation against known clinical AMR mechanisms.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metrics_df, shap_tables = parse_model_evaluation_md()

    mean_auc = metrics_df["ROC-AUC"].mean()
    mean_f1 = metrics_df["F1-Score"].mean()
    best_row = metrics_df.loc[metrics_df["ROC-AUC"].idxmax()]

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi("Mean ROC-AUC", f"{mean_auc:.4f}", "Across 5 core antibiotics")
    with k2:
        render_kpi("Mean F1-Score", f"{mean_f1:.4f}", "Class-weighted XGBoost")
    with k3:
        render_kpi(
            "Top Performing Drug",
            DRUG_DISPLAY[best_row["antibiotic"]][0],
            f"ROC-AUC {best_row['ROC-AUC']:.4f} | F1 {best_row['F1-Score']:.4f}",
        )
    with k4:
        render_kpi("Curated Gene Features", f"{n_features:,}", "CARD + NDARO + PATRIC")

    st.markdown("---")

    # Comparative Plotly bar chart
    plot_df = metrics_df.copy()
    plot_df["Drug"] = plot_df["antibiotic"].map(lambda d: DRUG_DISPLAY[d][0])
    melted = plot_df.melt(
        id_vars=["Drug"],
        value_vars=["ROC-AUC", "F1-Score", "Precision", "Recall"],
        var_name="Metric",
        value_name="Score",
    )

    fig_perf = px.bar(
        melted,
        x="Drug",
        y="Score",
        color="Metric",
        barmode="group",
        text_auto=".3f",
        color_discrete_sequence=["#2563eb", "#10b981", "#f59e0b", "#8b5cf6"],
        title="Held-Out Test Set Performance by Antibiotic (80/20 Stratified Split)",
    )
    fig_perf.update_traces(textposition="outside", cliponaxis=False)
    fig_perf.update_layout(
        yaxis=dict(range=[0.70, 1.04], title="Score"),
        xaxis_title="",
        height=420,
        margin=dict(t=65, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_perf, width="stretch")

    st.markdown("#### Detailed Evaluation Metrics Table")
    display_metrics = metrics_df.copy()
    display_metrics["antibiotic"] = display_metrics["antibiotic"].map(lambda d: DRUG_DISPLAY[d][0])
    display_metrics = display_metrics.rename(
        columns={
            "antibiotic": "Antibiotic",
            "n_total": "Cohort N",
            "n_train": "Train N",
            "n_test": "Test N",
            "test_r": "Test R",
            "test_s": "Test S",
            "scale_pos_weight": "scale_pos_weight",
        }
    )
    st.dataframe(display_metrics, width="stretch", hide_index=True)

    st.markdown("---")
    st.markdown("### Global Top 5 SHAP Genes & Clinical Biological Validation")

    tabs = st.tabs([DRUG_DISPLAY[d][0] for d in CORE_ANTIBIOTICS])
    for tab, drug in zip(tabs, CORE_ANTIBIOTICS):
        with tab:
            tdf = shap_tables.get(drug, pd.DataFrame())
            if not tdf.empty:
                c_chart, c_tbl = st.columns([1.0, 1.15])
                with c_chart:
                    plot_t = tdf.iloc[::-1].copy()
                    plot_t["Short Gene"] = plot_t["Resistance Gene"].map(
                        lambda g: shorten_gene_name(g, 34)
                    )
                    colors = [
                        "#ef4444" if "Resistant" in d else "#10b981"
                        for d in plot_t["Effect Direction"]
                    ]
                    fig_g = go.Figure(
                        go.Bar(
                            x=plot_t["Mean |SHAP|"],
                            y=plot_t["Short Gene"],
                            orientation="h",
                            marker_color=colors,
                            text=[f"{v:.3f}" for v in plot_t["Mean |SHAP|"]],
                            textposition="outside",
                            customdata=plot_t["Resistance Gene"],
                            hovertemplate="<b>%{customdata}</b><br>Mean |SHAP|: %{x:.4f}<extra></extra>",
                        )
                    )
                    fig_g.update_layout(
                        title=f"Top 5 Global SHAP Drivers — {DRUG_DISPLAY[drug][0]}",
                        xaxis_title="Mean |SHAP| Magnitude",
                        yaxis_title="",
                        height=300,
                        margin=dict(l=10, r=50, t=45, b=35),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig_g, width="stretch")
                with c_tbl:
                    st.dataframe(tdf, width="stretch", hide_index=True)

                st.markdown(
                    f"""
                    <div class="bio-callout">
                        <b>Clinical & Biological Validation ({DRUG_DISPLAY[drug][0]}):</b><br/>
                        {BIOLOGICAL_CALLOUTS[drug]}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# ---------------------------------------------------------------------------
# Main Application Entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    labels, genes, surv = load_datasets()
    models, explainers, feature_cols = load_models_and_explainers()

    with st.sidebar:
        st.markdown("## 🧬 AMR Genomics")
        st.caption("*Klebsiella pneumoniae* (Taxon 573)")
        st.markdown("---")

        view = st.radio(
            "Dashboard Navigation",
            options=[
                "🧬 In-Silico Antibiogram & SHAP",
                "🌍 AMR Surveillance Dashboard",
                "📊 Model Performance & Validation",
            ],
            index=0,
        )

        st.markdown("---")
        st.markdown("##### Dataset & Model Summary")
        st.markdown(
            f"""
            <div class="sidebar-stat">
                <b>{len(labels):,}</b> lab-verified isolates<br/>
                <span style="opacity:0.72;">BV-BRC Laboratory Method only</span>
            </div>
            <div class="sidebar-stat">
                <b>5</b> core antibiotics modeled<br/>
                <span style="opacity:0.72;">MEM · CRO · CIP · GEN · SXT</span>
            </div>
            <div class="sidebar-stat">
                <b>{len(feature_cols):,}</b> curated AMR genes<br/>
                <span style="opacity:0.72;">CARD · NDARO · PATRIC</span>
            </div>
            <div class="sidebar-stat">
                <b>{len(genes):,}</b> MVP cohort genomes<br/>
                <span style="opacity:0.72;">XGBoost + TreeSHAP explainers</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if view.startswith("🧬"):
        render_antibiogram_view(labels, genes, surv, models, explainers, feature_cols)
    elif view.startswith("🌍"):
        render_surveillance_view(labels, surv)
    else:
        render_performance_view(len(feature_cols))


if __name__ == "__main__":
    main()
