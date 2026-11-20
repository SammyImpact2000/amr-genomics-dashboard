# 🧬 *Klebsiella pneumoniae* AMR Genomics & Explainable ML Dashboard

An end-to-end clinical genomics and explainable machine learning dashboard for predicting phenotypic antimicrobial resistance (AMR) in ***Klebsiella pneumoniae*** (NCBI Taxon ID `573`) directly from curated resistance gene profiles, paired with global population surveillance analytics.

---

## Clinical Motivation

Multidrug-resistant *Klebsiella pneumoniae* — particularly extended-spectrum β-lactamase (ESBL) and carbapenemase-producing strains — is a WHO critical-priority pathogen. Conventional phenotypic antimicrobial susceptibility testing (AST) can take 24–72 hours after culture isolation. By pairing whole-genome resistance gene calls with explainable gradient-boosted trees (**XGBoost + TreeSHAP**), this dashboard delivers:

1. **Rapid In-Silico Antibiograms:** Predicts Resistant vs. Susceptible phenotypes and calibrated resistance probabilities across 5 clinically actionable core antibiotics.
2. **Transparent Gene-Level Explanations:** Uses local SHAP values to highlight the exact resistance determinants (e.g., `KPC-2`, `OXA-48`, `CTX-M-15`, `parC`, `gyrA`, `AAC(3)-IIe`, `sul1`/`sul2`) driving every prediction.
3. **Interactive Gene-Table Upload & Export:** Accepts CARD RGI, AMRFinderPlus, or custom CSV/TSV gene call tables (single or batch isolates) and exports downloadable CSV antibiogram reports with top 3 SHAP drivers per drug.
4. **Population AMR Surveillance:** Visualizes resistance prevalence, multidrug co-resistance heatmaps, and temporal/geographic resistance trends across 6,915 lab-verified isolates.

---

## Data Provenance

- **Phenotypic AST Labels (`data/processed/amr_label_matrix.csv`):**
  - Sourced from the public **BV-BRC** `/api/genome_amr/` REST API for `taxon_id = 573`.
  - Strictly filtered to **`evidence = "Laboratory Method"`** (0 computational predictions retained).
  - Cleaned and deduplicated across **6,915 unique isolates** (33,127 clean isolate-drug labels; 0 conflicting pairs).
  - *Note:* `ampicillin` was excluded from ML modeling due to intrinsic chromosomal resistance in *K. pneumoniae* (99.57% resistant; 230.6:1 imbalance).
- **AMR Gene Features (`data/processed/gene_feature_matrix.csv`):**
  - Queried from BV-BRC `/api/sp_gene/` (`property = "Antibiotic Resistance"`) across curated databases (**CARD**, **NDARO**, **PATRIC**) for a balanced 500-genome MVP modeling cohort (all 500 genomes labeled across all 5 core drugs).
  - Yields **702 binary AMR gene features** (`1` = detected, `0` = absent).
- **Surveillance Metadata (`data/processed/genome_surveillance_metadata.csv`):**
  - Queried from BV-BRC `/api/genome/` for all 6,915 isolates (`isolation_country`, `collection_year` spanning 1996–2024, `host_name`, `isolation_source`).

---

## Model Performance (80/20 Stratified Test Split)

Trained one class-balanced `XGBClassifier` (`scale_pos_weight`) per core antibiotic (`models/xgb_*.joblib`). Full metrics and global SHAP attributions are stored in [`reports/model_evaluation.md`](reports/model_evaluation.md).

| Antibiotic | Drug Class | Test R / S | ROC-AUC | F1-Score | Precision | Recall | Top Biological SHAP Drivers |
|---|---|---:|---:|---:|---:|---:|---|
| **Meropenem** | Carbapenem | 30 / 70 | **0.9538** | 0.8421 | 0.8889 | 0.8000 | `KPC-2`, `OXA-48 family`, `BRP(MBL)`, `parC` |
| **Ceftriaxone** | 3rd-Gen Cephalosporin | 74 / 26 | **0.9844** | 0.9583 | 0.9857 | 0.9324 | `CTX-M-15`, `parC`, `sul1` |
| **Ciprofloxacin** | Fluoroquinolone | 60 / 40 | **0.9850** | 0.9402 | 0.9649 | 0.9167 | `parC` (QRDR), `gyrA` (QRDR), `QnrS1` |
| **Gentamicin** | Aminoglycoside | 30 / 70 | **0.9619** | 0.8889 | 0.8485 | 0.9333 | `AAC(3)-IIe`, `AAC(3)-II` family |
| **TMP-SMX** | Folate Pathway Inhibitor | 56 / 44 | **0.9091** | 0.8889 | 0.8525 | 0.9286 | `qacEdelta1`, `sul2`, `sul1` |

> **Clinical Note on Meropenem Recall (80.0%):** While carbapenemase-mediated resistance (`KPC-2`, `OXA-48`, `NDM`/MBL) is captured with high accuracy (ROC-AUC **0.9538**, Precision **88.9%**), a subset of carbapenem-resistant *K. pneumoniae* isolates lack classical carbapenemases and instead achieve resistance via **outer-membrane porin loss/truncation (`ompK35`/`ompK36`) combined with ESBL/AmpC overexpression**. Because binary specialty-gene presence/absence matrices do not capture frameshift or loss-of-function porin mutations, incorporating variant-level `ompK35`/`ompK36` features is a key next step to raise Meropenem recall above 90%.

---

## Repository Structure

```text
microbiology-amr-dashboard/
├── app.py                                      # Streamlit interactive dashboard
├── requirements.txt                            # Python dependencies
├── data/
│   └── processed/
│       ├── amr_label_matrix.csv                # Cleaned binary AST labels (6,915 isolates)
│       ├── gene_feature_matrix.csv             # Binary AMR gene matrix (500 x 702)
│       └── genome_surveillance_metadata.csv    # Country, year, host metadata (6,915 isolates)
├── models/                                     # Pre-trained XGBoost models + feature_columns.joblib
├── reports/
│   ├── class_balance_summary.csv               # Per-drug class balance table
│   └── model_evaluation.md                     # Held-out metrics & top 5 SHAP genes per drug
└── scripts/
    ├── 01_fetch_metadata.py                    # Fetch BV-BRC laboratory AST metadata
    ├── 02_clean_labels.py                      # Clean & pivot binary susceptibility matrix
    ├── 03_fetch_gene_features.py               # Fetch CARD/NDARO/PATRIC specialty genes
    ├── 04_train_models.py                      # Train XGBoost models & compute TreeSHAP
    └── 05_fetch_surveillance_metadata.py       # Fetch country/year surveillance metadata
```

---

## Local Setup & Run Instructions

```bash
# 1. Create and activate a Python 3.10+ virtual environment
python -m venv .venv
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the Streamlit dashboard
streamlit run app.py
```

The app runs directly out of the box using the committed `data/processed/`, `models/`, and `reports/` artifacts — ready for one-click deployment on **Streamlit Community Cloud** (`app.py` as the main file path).
