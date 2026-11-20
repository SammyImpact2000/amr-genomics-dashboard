# Phase 2 Model Evaluation & SHAP Explainability Report

- **Cohort size (merged genomes):** 500
- **AMR gene features:** 702 (binary presence/absence from BV-BRC Specialty Genes: CARD, NDARO, PATRIC)
- **Model:** `XGBClassifier` (`n_estimators=200`, `max_depth=4`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, class-balanced via `scale_pos_weight`)
- **Validation:** 80/20 stratified train/test split (`random_state=42`)

## Per-Drug Classification Performance (Held-Out 20% Test Set)

| Antibiotic | N Total | Train / Test | Test R / S | `scale_pos_weight` | ROC-AUC | F1-Score | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `meropenem` | 500 | 400 / 100 | 30 / 70 | 2.31 | 0.9538 | 0.8421 | 0.8889 | 0.8000 |
| `ceftriaxone` | 500 | 400 / 100 | 74 / 26 | 0.36 | 0.9844 | 0.9583 | 0.9857 | 0.9324 |
| `ciprofloxacin` | 500 | 400 / 100 | 60 / 40 | 0.66 | 0.9850 | 0.9402 | 0.9649 | 0.9167 |
| `gentamicin` | 500 | 400 / 100 | 30 / 70 | 2.31 | 0.9619 | 0.8889 | 0.8485 | 0.9333 |
| `trimethoprim-sulfamethoxazole` | 500 | 400 / 100 | 56 / 44 | 0.77 | 0.9091 | 0.8889 | 0.8525 | 0.9286 |

## Top 5 SHAP Feature-Importance Genes per Antibiotic

### `meropenem`

| Rank | Resistance Gene | Mean \|SHAP\| | Mean SHAP (Gene=1) | Effect Direction | Test Presence |
|---:|---|---:|---:|---|---:|
| 1 | `Escherichia coli parC conferring resistance to fluoroquinolones` | 1.0502 | +0.7856 | Resistant (+) | 43/100 |
| 2 | `KPC-2` | 0.8425 | +2.1052 | Resistant (+) | 20/100 |
| 3 | `OXA-9` | 0.4840 | +0.9426 | Resistant (+) | 17/100 |
| 4 | `OXA-48 family` | 0.4110 | +1.6769 | Resistant (+) | 3/100 |
| 5 | `BRP(MBL)` | 0.4086 | +2.0759 | Resistant (+) | 8/100 |

### `ceftriaxone`

| Rank | Resistance Gene | Mean \|SHAP\| | Mean SHAP (Gene=1) | Effect Direction | Test Presence |
|---:|---|---:|---:|---|---:|
| 1 | `CTX-M-15` | 1.8715 | +3.2138 | Resistant (+) | 40/100 |
| 2 | `Escherichia coli parC conferring resistance to fluoroquinolones` | 1.1500 | +1.7053 | Resistant (+) | 49/100 |
| 3 | `AcrAB-TolC` | 0.3179 | -0.3603 | Susceptible (-) | 48/100 |
| 4 | `sul1` | 0.2698 | +0.3584 | Resistant (+) | 44/100 |
| 5 | `SHV-11` | 0.2577 | -0.4759 | Susceptible (-) | 31/100 |

### `ciprofloxacin`

| Rank | Resistance Gene | Mean \|SHAP\| | Mean SHAP (Gene=1) | Effect Direction | Test Presence |
|---:|---|---:|---:|---|---:|
| 1 | `Escherichia coli parC conferring resistance to fluoroquinolones` | 2.4020 | +3.4088 | Resistant (+) | 46/100 |
| 2 | `Salmonella isangi gyrA conferring resistance to fluoroquinolones` | 0.3989 | +1.1459 | Resistant (+) | 25/100 |
| 3 | `CTX-M-15` | 0.3661 | +0.4417 | Resistant (+) | 46/100 |
| 4 | `sul1` | 0.2428 | +0.3524 | Resistant (+) | 36/100 |
| 5 | `QnrS1` | 0.2333 | +0.8821 | Resistant (+) | 9/100 |

### `gentamicin`

| Rank | Resistance Gene | Mean \|SHAP\| | Mean SHAP (Gene=1) | Effect Direction | Test Presence |
|---:|---|---:|---:|---|---:|
| 1 | `AAC(3)-IIe` | 1.0532 | +3.7669 | Resistant (+) | 14/100 |
| 2 | `Escherichia coli parC conferring resistance to fluoroquinolones` | 0.8931 | +0.5519 | Resistant (+) | 42/100 |
| 3 | `AAC(3)-II,III,IV,VI,VIII,IX,X` | 0.4520 | +1.9541 | Resistant (+) | 8/100 |
| 4 | `TEM-1` | 0.3383 | +0.3741 | Resistant (+) | 33/100 |
| 5 | `catA1` | 0.2432 | +0.4096 | Resistant (+) | 11/100 |

### `trimethoprim-sulfamethoxazole`

| Rank | Resistance Gene | Mean \|SHAP\| | Mean SHAP (Gene=1) | Effect Direction | Test Presence |
|---:|---|---:|---:|---|---:|
| 1 | `qacEdelta1` | 1.1706 | +1.8467 | Resistant (+) | 40/100 |
| 2 | `sul2` | 0.8744 | +1.9554 | Resistant (+) | 27/100 |
| 3 | `sul1` | 0.4611 | +0.7336 | Resistant (+) | 42/100 |
| 4 | `Acinetobacter baumannii AbaF` | 0.4338 | +1.1954 | Resistant (+) | 15/100 |
| 5 | `aadA2` | 0.2979 | +0.8880 | Resistant (+) | 25/100 |
