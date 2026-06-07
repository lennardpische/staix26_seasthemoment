# Pipeline 2 — Transformer / Neural Expert
### STAI-X Challenge 2026 · Mixture-of-Experts Architecture

---

## 1. Overview

The full solution uses a **Mixture-of-Experts (MoE)** ensemble of four parallel prediction pipelines, each contributing one expert prediction file. The final submission is a learned (or tuned) weighted average of the four expert files.

| Expert | Owner | Approach | Output file |
|---|---|---|---|
| 1 | Jasmine | Healthcare-informed pipeline — public-health reasoning, state/time patterns, overdose surveillance interpretation, robust models | `expert_jasmine.csv` |
| **2** | **Lenny** | **Transformer / neural regression — FT-Transformer with regression head** | **`expert_transformer.csv`** |
| 3 | William | Classical statistics pipeline — lag models, rolling averages, fixed effects, ridge/lasso, time-series baselines | `expert_william.csv` |
| 4 | Eddy | Tree-based ML pipeline — Random Forest, XGBoost, LightGBM, CatBoost with feature engineering | `expert_eddy.csv` |

This document specifies **Expert 2** end-to-end: inputs, feature engineering, model architecture, and validation protocol.

The pipeline is intentionally **multimodal**: it fuses tabular covariates, free-text DOH press releases, and optional MAT-density image embeddings through a shared attention layer before a per-category regression head.

---

## 2. Data Sources + Targets

| File | Split | Role |
|---|---|---|
| `train/dose_sys_train.csv` | Train | Labels — `rate_per_10000_ed_visits` per (period_id, jurisdiction, overdose_category) |
| `train/covariates.csv` | Train | 11 numeric/text covariate columns keyed on (period_id, jurisdiction) |
| `val/covariates.csv` | Val | Same schema, target hidden |
| `train/images/mat_density/{ST}_{PERIOD_ID}.png` | Train | 256×256 viridis heatmap of within-state MAT prescriber density |
| `val/images/mat_density/{ST}_{PERIOD_ID}.png` | Val | Same, for validation periods |
| `sample_submission.csv` | — | Template — 918 rows (6 periods × 51 jurisdictions × 3 categories) |

**Target:** `rate_per_10000_ed_visits` (float, ≥ 0). Regression task.
**Scoring metric:** block-averaged MAE — mean of per-category MAEs across `all_drugs`, `all_opioids`, `all_stimulants`.
**Key join:** `(period_id, jurisdiction)` links covariates ↔ labels ↔ images; the third key `overdose_category` splits the target into three parallel prediction tasks.

---

## 3. Data Facts Driving the Design

| Fact | Implication |
|---|---|
| **Panel structure** — 51 jurisdictions × N periods | Positional encoding on temporal rank; cross-jurisdiction attention is not needed (independent rows) |
| **Three nested drug categories** — `all_drugs ⊃ all_opioids`, `all_stimulants` partially overlapping | Shared backbone; separate regression heads per category to capture distribution shift |
| **Free-text column** — `state_doh_release`, up to 500 tokens of DOH press-release language | Direct semantic signal unavailable to tree models; motivates a text encoder sub-tower |
| **MAT density images** — 256×256 PNG spatial patterns | Spatial prescriber distribution can proxy access barriers; motivates a lightweight vision encoder |
| **Google Trends × 5** — fentanyl, opioid, overdose, naloxone, methamphetamine | Pairwise products capture co-search dynamics (e.g. fentanyl + naloxone spike = harm-reduction awareness event) |
| **Target suppression** — `NaN` means cell is suppressed, not zero | Suppressed rows dropped from training; never imputed to zero |
| **Temporal ordering is recoverable** — cross-jurisdiction mean of `temp_avg_f` proxies calendar season | Enables lag features and period-rank positional encoding |
| **CPU-only environment** — no GPU | Model must fit in CPU memory; text/vision encoders limited to lightweight checkpoints or static embeddings |

---

## 4. Engineered Feature Set

All transforms are **fit on train, applied to val** with no leakage.

### 4a. Numeric tabular features

| Feature | Source | Transform |
|---|---|---|
| `unemployment_rate` | covariates | as-is |
| `log1p_labor_force` | `labor_force` | log1p |
| `temp_avg_f` | covariates | as-is (NaN → jurisdiction-period median from train) |
| `precip_in` | covariates | as-is |
| `gtrends_overdose`, `gtrends_fentanyl`, `gtrends_naloxone`, `gtrends_opioid`, `gtrends_methamphetamine` | covariates | as-is |
| `gt_fentanyl_x_opioid` | product | `gtrends_fentanyl × gtrends_opioid` |
| `gt_overdose_x_naloxone` | product | `gtrends_overdose × gtrends_naloxone` |
| `period_rank` | `temp_avg_f` cross-jurisdiction mean rank | integer temporal index |
| `rate_lag_1`, `rate_lag_2` | train target, per (jurisdiction, category) | previous period's rate; `NaN` for earliest periods → median impute |

All numeric NaN → training-set median (per column), applied after feature engineering.

### 4b. Text features — `state_doh_release`

Two options (select one at runtime based on available RAM):

| Mode | Method | Output dim |
|---|---|---|
| **Static (default)** | TF-IDF (5 000 vocab, sublinear_tf) → TruncatedSVD (32 components), fit on train | 32 |
| **Encoder (optional)** | `all-MiniLM-L6-v2` sentence-transformers (CPU, ~90 MB) → mean-pool | 384 |

Empty string → zero vector.

### 4c. Image features — MAT density

| Mode | Method | Output dim |
|---|---|---|
| **Static (default)** | Flatten 256×256×3 → PCA (64 components) fit on train images | 64 |
| **Encoder (optional)** | `timm` `efficientnet_lite0` (CPU, ~15 MB) — global average pool of last conv layer | 320 |

Missing image → zero vector.

### 4d. Final feature vector per row

```
[numeric (14 dims)] + [text (32 or 384 dims)] + [image (64 or 320 dims)]
→ concatenated and passed through a learned linear projection to d_model
```

---

## 5. Modeling Plan

### Architecture: FT-Transformer with modality projection

```
Input row
  ├─ Numeric block (14)  ──┐
  ├─ Text block  (32/384) ──┼──► Linear projection → d_model=128, per modality token
  └─ Image block (64/320) ──┘

  [3 tokens] → Transformer encoder (2 layers, 4 heads, d_ff=256)
             → [CLS] token pooling
             → Dropout(0.1)
             → Category-specific linear head × 3
                 ├─ head_all_drugs      → scalar
                 ├─ head_all_opioids    → scalar
                 └─ head_all_stimulants → scalar
```

**Loss:** Huber loss (delta=1.0) per head, summed. Huber is less sensitive to suppression-boundary outliers than MSE.

**Optimizer:** AdamW, lr=3e-4, weight_decay=1e-4, cosine annealing over 100 epochs.

**Batch size:** 64 rows (shuffle each epoch within fold).

**Training rule:** early stopping on fold-held-out MAE, patience=15 epochs.

**Implementation:** PyTorch (CPU). All matrix ops on float32.

### Fallback

If PyTorch is unavailable or training exceeds 90 minutes: fall back to an MLP (2 hidden layers, 256 units, ReLU, Dropout 0.1) trained with scikit-learn's `MLPRegressor` on the same concatenated feature vector. One model per category.

---

## 6. Validation Plan

```
GroupKFold(n_splits=5)
  groups = period_rank   (temporal group — prevents future-period leakage)
```

For each fold:
1. Fit all feature transforms (TF-IDF/PCA/encoder) on train split only.
2. Train the transformer model on train split.
3. Generate OOF predictions on held-out split.

**Metrics reported:**
- OOF MAE per category (`all_drugs`, `all_opioids`, `all_stimulants`)
- Block-averaged OOF MAE (competition metric)
- Per-fold MAE to detect temporal drift

**Output artefact:**

```python
expert_transformer = template[['row_id']].copy()
expert_transformer['rate_per_10000_ed_visits'] = val_preds.clip(lower=0)
assert expert_transformer.isna().sum().sum() == 0
expert_transformer.to_csv('expert_transformer.csv', index=False)
```

This file is passed to the MoE combiner alongside the other three expert files.

---

## 7. Main Takeaway

The transformer expert is the **multimodal, semantics-aware** leg of the ensemble. Its primary contribution over gradient-boosted trees is:

1. **Text semantics** — the `state_doh_release` column contains unstructured policy language (new naloxone distribution programs, fentanyl test strip legalization, etc.) that TF-IDF alone underweights; a learned text projection recovers these signals.
2. **Spatial prescriber patterns** — the MAT density heatmaps encode geographic access barriers that are invisible in the scalar covariates.
3. **Cross-modality attention** — joint attention over tabular, text, and image tokens allows the model to learn interactions (e.g., low MAT density + high fentanyl search interest → elevated stimulant/opioid co-use rate).

Expected OOF MAE target: within 10–15% of the GBM baseline on its own; the MoE combination should beat both individually via diversity.
