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

The pipeline is intentionally **multimodal**: it fuses tabular covariates, free-text DOH press releases, and MAT-density image embeddings through a shared transformer before per-category regression heads.

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
**Key join:** `(period_id, jurisdiction)` links covariates ↔ labels ↔ images; `overdose_category` splits the target into three parallel tasks handled by three heads.

---

## 3. Data Facts Driving the Design

| Fact | Implication |
|---|---|
| **Panel structure** — 51 jurisdictions × N periods | Period-rank positional signal; rows are independent across jurisdictions |
| **Three nested drug categories** — `all_drugs ⊃ all_opioids`, `all_stimulants` partially overlapping | Shared backbone; separate regression heads per category to capture distribution shift |
| **Free-text column** — `state_doh_release`, up to 500 tokens of DOH press-release language | Rich semantic signal unavailable to tree models; pre-trained language model captures policy language meaning |
| **MAT density images** — 256×256 PNG spatial heatmaps | Prescriber access patterns invisible in scalar covariates; vision encoder extracts spatial features |
| **Google Trends × 5** — fentanyl, opioid, overdose, naloxone, methamphetamine | Pairwise products capture co-search dynamics (e.g. fentanyl × naloxone spike = harm-reduction event) |
| **Target suppression** — `NaN` means suppressed, not zero | NaN cells masked from loss computation; never imputed to zero |
| **Temporal ordering recoverable** — cross-jurisdiction mean of `temp_avg_f` proxies calendar season | Enables lag features and period-rank positional signal; computed globally across train + val |
| **A100 GPU available** — Colab / Kaggle compute | Full pre-trained encoders viable; bf16 mixed precision; deep model (6 layers) within budget |

---

## 4. Engineered Feature Set

All transforms are **fit on train, applied to val** with no leakage.
Pre-computed embeddings are cached to `embeddings/` via `src/vectorize.py` and loaded by `src/features.py` at training time.

### 4a. Numeric tabular features

| Feature | Source | Transform |
|---|---|---|
| `unemployment_rate` | covariates | as-is |
| `log_labor_force` | `labor_force` | log1p |
| `temp_avg_f` | covariates | as-is |
| `precip_in` | covariates | as-is |
| `gtrends_overdose`, `gtrends_fentanyl`, `gtrends_naloxone`, `gtrends_opioid`, `gtrends_methamphetamine` | covariates | as-is |
| `gtrends_fentanyl_x_opioid` | product | `gtrends_fentanyl × gtrends_opioid` |
| `gtrends_overdose_x_naloxone` | product | `gtrends_overdose × gtrends_naloxone` |
| `census_division` | jurisdiction → int (1–9) | US Census division map |
| `period_rank` | `temp_avg_f` cross-jurisdiction mean rank | integer temporal index |
| `{cat}_lag1`, `{cat}_lag2` | train target per (jurisdiction, category) | backward-asof merge; NaN for earliest periods |
| `aux_{drug}` | aux overdose categories pivoted wide | heroin, fentanyl, cocaine, methamphetamine, benzodiazepine rates |

All numeric NaN → training-set median after feature engineering.

### 4b. Text features — `state_doh_release`

| Mode | Method | Dim | When used |
|---|---|---|---|
| **Pre-trained (default)** | `all-mpnet-base-v2` sentence-transformer, L2-normalised | **768** | `embeddings/text_*.csv` present |
| Fallback | TF-IDF (5 000 vocab, sublinear_tf, bigrams) → TruncatedSVD | 32 | Cache absent |

Run `python -m src.vectorize` once on A100 to populate the cache. Empty string → zero vector.

### 4c. Image features — MAT density

| Mode | Method | Dim | When used |
|---|---|---|---|
| **Pre-trained (default)** | `efficientnet_b2` (timm, ImageNet), global avg pool | **1408** | `embeddings/img_*.csv` present |
| Fallback | Resize 32×32 RGB → flatten → PCA (64 components, fit on train) | 64 | Cache absent |

Missing image → zero vector in both modes.

### 4d. Feature vector per row

```
Numeric  (~24 dims)  ──┐
Text     (768 dims)  ──┼──► separate learned projections → d_model = 256 each
Image    (1408 dims) ──┘
          ↓
[CLS, token_num, token_text, token_img]  — 4 tokens, each 256-dim
```

---

## 5. Modeling Plan

### Architecture: 6-layer FT-Transformer (A100 build)

```
Input row (period_id × jurisdiction)
  ├─ Numeric  (~24)  → Linear(n, 512) → GELU → Dropout → Linear(512, 256) → LayerNorm
  ├─ Text     (768)  → Linear(768, 512) → GELU → Dropout → Linear(512, 256) → LayerNorm
  └─ Image    (1408) → Linear(1408, 512) → GELU → Dropout → Linear(512, 256) → LayerNorm

[CLS(256), t_num(256), t_text(256), t_img(256)]  — 4 tokens
          ↓
  TransformerEncoder × 6 layers
    d_model=256, n_heads=8, d_ff=1024
    dropout=0.2, activation=GELU
    norm_first=True  ← pre-norm (more stable at depth ≥ 4)
          ↓
  CLS token output → LayerNorm → Dropout(0.2)
          ↓
  head_all_drugs      → scalar
  head_all_opioids    → scalar
  head_all_stimulants → scalar
```

**Loss:** Huber (δ=1.0), NaN-masked per cell. Suppressed rows contribute zero to the gradient.

**Optimizer:** AdamW, lr=3e-4, weight_decay=1e-3.

**Scheduler:** `CosineAnnealingWarmRestarts(T_0=100, T_mult=2)` — learning rate resets at epochs 100, 200, 400. Helps escape local minima on the small dataset.

**Precision:** bf16 mixed precision on CUDA (`torch.amp.autocast`), fp32 fallback on CPU.

**Batch size:** 128. **Max epochs:** 500. **Early stopping patience:** 50.

**Augmentation:** Gaussian noise N(0, 0.02) added to numeric features each training batch only.

**Compute:** ~20–30 min for 5-fold CV on A100. Well within 9-hour budget.

### Fallback

If PyTorch is unavailable or the 90-minute wall-clock limit is hit: sklearn `MLPRegressor` (2 × 256 hidden units) trained per category on the concatenated feature vector.

---

## 6. Validation Plan

```
GroupKFold(n_splits=5)
  groups = period_rank   (temporal — prevents future-period leakage)
```

For each fold:
1. Feature transforms (TF-IDF/PCA if no cache) fit on train split only.
2. Pre-computed embeddings loaded from cache and index-aligned on (period_id, jurisdiction).
3. Model trained on train split; early stopping on held-out Huber loss.
4. OOF predictions collected for va_idx rows.

**Metrics reported:**
- OOF MAE per category (`all_drugs`, `all_opioids`, `all_stimulants`)
- Block-averaged OOF MAE (competition metric)
- Total wall-clock time

**Output artefact:**

```python
# written by src/predict.py
expert_transformer.csv   — row_id, rate_per_10000_ed_visits (918 rows, 0 NaN)
```

Passed to the MoE combiner alongside the other three expert files.

---

## 7. Run Order

```bash
# 1. Pre-compute embeddings (run once on A100, ~5–10 min)
python -m src.vectorize

# 2. Train + produce expert predictions (~20–30 min on A100)
python -m src.predict
```

`embeddings/` is gitignored — generate it locally before training.

---

## 8. Main Takeaway

The transformer expert is the **multimodal, semantics-aware** leg of the ensemble. Its primary advantages over the tree-based expert (Eddy) are:

1. **Deep text understanding** — `all-mpnet-base-v2` encodes actual policy meaning from DOH press releases (naloxone access expansions, fentanyl test strip legalisation, etc.) rather than word frequencies.
2. **Spatial prescriber patterns** — EfficientNet-B2 features from the MAT density heatmaps encode geographic access barriers invisible in the scalar covariates.
3. **Cross-modality attention** — 6 layers of self-attention allow text, image, and tabular signals to inform each other (e.g. sparse MAT coverage + rising fentanyl search interest → amplified overdose rate).
4. **Non-linear feature interactions** — depth and width (d_model=256, d_ff=1024) capture interactions the other pipelines may linearise away.

Expected OOF MAE: within 10–15% of the tree-based baseline individually; diversity with Eddy's GBM and William's lag models should produce a meaningful MoE lift.
