# STAI-X Challenge 2026

Predict `rate_per_10000_ed_visits` (nonfatal overdose ED visits per 10 000) for 918 held-out rows
across 6 validation periods × 51 jurisdictions × 3 drug categories.

**Metric:** block-averaged MAE (mean of per-category MAEs).
**Final score:** 0.3 × Stage 1 + 0.7 × Stage 2.

---

## Branch: `lenny/transformer-expert` — Expert 2 (Transformer Pipeline)

This branch contains the **FT-Transformer neural expert** for the team's Mixture-of-Experts (MoE) ensemble.
See [`PIPELINE_TRANSFORMER.md`](PIPELINE_TRANSFORMER.md) for the full design specification.

### Architecture summary

```
Input row (period_id × jurisdiction)
  ├─ Numeric features (tabular covariates + lag features)
  ├─ Text features    (state_doh_release → TF-IDF + SVD)
  └─ Image features   (MAT-density PNG → PCA of flat pixels)
          │
          ▼  per-modality linear projection → d_model = 128
  [CLS, numeric_token, text_token, image_token]
          │
          ▼  TransformerEncoder (2 layers, 4 heads, d_ff = 256)
  CLS output → Dropout(0.1)
          │
          ▼  3 regression heads
  [pred_all_drugs, pred_all_opioids, pred_all_stimulants]
```

**Training:** GroupKFold (n=5, groups=period_rank), Huber loss, AdamW + cosine annealing.
**Output:** `expert_transformer.csv` — consumed by the MoE combiner.

### MoE context

| Expert | Branch | Approach |
|---|---|---|
| 1 | — | Gradient-boosted trees (LGB / XGB / CatBoost) |
| **2** | **`lenny/transformer-expert`** | **FT-Transformer (this branch)** |
| 3 | — | — |
| 4 | — | — |

---

## Quick start

```bash
pip install -r requirements.txt
python -m src.predict          # writes expert_transformer.csv at repo root
```

Or via notebook:

```bash
cd notebooks && jupyter lab
# Run 04_submission.ipynb top-to-bottom
```

## Structure

```
stai-x-challenge-2026/
├── src/
│   ├── data_loader.py          # load_train / load_val / load_images_for
│   ├── features.py             # FeaturePipeline — numeric + text + image modalities
│   ├── models.py               # FTTransformer (PyTorch) + MLP fallback
│   └── predict.py              # end-to-end pipeline → expert_transformer.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_features.ipynb
│   ├── 03_model.ipynb
│   └── 04_submission.ipynb
├── PIPELINE_TRANSFORMER.md     # full pipeline design spec
├── CLAUDE.md                   # Award B autonomous agent spec
├── Data_Description.md         # dataset schema
├── requirements.txt
└── expert_transformer.csv      # written at runtime (gitignored)
```

## Award tracks

- **Award A:** best leaderboard MAE — run `04_submission.ipynb` (MoE combines all expert CSVs)
- **Award B:** autonomous pipeline triggered by `"Do the data analysis"` — see `CLAUDE.md`

## Key design decisions

- One wide row per `(period_id × jurisdiction)` fed to all 3 heads simultaneously
- `GroupKFold` grouped by `period_rank` — prevents temporal data leakage
- Temporal order inferred from `temp_avg_f` cross-jurisdiction mean (season proxy)
- `state_doh_release` → TF-IDF (5 000 vocab, bigrams) + TruncatedSVD (32 components)
- MAT-density PNGs resized to 32×32 RGB → flattened → PCA (64 components)
- NaN target cells (suppressed rows) masked from loss — never imputed to zero
- MLP fallback auto-activates if PyTorch is unavailable or 90-min budget exceeded
