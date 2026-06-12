# CLAUDE.md — Award B Agent Workflow

When given the prompt **"Do the data analysis"**, follow this workflow exactly.
This specification is domain-agnostic and will work on any dataset that follows
the folder layout described in `data/DATA_DESCRIPTION.md` (or `Data_Description.md`).

> **Scope.** This workflow is self-contained. **Ignore `experts/` and
> `SEAStheMoment_STAIX26_submission.ipynb`** — those are the Award A Kaggle
> submission, hardcoded to the original overdose domain. Do **not** reuse their
> features or models here; the held-out dataset may be a different domain. Build
> everything below from the data in `data/` alone.

> **Environment.** CPU-only, no GPU. Network is available for installing Python
> packages but **must not** be used to download any external dataset. Before
> step 2, make sure the libraries are importable; install any that are missing:
> `pip install -q lightgbm xgboost catboost reportlab scikit-learn matplotlib seaborn`.

---

## Hard constraints

| Constraint | Value |
|---|---|
| External data | **Forbidden.** Use only files under `data/`. |
| GPU | **Not available.** CPU-only models only. |
| Token budget | ≤ 1 000 000 tokens total across all tool calls. |
| Wall-clock time | ≤ 2 hours. |
| Submission schema | Must match `DATA_DESCRIPTION.md` exactly — same columns, same dtypes, no extra rows, no missing rows, no NaN. |

---

## Step-by-step workflow

### 1 — Read data description

```
Read data/DATA_DESCRIPTION.md (or Data_Description.md if that path fails).
```

Parse and record:
- Target column name and dtype
- Key column names (ID, join keys, group key)
- Suppression semantics (NaN = suppressed or missing?)
- Whether a submission template exists and its path

### 2 — Load and inspect data

```python
import pandas as pd, numpy as np
train = pd.read_csv('data/train/...csv')
print(train.shape, train.dtypes, train.isna().mean().sort_values(ascending=False).head(20))
```

Always load the submission template first to establish the exact rows to predict.
Print shape, dtypes, and NaN rates for every file loaded.

### 3 — Infer task structure

Determine automatically:
- **Regression vs classification**: float target → regression; integer/bool/categorical target → classification
- **Single vs multi-group**: if there is a `category`/`type`/`overdose_category`-style column, train one model per group value
- **Temporal structure**: if there is a `period_id` or date-like column, infer temporal order (use temperature proxy or alphabetical fallback); enable lag features only when >1 unique time step exists

### 4 — Feature engineering

Apply all applicable transforms; skip those that don't apply:

| Condition | Transform |
|---|---|
| Text column present | TF-IDF (max 5 000 features, sublinear_tf) + TruncatedSVD (20 components) |
| Geographic column present | Map to census division integer (1–9) |
| Continuous labor/population column | log1p transform |
| `gtrends_*` columns present | Pairwise product interactions: fentanyl×opioid, overdose×naloxone |
| Temporal order recoverable | rate_lag_1, rate_lag_2 per (jurisdiction, category) sorted by period_rank |

All transforms: **fit on train, apply to val**. Never fit on val.

Impute all numeric NaN with training-set medians after feature engineering.

### 5 — Train models

For each target group (or once if no grouping):

1. Drop rows where target is NaN
2. GroupKFold (n_splits=5, groups=period_rank or row-index if no temporal order)
3. Train LightGBM (`objective=regression_l1` or `binary` depending on task, `early_stopping_rounds=100`)
4. Train XGBoost and CatBoost with matching objectives
5. OOF ensemble = mean of all model × fold predictions
6. Print OOF MAE (regression) or AUC (classification) per group

Report mean OOF metric across all groups — this is the competition metric.

### 6 — Write submission.csv

```python
# target_col and the id/key columns come from DATA_DESCRIPTION.md — never hardcode them
submission = template[[id_col]].copy()         # always start from the template
# ... merge predictions ...
submission[target_col] = preds.clip(lower=0)   # clip only if the target is a non-negative rate
# Fill any remaining NaN with the per-group (or global) median from train
assert submission.isna().sum().sum() == 0, "NaN in submission"
assert len(submission) == len(template), "Row count mismatch"
submission.to_csv('submission.csv', index=False)
```

Two columns only: the id column and the target column, exactly as named in
`DATA_DESCRIPTION.md`. No extras.

### 7 — Write report.pdf

Use `reportlab` to generate `report.pdf` with the following sections:

1. **Dataset overview** — shape, NaN rates, suppression rate
2. **EDA plots** — target distribution per group (histogram), feature correlation heatmap
3. **CV results table** — OOF MAE per group and model family
4. **Feature importances** — top-15 LightGBM importances per group (bar chart)
5. **Submission preview** — head(10) of submission.csv

Minimal code; use `reportlab.platypus` (SimpleDocTemplate, Table, Image, Paragraph).

---

## Error recovery

- If a file path from DATA_DESCRIPTION.md doesn't exist, try common alternatives (`data/`, `train/`, root).
- If temporal ordering cannot be recovered (all temp_avg_f are NaN), skip lag features silently.
- If a model raises an error on a group, log the error, fill that group with the train median, and continue.
- If wall-clock approaches 90 minutes, skip CatBoost and use LGB + XGB only.

---

## Output checklist

Before stopping, verify:
- [ ] `submission.csv` exists, has correct column names, 0 NaN, all row_ids from template
- [ ] `report.pdf` exists and is readable
- [ ] No external URLs were accessed
- [ ] Total token usage stayed under 1 000 000
