# Classical Statistics Pipeline (v12)
## STAI-X Challenge 2026 · Overdose ED-Visit Rate Prediction

Classical, interpretable statistics only — no gradient boosting, no neural
networks, no external data. Everything derives from the official competition
files plus the organizer-published `period_id_map.json` (sanctioned: "You can
use this file in your training"; embedded verbatim in the notebook so it runs
on Kaggle with internet disabled).

## OOF scores

**GroupKFold OOF MAE** (5-fold over period_id, val-like economic missingness
simulated) — this is `oof_mae_william.txt`, the conventional number most
comparable to other experts' random-fold CV:

| Category | MAE |
|---|---|
| all_drugs | 2.9066 |
| all_opioids | 1.3319 |
| all_stimulants | 0.6664 |
| **Block average** | **1.6350** |

**Time-aware rolling-origin MAE** (`oof_time_aware_william.txt`) — the metric
that actually mirrors the competition: train on months ≤ T, score horizons
2–7 (Stage 1) and 8–13 (Stage 2), weighted 0.3·S1 + 0.7·S2, averaged over 7
origins. This describes the shipped predictions (including the temporal
anchor, which random-fold CV cannot measure):

| Category | Weighted MAE |
|---|---|
| all_drugs | 2.9530 |
| all_opioids | 1.2644 |
| all_stimulants | 0.7714 |
| **Block weighted** | **1.6629** |

## Architecture

```
Layer 1: per-(jurisdiction, category) MEAN baseline
         × multiplicative period factor (RidgeCV: national Trends+weather →
           log national rate; clipped to training range ±5% for Stage 2 safety)
Layer 2: Huber regression on log-residuals (ε, α tuned by inner GroupKFold
         on raw-scale MAE; 18–20 covariate features, econ down-weighted)
Layer 3: per-jurisdiction bias correction, shrunk toward the global bias (k=30)
Layer 4: per-category ridge/corrected blend (weights grid-searched on OOF)
Layer 4b: temporal anchor — final = (1−α)·stack + α·level_projection(h);
         level_projection = per-state EWMA (half-life 24 mo) + damped (φ=0.95)
         cross-state POOLED 18-month trend slope, projected to the target
         month's horizon h (from the period map). α: drugs 0.60, opioids 0.50,
         stimulants 0 (its Trends-driven period factor already tracks drift).
Layer 5: nesting constraint (all_drugs ≥ subcategories) + floor clip ≥ 0
```

All anchor parameters were tuned on rolling origins {54,57,60,63} and
confirmed on held-out fresh origins {55,58,61} before acceptance. Honest
progression on the time-aware metric: v10 stack 1.7074 → v11 anchor 1.6794 →
v12 pooled-slope anchor 1.6629.

## Stage 2 safety

- Runs unmodified on Kaggle: scans `/kaggle/input/*` for the data mount,
  writes `/kaggle/working/submission.csv` via the mandatory pattern
  (row_id int, finite floats, exactly two columns).
- All row_ids / period_ids / row counts derived from the input files at
  runtime — verified by re-execution on a synthetic Stage 2 dataset with
  extra period_ids (1224 rows) and extreme covariates.
- The period map is embedded in the setup cell (2.5 KB; a map file found on
  disk takes precedence, so an updated Stage 2 map wins automatically).
  Rows with period_ids absent from the map keep the pure stack prediction;
  with no map at all the pipeline degrades exactly to the v10 stack.
- Anchor horizons capped at 24 months; the damped trend sum converges —
  no unbounded extrapolation possible.
- Dependencies: pandas / numpy / scikit-learn / scipy only (all on PyPI).

## Reproduction

Run `pipeline_william_v12.ipynb` top to bottom (Kaggle / Colab / local are
auto-detected). The committed notebook contains the official Colab outputs.
