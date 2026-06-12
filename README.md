# SEAS the Moment — STAI-X Challenge 2026

Harvard SEAS team submission. This repository contains **two** deliverables:

| Award | What it is | Entry point |
|---|---|---|
| **Award B** | An autonomous **Claude Code data-analysis agent**: given a never-before-seen dataset and a single prompt, it builds models and writes `submission.csv` + `report.pdf` with no human help. | [`CLAUDE.md`](CLAUDE.md) |
| **Award A** | A 4-expert **Mixture-of-Experts** pipeline for the Kaggle leaderboard (the overdose ED-visit task). | [`SEAStheMoment_STAIX26_submission.ipynb`](SEAStheMoment_STAIX26_submission.ipynb) |

The two are independent. Award B is judged by re-running this repo on a **held-out dataset of an undisclosed domain**; it must work without modification. Award A is the Kaggle notebook for the original competition domain.

---

## Award B — Autonomous data-analysis agent

### The idea
The "model" for Award B is not a fixed network — it is an **agent program** written in natural language in [`CLAUDE.md`](CLAUDE.md). When an evaluator opens this repo in Claude Code and types **`Do the data analysis`**, the agent reads the dataset description, infers the task, engineers features, trains a CPU model ensemble, and writes the required outputs — entirely on its own.

### How it is evaluated (held-out procedure)
1. Organizers clone this repo and drop a held-out dataset into `data/` (empty at submission time).
2. They add `data/DATA_DESCRIPTION.md` describing the training file, covariates, validation panel (hidden target), and the required `submission.csv` schema.
3. They open the repo in Claude Code: `claude --dangerously-skip-permissions`, model **Sonnet 4.6 (medium effort)**.
4. They issue exactly one prompt: **`Do the data analysis`**.
5. The agent must write `submission.csv` (two columns: `row_id`, `<target>`) **and** `report.pdf` to the repo root, unattended.
6. `submission.csv` is scored against held-out truth with **block-averaged MAE** (mean of per-group MAEs).

The held-out data may be a **different domain or time window** than the overdose task — so the agent must not assume any particular columns, period set, or row count.

### Pipeline design (what `CLAUDE.md` instructs the agent to do)
A 7-step, domain-agnostic recipe:

1. **Read the data description** — parse target column, key/ID columns, suppression semantics, template path.
2. **Load & inspect** — load the submission template first (it defines the exact rows to predict); print shapes, dtypes, NaN rates for every file.
3. **Infer task structure** — regression vs classification from the target dtype; one model per group if a category column exists; detect temporal order for lag features.
4. **Feature engineering** — apply only the transforms that fit the columns present (text → TF-IDF + SVD, geographic → region encoding, skewed counts → log1p, temporal → lags). **Fit on train, apply to validation** — never fit on validation.
5. **Train models** — per group: drop NaN targets, GroupKFold CV, train **LightGBM + XGBoost + CatBoost**, ensemble the out-of-fold predictions, report OOF MAE/AUC.
6. **Write `submission.csv`** — start from the template, fill the target column, clip/impute, assert no NaN and exact row count.
7. **Write `report.pdf`** — dataset overview, EDA plots, CV results, feature importances, submission preview (via `reportlab`).

### Constraints the agent respects (all encoded in `CLAUDE.md`)
| Constraint | How it's handled |
|---|---|
| **CPU only, no GPU** | Gradient-boosted trees (LightGBM/XGBoost/CatBoost) — no neural nets |
| **No external dataset downloads** | Uses only files under `data/`; network only to `pip install` libraries |
| **≤ 1,000,000 tokens / run** | Compact prints (shapes/heads, not full frames); minimal report code |
| **≤ 2 hours wall-clock** | Skips CatBoost if it approaches the 90-minute mark |
| **No human intervention** | Built-in error recovery: fall back to medians, skip failing models/folds, continue |
| **Schema-exact output** | Always starts from the template; asserts column names, row count, zero NaN |

### Tutorial — run the agent yourself
```bash
# 1. Clone and enter the repo
git clone https://github.com/lennardpische/staix26_seasthemoment.git
cd staix26_seasthemoment

# 2. Put a dataset into data/ in the documented layout, e.g.:
#    data/
#    ├── DATA_DESCRIPTION.md
#    ├── train/  (training file + covariates)
#    ├── val/    (validation covariates, hidden target)
#    └── sample_submission.csv

# 3. Open Claude Code with Sonnet 4.6 and run the single prompt
claude --dangerously-skip-permissions
> Do the data analysis
```
The agent writes `submission.csv` and `report.pdf` to the repo root. `data/` is **empty in this repo** — the organizers populate it at evaluation.

---

## Award A — Mixture of Experts (Kaggle)

Four independent experts, each in `experts/<name>/`, combined per-category with inverse-MAE weights:

| Expert | Folder | Model |
|---|---|---|
| Lenny | `experts/lenny/` | FT-Transformer (multimodal: numeric + text + image) |
| William | `experts/william/` | Classical statistics (5-layer + temporal anchor) |
| Jasmine | `experts/jasmine/` | Healthcare LightGBM |
| Eddy | `experts/eddy/` | XGBoost tree ensemble |

[`SEAStheMoment_STAIX26_submission.ipynb`](SEAStheMoment_STAIX26_submission.ipynb) runs all four (each isolated in its own subprocess so their identically-named modules don't collide), then combines them into `submission.csv`.

### Tutorial — reproduce the Kaggle submission
1. Open the notebook in Google Colab; set **Runtime → A100 GPU** (Lenny's transformer needs it; the rest are CPU).
2. Upload the competition data bundle `staix-challenge.zip` to `/content/`.
3. **Run All** (~35–45 min). It clones the repo, installs `requirements.txt`, runs the four experts, and writes `submission.csv`.

> **Note (Award A reproducibility):** Lenny's expert uses pretrained `all-mpnet-base-v2` and `EfficientNet-B2`. For Kaggle reproduction these must be uploaded to Kaggle as public datasets and attached to the notebook. The notebook falls back to TF-IDF/PCA features if the embeddings are unavailable.

---

## Repository structure
```
staix26_seasthemoment/
├── CLAUDE.md          # Award B agent workflow ("Do the data analysis")
├── .claude/           # Claude Code project configuration
├── README.md          # this file
├── data/              # EMPTY at submission; organizers populate at evaluation
├── experts/           # Award A — four expert pipelines
│   ├── lenny/         #   FT-Transformer
│   ├── william/       #   classical statistics
│   ├── jasmine/       #   healthcare LightGBM
│   └── eddy/          #   XGBoost trees
├── SEAStheMoment_STAIX26_submission.ipynb   # Award A orchestrator notebook
├── period_id_map.json # period_id → calendar date map (original domain)
├── requirements.txt   # dependencies (superset; Award B installs a CPU subset)
└── Data_Description.md # original-domain data description (Award A reference)
```

`data/` is intentionally empty here (a `.gitkeep` placeholder keeps the folder). Nothing under `data/` is committed.

---

## Team — SEAS the Moment (Harvard SEAS)
| Member | Email | Kaggle |
|---|---|---|
| Jasmine Andresol | jasmineandresol@college.harvard.edu | Jasmine Andresol |
| Eddy Kang | eddykang@college.harvard.edu | eddykang06 |
| Lenny Pische | lenny_pische@college.harvard.edu | lpiske |
| William Liu | wmliu@college.harvard.edu | William Liu |
