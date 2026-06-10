"""Expert William — Classical Statistics Pipeline (v12).
Faithful export of pipeline_william_v12.ipynb (code cells only, in order).
Runs end-to-end: load → OOF → weight-opt → fit → predict → anchor → write.
Writes expert_william.csv to the repo root."""

import sys, warnings, json, pickle
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import RidgeCV, HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

# Environment detection: Kaggle (evaluation) → Colab (team workflow) → local.
# RULES.md quotes /kaggle/input/staix-challenge, but scan every attached input in
# case the competition mount uses a different slug — silently falling through to
# local mode on Kaggle would crash the Stage 2 re-execution.
def _find_kaggle_root():
    base = Path("/kaggle/input")
    if not base.exists():
        return None
    preferred = base / "staix-challenge"
    candidates = [preferred] + sorted(p for p in base.iterdir() if p.is_dir())
    for c in candidates:
        if (c / "train" / "dose_sys_train.csv").exists() and (c / "sample_submission.csv").exists():
            return c
    return None

_KAGGLE_ROOT = _find_kaggle_root()
IS_KAGGLE = _KAGGLE_ROOT is not None
IS_COLAB  = False
if IS_KAGGLE:
    DATA   = _KAGGLE_ROOT
    GDRIVE = Path("/kaggle/working/cache")   # fresh every run; caching is a no-op
    print(f"✓ Kaggle environment — data root: {DATA}")
else:
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        GDRIVE   = Path("/content/drive/MyDrive/stai_x_william")
        IS_COLAB = True
        print("✓ Colab + Drive mounted")
    except ImportError:
        GDRIVE = Path("cache")
        print(f"ℹ  Local mode — cache at: {GDRIVE.resolve()}")
    # Data root: Drive layout on Colab; fall back to ./data or repo-root ./ locally.
    DATA = GDRIVE / "data"
    for _cand in [DATA, Path("data"), Path(".")]:
        if (_cand / "train" / "dose_sys_train.csv").exists():
            DATA = _cand
            break

GDRIVE.mkdir(parents=True, exist_ok=True)
TRAIN_TGT  = DATA / "train" / "dose_sys_train.csv"
TRAIN_COV  = DATA / "train" / "covariates.csv"
VAL_COV    = DATA / "val"   / "covariates.csv"
SUBMISSION = DATA / "sample_submission.csv"

SCORED_CATS   = ["all_drugs", "all_opioids", "all_stimulants"]
GTRENDS_COLS  = ["gtrends_overdose", "gtrends_fentanyl", "gtrends_naloxone",
                 "gtrends_opioid", "gtrends_methamphetamine"]
WEATHER_COLS  = ["temp_avg_f", "precip_in"]
ECONOMIC_COLS = ["unemployment_rate", "labor_force"]

GTRENDS_ACTIVE = {
    "all_opioids":    ["gtrends_overdose", "gtrends_fentanyl", "gtrends_naloxone"],
    "all_stimulants": ["gtrends_overdose", "gtrends_fentanyl", "gtrends_naloxone",
                       "gtrends_methamphetamine"],
    "all_drugs":      ["gtrends_overdose", "gtrends_fentanyl", "gtrends_naloxone"],
}

PP_EXTRA_FEATURES = []          # v6 post-mortem: secular-trend proxies overfit

ALPHA_GRID = np.logspace(-2, 4, 40)   # RidgeCV grid (period predictor)
# v10 Layer 2: Huber on log-residuals; inner 3-fold GroupKFold picks (eps, alpha)
# by raw-scale MAE including the bias correction.
HUBER_EPS_GRID   = [1.1, 1.35, 2.0]
HUBER_ALPHA_GRID = [1e-4, 1e-2, 1.0, 100.0]
BIAS_SHRINK_K    = 30.0    # per-jurisdiction bias partial pooling strength
FACTOR_CLIP_MARGIN = 1.05  # period factor clipped to train range ± 5%
N_CV_FOLDS = 5
MISS_RATES = {"unemployment_rate": 0.55, "labor_force": 0.58}

# Initial weights; replaced by the OOF grid search in Section 4b.
_RAW_WEIGHTS = {
    "all_drugs":      {"ridge": 0.10, "corrected": 0.90},
    "all_opioids":    {"ridge": 0.10, "corrected": 0.90},
    "all_stimulants": {"ridge": 0.10, "corrected": 0.90},
}
WEIGHTS = {
    cat: {k: v / sum(d.values()) for k, v in d.items()}
    for cat, d in _RAW_WEIGHTS.items()
}

KEYWORD_MAP = {
    "all_opioids":    ["opioid", "opioids", "heroin", "fentanyl", "naloxone",
                       "narcan", "overdose", "buprenorphine", "methadone", "suboxone"],
    "all_stimulants": ["methamphetamine", "meth", "cocaine", "stimulant", "stimulants",
                       "amphetamine", "crack", "crystal"],
    "all_drugs":      ["opioid", "opioids", "heroin", "fentanyl", "naloxone",
                       "methamphetamine", "meth", "cocaine", "overdose", "drug", "drugs",
                       "substance", "alert", "harm", "naltrexone"],
}

# v12: period_id -> calendar month map (official organizer file, published at
# https://storage.googleapis.com/kaggle-forum-message-attachments/3455144/42374/period_id_map.json
# with explicit permission: "You can use this file in your training").
# A file found on disk takes precedence (so an updated Stage 2 map in
# /kaggle/input wins); otherwise the verbatim embedded copy below is used,
# keeping the notebook self-contained with internet disabled.
def _find_period_map():
    candidates = [DATA / "period_id_map.json", Path("period_id_map.json"),
                  GDRIVE / "period_id_map.json", DATA.parent / "period_id_map.json"]
    if IS_KAGGLE:
        candidates += sorted(Path("/kaggle/input").glob("*/period_id_map.json"))
    for c in candidates:
        if c.exists():
            return c
    return None

_EMBEDDED_MAP = '{"2019-01-31":"uTjgI1Sv","2019-02-28":"wf016pk5","2019-03-31":"BkTW58Ff","2019-04-30":"shDD7wDP","2019-05-31":"aZFXT65l","2019-06-30":"fizSTkFs","2019-07-31":"wQLd1SNL","2019-08-31":"kmxcVN2e","2019-09-30":"x24Jbzaz","2019-10-31":"FuLb1kk4","2019-11-30":"1Tl9271R","2019-12-31":"Fj7ebbrB","2020-01-31":"h9Re4kM3","2020-02-29":"gqVDbZc7","2020-03-31":"UKjQnuej","2020-04-30":"TedmliP4","2020-05-31":"PJu8Wb2C","2020-06-30":"mZcpe0Ud","2020-07-31":"NOtvYKB9","2020-08-31":"YBlSTfgc","2020-09-30":"rX3aMRGn","2020-10-31":"44RA6kMl","2020-11-30":"88NtGYTF","2020-12-31":"tVb8fHGc","2021-01-31":"aHT3VIho","2021-02-28":"a239r7U4","2021-03-31":"kTaI18at","2021-04-30":"FZxIVFvr","2021-05-31":"27DK2m8F","2021-06-30":"y5ysDDpd","2021-07-31":"NWU8bRHI","2021-08-31":"3JILuYCd","2021-09-30":"56aULHvm","2021-10-31":"KDy1VIvO","2021-11-30":"j1dZWmlF","2021-12-31":"OugqP9RF","2022-01-31":"4VCAqmuO","2022-02-28":"nICRHvl9","2022-03-31":"omhpgEVm","2022-04-30":"WB9kCj4E","2022-05-31":"iIi2mgES","2022-06-30":"CDQGTxV0","2022-07-31":"ePA08XXo","2022-08-31":"MZ0ENeKD","2022-09-30":"4MVfmuye","2022-10-31":"N81HwK1a","2022-11-30":"QpWgWZqu","2022-12-31":"68B5zQl0","2023-01-31":"BhtGJhRU","2023-02-28":"9Dp3l3qq","2023-03-31":"LALpfR23","2023-04-30":"wa7tAVQg","2023-05-31":"eVeAG5UX","2023-06-30":"0Un18Xny","2023-07-31":"9FQthr9A","2023-08-31":"xtjIUpyk","2023-09-30":"yIHgtqjY","2023-10-31":"DpR0556d","2023-11-30":"lSdEh765","2023-12-31":"7cCeqHbf","2024-01-31":"OqDkgaDk","2024-02-29":"k4mmkR0U","2024-03-31":"63zxcdKZ","2024-04-30":"S2Qn2n8u","2024-05-31":"i9aSkhZb","2024-06-30":"UJgFAh3i","2024-07-31":"OIpwoBOI","2024-08-31":"lfTz14iT","2024-09-30":"3CdEQbdr","2024-10-31":"Kk6iVNym","2024-11-30":"tLoy7Zpr","2024-12-31":"S1xSdqr5","2025-01-31":"Hy8SBtar","2025-02-28":"rle4IZEn","2025-03-31":"5Lptd03a","2025-04-30":"jtUOZLP4","2025-05-31":"dp3VfN8B","2025-06-30":"dsZhPyK4","2025-07-31":"aL5zkp6g","2025-08-31":"yFh3wzPe","2025-09-30":"lXSJn8AD","2025-10-31":"kbpS9xmS","2025-11-30":"DmKNJoJt","2025-12-31":"if5b8Sut","2026-01-31":"WNFmh9iQ","2026-02-28":"myO2m6ax","2026-03-31":"fN5pFXQU","2026-04-30":"Ja22UVH5","2026-05-31":"JuylH0n8","2026-06-30":"E8YQntpd"}'

_MAP_PATH = _find_period_map()
if _MAP_PATH is not None:
    with open(_MAP_PATH) as f:
        _raw_map = json.load(f)                      # {"YYYY-MM-DD": period_id}
    _map_src = str(_MAP_PATH)
else:
    _raw_map = json.loads(_EMBEDDED_MAP)
    _map_src = "embedded copy (organizer-published)"
DATE_OF_PERIOD = {pid: pd.Timestamp(d) for d, pid in _raw_map.items()}
print(f"✓ period map: {_map_src} ({len(DATE_OF_PERIOD)} periods)")

# Layer 4b temporal anchor. v12: pooled cross-state trend slope (see
# level_projection) lets alpha rise — tuned on origins {54,57,60,63}, confirmed
# on fresh {55,58,61}. Stimulants stay 0 (worse at any alpha tested; its
# period factor already tracks the drift).
ALPHA_LEVEL = {"all_drugs": 0.60, "all_opioids": 0.50, "all_stimulants": 0.0}
LEVEL_HALFLIFE     = 24    # months, EWMA level
LEVEL_PHI          = 0.95  # trend damping
LEVEL_SLOPE_WINDOW = 18    # months used for the slope fit
MAX_ANCHOR_HORIZON = 24    # beyond this, fall back to the pure stack

print("✓ Setup complete")

def keyword_score(text_series, keywords):
    texts  = text_series.fillna("").str.lower()
    scores = pd.Series(0.0, index=texts.index)
    for kw in keywords:
        scores += texts.str.count(r"\b" + kw + r"\b")
    return scores


def simulate_missingness(df, seed=42):
    rng = np.random.default_rng(seed)
    df  = df.copy()
    for col, rate in MISS_RATES.items():
        if col in df.columns:
            df.loc[rng.random(len(df)) < rate, col] = np.nan
    return df


def compute_period_means(cov_df):
    cols = [c for c in GTRENDS_COLS + WEATHER_COLS if c in cov_df.columns]
    return cov_df.groupby("period_id")[cols].mean()


def compute_jur_trend_means(cov_df):
    cols = [c for c in GTRENDS_COLS if c in cov_df.columns]
    return cov_df.groupby("jurisdiction")[cols].mean()


def get_active_feat_cols(cat, feat_cols):
    inactive = set(GTRENDS_COLS) - set(GTRENDS_ACTIVE.get(cat, GTRENDS_COLS))
    dead = (list(inactive)
            + [f"{c}_dev"   for c in inactive]
            + [f"{c}_jurdm" for c in inactive])
    return [c for c in feat_cols if c not in dead]


def fit_period_predictor(cat, train_df, train_cov_df):
    active      = GTRENDS_ACTIVE[cat]
    pp_features = active + PP_EXTRA_FEATURES + WEATHER_COLS

    cat_df       = train_df[train_df["overdose_category"] == cat]
    period_rates = cat_df.groupby("period_id")["rate_per_10000_ed_visits"].mean()
    period_feats = train_cov_df.groupby("period_id")[pp_features].mean()
    period_feats = period_feats.fillna(period_feats.median())

    common         = period_rates.index.intersection(period_feats.index)
    y_log          = np.log(period_rates.loc[common].values)
    X              = period_feats.loc[common].values
    grand_mean_log = float(y_log.mean())

    scaler    = StandardScaler()
    Xs        = scaler.fit_transform(X)
    predictor = RidgeCV(alphas=ALPHA_GRID)
    predictor.fit(Xs, y_log)

    # v10: training-observed factor range for Stage 2 extrapolation insurance.
    # Clipping at ±5% is CV-neutral (verified) and only binds when a future
    # period's covariates fall outside anything seen in training.
    fac_in = np.exp(predictor.predict(Xs) - grand_mean_log)
    factor_lo = float(fac_in.min()) / FACTOR_CLIP_MARGIN
    factor_hi = float(fac_in.max()) * FACTOR_CLIP_MARGIN

    return {"predictor":      predictor,
            "grand_mean_log": grand_mean_log,
            "grand_mean":     float(np.exp(grand_mean_log)),
            "scaler":         scaler,
            "active":         active,
            "pp_features":    pp_features,
            "factor_lo":      factor_lo,
            "factor_hi":      factor_hi,
            "alpha":          float(predictor.alpha_),
            "r2_log":         float(predictor.score(Xs, y_log))}


def predict_period_factor(cat, feat_df, period_predictors):
    info        = period_predictors[cat]
    pp_features = info.get("pp_features", info["active"])

    avail        = [c for c in pp_features if c in feat_df.columns]
    period_feats = feat_df.groupby("period_id")[avail].mean()
    for col in pp_features:
        if col not in period_feats.columns:
            period_feats[col] = 0.0
    period_feats = period_feats[pp_features].fillna(period_feats.median())

    periods  = period_feats.index
    Xs       = info["scaler"].transform(period_feats.values)
    pred_log = info["predictor"].predict(Xs)
    factors  = np.exp(pred_log - info["grand_mean_log"])
    if "factor_lo" in info:
        factors = np.clip(factors, info["factor_lo"], info["factor_hi"])
    factor_map = dict(zip(periods, factors))
    return feat_df["period_id"].map(factor_map).fillna(1.0).values


def build_features(cov_df, period_means_df, econ_medians=None, jur_trend_means=None):
    df = cov_df.copy()

    pm = period_means_df.reset_index().rename(
        columns={c: f"{c}_natl" for c in period_means_df.columns}
    )
    df = df.merge(pm, on="period_id", how="left")
    for col in GTRENDS_COLS + WEATHER_COLS:
        natl = f"{col}_natl"
        df[f"{col}_dev"] = (df[col] - df[natl]) if (col in df.columns and natl in df.columns) else 0.0

    for col in GTRENDS_COLS:
        if col in df.columns and jur_trend_means is not None and col in jur_trend_means.columns:
            df[f"{col}_jurdm"] = df[col] - df["jurisdiction"].map(jur_trend_means[col])
        else:
            df[f"{col}_jurdm"] = 0.0

    if "temp_avg_f" in period_means_df.columns:
        rank_map = period_means_df["temp_avg_f"].rank(pct=True).to_dict()
        df["seasonal_position"] = df["period_id"].map(rank_map).fillna(0.5)
    else:
        df["seasonal_position"] = 0.5

    if "state_doh_release" in df.columns:
        df["release_issued"]    = df["state_doh_release"].notna().astype(float)
        df["release_wordcount"] = df["state_doh_release"].fillna("").str.split().str.len().astype(float)
    else:
        df["release_issued"]    = 0.0
        df["release_wordcount"] = 0.0

    for col in ECONOMIC_COLS:
        if col in df.columns:
            df[f"{col}_missing"] = df[col].isna().astype(float)
            fill_val = econ_medians[col] if (econ_medians and col in econ_medians) else df[col].median()
            df[col]  = df[col].fillna(fill_val)
        else:
            df[f"{col}_missing"] = 1.0
            df[col]              = 0.0

    return df


FEAT_COLS = (
    [f"{c}_dev"   for c in GTRENDS_COLS]
    + [f"{c}_jurdm" for c in GTRENDS_COLS]
    + WEATHER_COLS
    + [f"{c}_dev"   for c in WEATHER_COLS]
    + ["seasonal_position", "release_issued", "release_wordcount"]
    + ECONOMIC_COLS
    + [f"{c}_missing" for c in ECONOMIC_COLS]
    + ["keyword_score"]
)  # 22 total


def compute_baselines(train_df):
    scored    = train_df[train_df["overdose_category"].isin(SCORED_CATS)]
    baselines = (
        scored.groupby(["jurisdiction", "overdose_category"])["rate_per_10000_ed_visits"]
        .agg(baseline_median="median", baseline_mean="mean")
        .reset_index()
    )
    global_fb = scored.groupby("overdose_category")["rate_per_10000_ed_visits"].median().to_dict()
    return baselines, global_fb


def fit_cat_models(cat, cat_train_df, feat_df, baselines, global_fb,
                   period_predictors=None):
    merged   = cat_train_df.merge(feat_df, on=["jurisdiction", "period_id"], how="left")
    cat_base = baselines[baselines["overdose_category"] == cat][["jurisdiction", "baseline_mean"]]
    merged   = merged.merge(cat_base, on="jurisdiction", how="left")
    merged["baseline_mean"] = merged["baseline_mean"].fillna(global_fb.get(cat, 0.0))

    text_col = (merged["state_doh_release"] if "state_doh_release" in merged.columns
                else pd.Series("", index=merged.index))
    merged["keyword_score"] = keyword_score(text_col, KEYWORD_MAP[cat])

    active    = get_active_feat_cols(cat, FEAT_COLS)
    available = [c for c in active if c in merged.columns]
    X         = merged[available].fillna(0).values
    y         = merged["rate_per_10000_ed_visits"].values
    # v9: per-jurisdiction MEAN baseline (was median). With the multiplicative
    # period factor and log-residual Ridge downstream, residuals off the mean are
    # better-centred; verified -0.011 block-MAE in CV.
    base      = merged["baseline_mean"].values

    period_factor = (predict_period_factor(cat, merged, period_predictors)
                     if period_predictors is not None and cat in period_predictors
                     else np.ones(len(merged)))
    base_adj = np.clip(base * period_factor, 0, None)

    y_log_resid = np.log1p(y) - np.log1p(base_adj)

    scaler = StandardScaler()
    Xs     = scaler.fit_transform(X)

    # v10 Layer 2: Huber regression, robust to heavy-tailed rate spikes.
    # (eps, alpha) chosen by inner GroupKFold over period_id, scoring raw-scale
    # MAE including the bias correction so the selection matches the pipeline.
    period_groups = merged["period_id"].values
    inner_gkf     = GroupKFold(n_splits=3)
    best_cfg, best_inner = (1.35, 1e-2), np.inf
    for eps in HUBER_EPS_GRID:
        for alpha in HUBER_ALPHA_GRID:
            fold_maes = []
            for tr_i, va_i in inner_gkf.split(Xs, groups=period_groups):
                h = HuberRegressor(epsilon=eps, alpha=alpha, max_iter=500)
                h.fit(Xs[tr_i], y_log_resid[tr_i])
                pred_tr = np.expm1(h.predict(Xs[tr_i]) + np.log1p(base_adj[tr_i]))
                b       = float(np.mean(y[tr_i] - pred_tr))
                pred_va = np.expm1(h.predict(Xs[va_i]) + np.log1p(base_adj[va_i])) + b
                fold_maes.append(np.mean(np.abs(y[va_i] - pred_va)))
            m = np.mean(fold_maes)
            if m < best_inner:
                best_inner, best_cfg = m, (eps, alpha)

    model = HuberRegressor(epsilon=best_cfg[0], alpha=best_cfg[1], max_iter=500)
    model.fit(Xs, y_log_resid)

    # Layer 3 (v10): per-jurisdiction bias, shrunk toward the global bias
    # (partial pooling). Corrects both the Jensen back-transform underprediction
    # and any per-jurisdiction residual level the log-space fit leaves behind.
    pred_in = np.expm1(model.predict(Xs) + np.log1p(base_adj))
    resid   = y - pred_in
    bias    = float(np.mean(resid))
    rd      = pd.DataFrame({"jur": merged["jurisdiction"].values, "r": resid})
    g       = rd.groupby("jur")["r"].agg(["mean", "count"])
    bias_jur = ((g["mean"] * g["count"] + bias * BIAS_SHRINK_K)
                / (g["count"] + BIAS_SHRINK_K)).to_dict()

    return {"ridge": model, "scaler": scaler, "bias": bias, "bias_jur": bias_jur,
            "epsilon": best_cfg[0], "alpha": best_cfg[1],
            "n_features": X.shape[1], "feat_cols": available}


def predict_cat(cat, feat_df, baselines, global_fb, models,
                period_predictors=None, return_components=False):
    df = feat_df.copy()
    cat_base = baselines[baselines["overdose_category"] == cat].set_index("jurisdiction")
    df["baseline_mean"] = df["jurisdiction"].map(cat_base["baseline_mean"]).fillna(global_fb.get(cat, 0))

    text_col = (df["state_doh_release"] if "state_doh_release" in df.columns
                else pd.Series("", index=df.index))
    df["keyword_score"] = keyword_score(text_col, KEYWORD_MAP[cat])

    m         = models[cat]
    available = [c for c in m["feat_cols"] if c in df.columns]
    X         = df[available].fillna(0).values
    n_tr      = m["n_features"]
    if X.shape[1] < n_tr:
        X = np.hstack([X, np.zeros((len(df), n_tr - X.shape[1]))])
    X  = X[:, :n_tr]
    Xs = m["scaler"].transform(X)

    base = df["baseline_mean"].values

    period_factor = (predict_period_factor(cat, df, period_predictors)
                     if period_predictors is not None and cat in period_predictors
                     else np.ones(len(df)))
    base_adj = np.clip(base * period_factor, 0, None)

    ridge_rate = np.expm1(m["ridge"].predict(Xs) + np.log1p(base_adj))
    bias_vec   = (df["jurisdiction"].map(m["bias_jur"]).fillna(m["bias"]).values
                  if m.get("bias_jur") else m["bias"])
    corrected_rate = ridge_rate + bias_vec

    w    = WEIGHTS[cat]
    pred = w["ridge"] * ridge_rate + w["corrected"] * corrected_rate

    idx = pd.MultiIndex.from_arrays([df["jurisdiction"], df["period_id"]],
                                    names=["jurisdiction", "period_id"])
    preds_ser = pd.Series(pred, index=idx, name="pred")

    if return_components:
        comp_df = df[["jurisdiction", "period_id"]].copy().reset_index(drop=True)
        comp_df["base_adj"]       = base_adj
        comp_df["ridge_rate"]     = ridge_rate
        comp_df["corrected_rate"] = corrected_rate
        return preds_ser, comp_df

    return preds_ser


def months_between(d0, d1):
    return (d1.year - d0.year) * 12 + (d1.month - d0.month)


def build_rate_pivots(train_df):
    """Per-category date × jurisdiction rate matrices (mapped periods only)."""
    sc = train_df[train_df["overdose_category"].isin(SCORED_CATS)].copy()
    sc["date"] = sc["period_id"].map(DATE_OF_PERIOD)
    sc = sc.dropna(subset=["date"])
    return {cat: (sc[sc["overdose_category"] == cat]
                  .pivot_table(index="date", columns="jurisdiction",
                               values="rate_per_10000_ed_visits").sort_index())
            for cat in SCORED_CATS}


def level_projection(piv, h):
    """EWMA level + damped POOLED linear trend, projected h months ahead.
    v12: per-state 18-month slopes are noise — the cross-state mean slope
    improved the weighted block on both tuning and fresh origin sets."""
    lvl = piv.ewm(halflife=LEVEL_HALFLIFE).mean().iloc[-1]
    recent = piv.iloc[-LEVEL_SLOPE_WINDOW:]
    xs = np.arange(len(recent))
    slope = recent.apply(lambda col: np.polyfit(xs, col.values, 1)[0])
    slope = pd.Series(slope.mean(), index=slope.index)   # full pooling (lambda=1)
    damp = sum(LEVEL_PHI ** i for i in range(1, h + 1))
    return (lvl + slope * damp).clip(lower=0.0)


def apply_temporal_anchor(val_keys, pivots):
    """Layer 4b: blend stack predictions with per-series level projections.
    Rows with unmapped period_ids (or h outside [1, MAX_ANCHOR_HORIZON]) keep
    the pure stack prediction — this is the Stage 2 degradation path."""
    if not DATE_OF_PERIOD:
        print("  (no period map — anchor skipped)")
        return val_keys
    train_end = max(p.index.max() for p in pivots.values())
    out = val_keys.copy()
    proj_cache = {}
    for cat in SCORED_CATS:
        a = ALPHA_LEVEL.get(cat, 0.0)
        if a <= 0:
            continue
        blended, n_anchored = [], 0
        for _, row in out.iterrows():
            v = row[cat]
            d = DATE_OF_PERIOD.get(row["period_id"])
            if d is not None:
                h = months_between(train_end, d)
                if 1 <= h <= MAX_ANCHOR_HORIZON:
                    key = (cat, h)
                    if key not in proj_cache:
                        proj_cache[key] = level_projection(pivots[cat], h)
                    lp = proj_cache[key].get(row["jurisdiction"])
                    if lp is not None and np.isfinite(lp):
                        v = (1 - a) * v + a * lp
                        n_anchored += 1
            blended.append(v)
        out[cat] = blended
        print(f"  {cat}: anchored {n_anchored}/{len(out)} rows (alpha={a})")
    return out


print("✓ Functions defined")

train      = pd.read_csv(TRAIN_TGT)
train_cov  = pd.read_csv(TRAIN_COV)
val_cov    = pd.read_csv(VAL_COV)
submission = pd.read_csv(SUBMISSION)

# RULES.md schema portability: never assume a specific row count or period set.
# Stage 2 re-execution adds new period_ids to covariates.csv and sample_submission.csv.
assert submission.shape[0] > 0
assert set(submission.columns) >= {"row_id", "period_id", "jurisdiction",
                                    "overdose_category", "rate_per_10000_ed_visits"}
N_SUB_ROWS = submission.shape[0]

print(f"train targets  : {train.shape}")
print(f"train_cov      : {train_cov.shape}")
print(f"val_cov        : {val_cov.shape}")
print(f"submission     : {submission.shape}")

ECON_MEDIANS_TRAIN    = {col: float(train_cov[col].median())
                         for col in ECONOMIC_COLS if col in train_cov.columns}
JUR_TREND_MEANS_TRAIN = compute_jur_trend_means(train_cov)
ALL_PM = compute_period_means(pd.concat([train_cov, val_cov], ignore_index=True))

print(f"\nEconomic medians (train) : {ECON_MEDIANS_TRAIN}")
print(f"Jurisdiction Trends means: {len(JUR_TREND_MEANS_TRAIN)} jurisdictions")
print(f"All known periods        : {len(ALL_PM)} (train + val)")
print("✓ Data loaded")

scored = train[train["overdose_category"].isin(SCORED_CATS)]

print("=== Target scale ===")
print(scored.groupby("overdose_category")["rate_per_10000_ed_visits"]
      .agg(["mean", "median", "std", "min", "max"]).round(2))

print("\n=== Within-state Trends correlations ===")
print(f"  {'Feature':<30}" + "".join(f"{c:>16}" for c in SCORED_CATS))
print("  " + "-" * (30 + 16 * 3))
cov_merged = scored.merge(train_cov, on=["jurisdiction", "period_id"], how="left")
for col in GTRENDS_COLS:
    corrs = []
    for cat in SCORED_CATS:
        sub = cov_merged[cov_merged["overdose_category"] == cat].copy()
        jm  = sub.groupby("jurisdiction")["rate_per_10000_ed_visits"].transform("mean")
        corrs.append(sub[col].corr(sub["rate_per_10000_ed_visits"] - jm))
    active_cats = [c for c in SCORED_CATS if col in GTRENDS_ACTIVE[c]]
    flag = (" [ridge: " + ", ".join(a.replace("all_", "") for a in active_cats) + "]"
            if active_cats else " [ridge: EXCLUDED]")
    print(f"  {col:<30}" + "".join(f"{r:>16.3f}" for r in corrs) + flag)

print("\n=== Period predictor R² (GTRENDS_ACTIVE + WEATHER_COLS) ===")
print(f"  {'Category':<20} {'alpha':>8} {'R²_log':>8}")
print("  " + "-" * 40)
for cat in SCORED_CATS:
    pp = fit_period_predictor(cat, scored, train_cov)
    print(f"  {cat:<20} {pp['alpha']:>8.3f} {pp['r2_log']:>8.3f}")

print("\n=== Missingness ===")
cols_check = ECONOMIC_COLS + GTRENDS_COLS + WEATHER_COLS
miss = pd.DataFrame({
    "train": train_cov[cols_check].isna().mean(),
    "val":   val_cov[cols_check].isna().mean()
}).round(3)
print(miss)

if DATE_OF_PERIOD:
    tr_dates = sorted(DATE_OF_PERIOD[p] for p in train["period_id"].unique() if p in DATE_OF_PERIOD)
    va_dates = sorted(DATE_OF_PERIOD[p] for p in val_cov["period_id"].unique() if p in DATE_OF_PERIOD)
    print("\n=== Timeline (period map) ===")
    print(f"  train: {tr_dates[0]:%Y-%m} .. {tr_dates[-1]:%Y-%m} ({len(tr_dates)} months)")
    if va_dates:
        print(f"  val  : {va_dates[0]:%Y-%m} .. {va_dates[-1]:%Y-%m} "
              f"(horizons {months_between(tr_dates[-1], va_dates[0])}.."
              f"{months_between(tr_dates[-1], va_dates[-1])} months past train end)")
    unmapped = [p for p in val_cov["period_id"].unique() if p not in DATE_OF_PERIOD]
    if unmapped:
        print(f"  ⚠ unmapped val periods (anchor disabled for these): {unmapped}")

OOF_CACHE      = GDRIVE / "oof_results_v12.json"
OOF_COMP_CACHE = GDRIVE / "oof_components_v12.parquet"

if OOF_CACHE.exists() and OOF_COMP_CACHE.exists():
    with open(OOF_CACHE) as f:
        oof_saved = json.load(f)
    oof_components_df = pd.read_parquet(OOF_COMP_CACHE)
    print("✓ OOF results and components loaded from cache")
else:
    print("Running OOF cross-validation...")
    scored_cv = train[train["overdose_category"].isin(SCORED_CATS)].copy()
    gkf       = GroupKFold(n_splits=N_CV_FOLDS)

    fold_oof_dfs  = []
    fold_comp_dfs = []

    for fold, (tr_idx, va_idx) in enumerate(
        gkf.split(scored_cv, groups=scored_cv["period_id"])
    ):
        fold_tr = scored_cv.iloc[tr_idx]
        fold_va = scored_cv.iloc[va_idx]
        tr_pids = fold_tr["period_id"].unique()
        va_pids = fold_va["period_id"].unique()

        fc_tr      = train_cov[train_cov["period_id"].isin(tr_pids)].copy()
        fc_va_miss = simulate_missingness(
            train_cov[train_cov["period_id"].isin(va_pids)], seed=42 + fold
        )

        econ_med_fold        = {col: float(fc_tr[col].median())
                                 for col in ECONOMIC_COLS if col in fc_tr.columns}
        jur_trend_means_fold = compute_jur_trend_means(fc_tr)

        feat_tr = build_features(fc_tr, compute_period_means(fc_tr),
                                  jur_trend_means=jur_trend_means_fold)
        feat_va = build_features(fc_va_miss, compute_period_means(fc_va_miss),
                                  econ_medians=econ_med_fold,
                                  jur_trend_means=jur_trend_means_fold)

        fold_baselines, fold_fb = compute_baselines(fold_tr)

        period_predictors_fold = {
            cat: fit_period_predictor(cat, fold_tr, fc_tr)
            for cat in SCORED_CATS
        }

        # v9: no stacking pass — each category is fit independently.
        fold_models = {
            cat: fit_cat_models(
                cat, fold_tr[fold_tr["overdose_category"] == cat],
                feat_tr, fold_baselines, fold_fb, period_predictors_fold
            )
            for cat in SCORED_CATS
        }

        for cat in SCORED_CATS:
            cat_va = fold_va[fold_va["overdose_category"] == cat].copy()
            preds_ser, comp_df = predict_cat(
                cat, feat_va, fold_baselines, fold_fb, fold_models,
                period_predictors_fold, return_components=True
            )

            pred_map = preds_ser.to_dict()
            cat_va["pred"] = [
                pred_map.get((j, p), np.nan)
                for j, p in zip(cat_va["jurisdiction"], cat_va["period_id"])
            ]
            cat_va["pred"] = cat_va["pred"].fillna(fold_fb.get(cat, 0.0)).clip(lower=0)

            fold_oof_dfs.append(
                cat_va[["rate_per_10000_ed_visits", "pred"]]
                .rename(columns={"rate_per_10000_ed_visits": "actual"})
                .assign(overdose_category=cat)
            )

            comp_aligned = (
                cat_va[["jurisdiction", "period_id", "rate_per_10000_ed_visits"]]
                .merge(comp_df, on=["jurisdiction", "period_id"], how="left")
                .rename(columns={"rate_per_10000_ed_visits": "actual"})
                .assign(overdose_category=cat)
            )
            fold_comp_dfs.append(comp_aligned)

        print(f"  Fold {fold+1}/{N_CV_FOLDS} complete")

    oof_df   = pd.concat(fold_oof_dfs, ignore_index=True)
    cat_maes = {
        cat: float(np.mean(np.abs(
            oof_df.loc[oof_df["overdose_category"] == cat, "actual"]
            - oof_df.loc[oof_df["overdose_category"] == cat, "pred"]
        )))
        for cat in SCORED_CATS
    }
    block_mae = float(np.mean(list(cat_maes.values())))

    oof_saved = {"cat_maes": cat_maes, "block_mae": block_mae}
    with open(OOF_CACHE, "w") as f:
        json.dump(oof_saved, f, indent=2)

    oof_components_df = pd.concat(fold_comp_dfs, ignore_index=True)
    oof_components_df.to_parquet(OOF_COMP_CACHE, index=False)
    print(f"✓ OOF complete → {OOF_CACHE}")

print("\n=== OOF MAE (v12, initial weights) ===")
for cat in SCORED_CATS:
    print(f"  {cat:<20} {oof_saved['cat_maes'][cat]:.4f}")
print(f"  {'Block-averaged':<20} {oof_saved['block_mae']:.4f}")

WEIGHT_OPT_CACHE = GDRIVE / "optimized_weights_v12.json"

if WEIGHT_OPT_CACHE.exists():
    with open(WEIGHT_OPT_CACHE) as f:
        OPTIMIZED_WEIGHTS = json.load(f)
    print("✓ Optimized weights loaded from cache")
else:
    print("Running weight grid search over OOF components...")
    OPTIMIZED_WEIGHTS = {}
    ridge_fracs = np.arange(0.0, 1.01, 0.02)

    for cat in SCORED_CATS:
        sub = oof_components_df[oof_components_df["overdose_category"] == cat]
        y   = sub["actual"].values
        R   = sub["ridge_rate"].values
        C   = sub["corrected_rate"].values

        best_mae, best_w = np.inf, None
        for rf in ridge_fracs:
            cf   = round(1.0 - rf, 6)
            pred = rf * R + cf * C
            mae  = float(np.mean(np.abs(y - pred)))
            if mae < best_mae:
                best_mae = mae
                best_w   = {"ridge":     round(float(rf), 4),
                            "corrected": round(float(cf), 4),
                            "_oof_mae":  round(mae, 4)}

        OPTIMIZED_WEIGHTS[cat] = best_w

    with open(WEIGHT_OPT_CACHE, "w") as f:
        json.dump(OPTIMIZED_WEIGHTS, f, indent=2)
    print(f"✓ Weights saved → {WEIGHT_OPT_CACHE}")

print("\n=== Optimized Ensemble Weights ===")
for cat in SCORED_CATS:
    w = OPTIMIZED_WEIGHTS[cat]
    print(f"  {cat}: ridge={w['ridge']:.3f}  corrected={w['corrected']:.3f}"
          f"  | oof_mae={w.get('_oof_mae', '?')}")

WEIGHTS = {cat: {k: v for k, v in d.items() if not k.startswith("_")}
           for cat, d in OPTIMIZED_WEIGHTS.items()}

block_opt = float(np.mean([w["_oof_mae"] for w in OPTIMIZED_WEIGHTS.values()]))
print(f"\n  Block-averaged OOF MAE (optimized weights): {block_opt:.4f}")
print("✓ WEIGHTS updated to optimized values")

TIME_EVAL_CACHE = GDRIVE / "time_aware_eval_v12.json"

if not DATE_OF_PERIOD:
    print("period map unavailable — time-aware evaluation skipped")
    TIME_EVAL = None
elif TIME_EVAL_CACHE.exists():
    with open(TIME_EVAL_CACHE) as f:
        TIME_EVAL = json.load(f)
    print("✓ time-aware evaluation loaded from cache")
else:
    print("Running rolling-origin evaluation (7 stack fits)...")
    scored_t   = train[train["overdose_category"].isin(SCORED_CATS)].copy()
    scored_t["date"] = scored_t["period_id"].map(DATE_OF_PERIOD)
    scored_t   = scored_t.dropna(subset=["date"])
    cov_t      = train_cov.assign(date=train_cov["period_id"].map(DATE_OF_PERIOD)).dropna(subset=["date"])
    tdates     = sorted(scored_t["date"].unique())
    pivots_all = build_rate_pivots(train)

    def eval_origin(T_idx, seed):
        T_date  = tdates[T_idx]
        tr_df   = scored_t[scored_t["date"] <= T_date].drop(columns=["date"])
        fc_tr   = cov_t[cov_t["date"] <= T_date].drop(columns=["date"])
        te_dates = [tdates[T_idx + h] for h in range(2, 14) if T_idx + h < len(tdates)]
        fc_te   = simulate_missingness(
            cov_t[cov_t["date"].isin(te_dates)].drop(columns=["date"]), seed=seed)

        econ_med = {c: float(fc_tr[c].median()) for c in ECONOMIC_COLS}
        jur_tm   = compute_jur_trend_means(fc_tr)
        feat_tr  = build_features(fc_tr, compute_period_means(fc_tr), jur_trend_means=jur_tm)
        feat_te  = build_features(fc_te, compute_period_means(fc_te),
                                  econ_medians=econ_med, jur_trend_means=jur_tm)
        bl, fb   = compute_baselines(tr_df)
        pps      = {cat: fit_period_predictor(cat, tr_df, fc_tr) for cat in SCORED_CATS}
        models   = {cat: fit_cat_models(cat, tr_df[tr_df["overdose_category"] == cat],
                                        feat_tr, bl, fb, pps) for cat in SCORED_CATS}

        pivs_T = {cat: pivots_all[cat].loc[:T_date] for cat in SCORED_CATS}
        errs   = {cat: {"S1": [], "S2": [], "S1a": [], "S2a": []} for cat in SCORED_CATS}
        for cat in SCORED_CATS:
            _, comp = predict_cat(cat, feat_te, bl, fb, models, pps, return_components=True)
            stack = comp.set_index(["jurisdiction", "period_id"])["corrected_rate"]
            for h in range(2, 14):
                if T_idx + h >= len(tdates):
                    continue
                d   = tdates[T_idx + h]
                pid = scored_t.loc[scored_t["date"] == d, "period_id"].iloc[0]
                actual = pivots_all[cat].loc[d]
                sp  = stack.xs(pid, level="period_id").reindex(actual.index).values
                band = "S1" if h <= 7 else "S2"
                errs[cat][band].extend(np.abs(actual.values - np.clip(sp, 0, None)))
                a = ALPHA_LEVEL.get(cat, 0.0)
                ap = sp
                if a > 0:
                    lp = level_projection(pivs_T[cat], h).reindex(actual.index).values
                    ap = (1 - a) * sp + a * lp
                errs[cat][band + "a"].extend(np.abs(actual.values - np.clip(ap, 0, None)))
        return errs

    ORIGINS = [54, 57, 60, 63, 55, 58, 61]
    all_errs = [eval_origin(T, 42 + i) for i, T in enumerate(ORIGINS)]
    print(f"  {len(ORIGINS)} origins evaluated")

    TIME_EVAL = {}
    for cat in SCORED_CATS:
        agg = {k: float(np.mean(np.concatenate([np.asarray(e[cat][k]) for e in all_errs])))
               for k in ["S1", "S2", "S1a", "S2a"]}
        TIME_EVAL[cat] = {
            "stack":    {"S1": agg["S1"],  "S2": agg["S2"],
                         "W": 0.3 * agg["S1"] + 0.7 * agg["S2"]},
            "anchored": {"S1": agg["S1a"], "S2": agg["S2a"],
                         "W": 0.3 * agg["S1a"] + 0.7 * agg["S2a"]},
        }
    for kind in ["stack", "anchored"]:
        TIME_EVAL.setdefault("block", {})[kind] = {
            k: float(np.mean([TIME_EVAL[c][kind][k] for c in SCORED_CATS]))
            for k in ["S1", "S2", "W"]}
    with open(TIME_EVAL_CACHE, "w") as f:
        json.dump(TIME_EVAL, f, indent=2)
    print(f"✓ saved → {TIME_EVAL_CACHE}")

if TIME_EVAL:
    print("\n=== Time-aware evaluation (0.3·S1 + 0.7·S2) ===")
    print(f"  {'':<18}{'stack W':>10}{'anchored W':>12}")
    for cat in SCORED_CATS:
        print(f"  {cat:<18}{TIME_EVAL[cat]['stack']['W']:>10.4f}"
              f"{TIME_EVAL[cat]['anchored']['W']:>12.4f}")
    b = TIME_EVAL["block"]
    print(f"  {'block':<18}{b['stack']['W']:>10.4f}{b['anchored']['W']:>12.4f}")

MODEL_CACHE = GDRIVE / "final_models_v12.pkl"

if MODEL_CACHE.exists():
    with open(MODEL_CACHE, "rb") as f:
        saved = pickle.load(f)
    final_models      = saved["models"]
    final_baselines   = saved["baselines"]
    final_fb          = saved["global_fb"]
    PERIOD_PREDICTORS = saved["period_predictors"]
    print("✓ Final models loaded from cache")
else:
    print("Fitting final models on all training data...")
    scored_all = train[train["overdose_category"].isin(SCORED_CATS)]
    final_baselines, final_fb = compute_baselines(scored_all)

    PERIOD_PREDICTORS = {
        cat: fit_period_predictor(cat, scored_all, train_cov)
        for cat in SCORED_CATS
    }

    feat_tr = build_features(train_cov, ALL_PM,
                              jur_trend_means=JUR_TREND_MEANS_TRAIN)

    final_models = {}
    for cat in SCORED_CATS:
        final_models[cat] = fit_cat_models(
            cat, scored_all[scored_all["overdose_category"] == cat],
            feat_tr, final_baselines, final_fb, PERIOD_PREDICTORS
        )
        m  = final_models[cat]
        pp = PERIOD_PREDICTORS[cat]
        print(f"  [{cat}] {m['n_features']} features  "
              f"eps={m['epsilon']:.2f}  alpha={m['alpha']:.4g}  bias={m['bias']:+.4f}  "
              f"factor_range=[{pp['factor_lo']:.3f}, {pp['factor_hi']:.3f}]  "
              f"period_R²={pp['r2_log']:.3f}")

    with open(MODEL_CACHE, "wb") as f:
        pickle.dump({"models": final_models, "baselines": final_baselines,
                     "global_fb": final_fb, "period_predictors": PERIOD_PREDICTORS}, f)
    print(f"✓ Final models saved → {MODEL_CACHE}")

val_feat = build_features(
    val_cov, ALL_PM,
    econ_medians=ECON_MEDIANS_TRAIN,
    jur_trend_means=JUR_TREND_MEANS_TRAIN
)

print("Generating predictions...")
val_keys = val_feat[["jurisdiction", "period_id"]].drop_duplicates().reset_index(drop=True)

for cat in SCORED_CATS:
    preds    = predict_cat(cat, val_feat, final_baselines, final_fb,
                           final_models, PERIOD_PREDICTORS)
    pred_map = preds.to_dict()
    val_keys[cat] = [
        pred_map.get((j, p), np.nan)
        for j, p in zip(val_keys["jurisdiction"], val_keys["period_id"])
    ]
    print(f"  {cat:<20} mean={val_keys[cat].mean():.2f}  "
          f"min={val_keys[cat].min():.2f}  max={val_keys[cat].max():.2f}")

print("✓ Raw stack predictions complete")

# Layer 4b: temporal anchor (v12)
print("\nApplying temporal anchor (Layer 4b)...")
PIVOTS_FULL = build_rate_pivots(train)
val_keys = apply_temporal_anchor(val_keys, PIVOTS_FULL)
for cat in SCORED_CATS:
    print(f"  {cat:<20} mean={val_keys[cat].mean():.2f}  "
          f"min={val_keys[cat].min():.2f}  max={val_keys[cat].max():.2f}")

for sub_cat in ["all_opioids", "all_stimulants"]:
    n_viol = (val_keys["all_drugs"] < val_keys[sub_cat] - 1e-9).sum()
    val_keys["all_drugs"] = np.maximum(val_keys["all_drugs"], val_keys[sub_cat])
    print(f"  Nesting all_drugs >= {sub_cat}: {n_viol} violations corrected")

for cat in SCORED_CATS:
    n_neg = (val_keys[cat] < 0).sum()
    val_keys[cat] = val_keys[cat].clip(lower=0.0)
    if n_neg:
        print(f"  Floor clip {cat}: {n_neg} negatives → 0")

print("\n✓ Post-processing complete")
print(val_keys[SCORED_CATS].describe().round(2))

long = val_keys.melt(
    id_vars=["jurisdiction", "period_id"],
    value_vars=SCORED_CATS,
    var_name="overdose_category",
    value_name="pred"
)

out = submission.merge(long, on=["jurisdiction", "period_id", "overdose_category"], how="left")

for cat in SCORED_CATS:
    base_map = (final_baselines[final_baselines["overdose_category"] == cat]
                .set_index("jurisdiction")["baseline_mean"])
    mask = out["pred"].isna() & (out["overdose_category"] == cat)
    if mask.any():
        out.loc[mask, "pred"] = out.loc[mask, "jurisdiction"].map(base_map).fillna(final_fb.get(cat, 0))

out["rate_per_10000_ed_visits"] = out["pred"].fillna(0.0).clip(lower=0.0)
final_sub = out[["row_id", "rate_per_10000_ed_visits"]].copy()

# RULES.md: every row_id from sample_submission, no more, no fewer — derived at
# runtime, never hardcoded (Stage 2 re-execution changes the row count).
assert final_sub.shape[0] == N_SUB_ROWS
assert list(final_sub.columns) == ["row_id", "rate_per_10000_ed_visits"]
assert final_sub["rate_per_10000_ed_visits"].notna().all()
assert np.isfinite(final_sub["rate_per_10000_ed_visits"]).all()
assert (final_sub["rate_per_10000_ed_visits"] >= 0).all()
assert set(final_sub["row_id"]) == set(submission["row_id"])

print("✓ All submission validation checks passed")
print(f"  Shape : {final_sub.shape}")
print(f"  Range : [{final_sub['rate_per_10000_ed_visits'].min():.4f}, "
      f"{final_sub['rate_per_10000_ed_visits'].max():.4f}]")
print(f"  Mean  : {final_sub['rate_per_10000_ed_visits'].mean():.4f}")

cat_maes_opt  = {cat: OPTIMIZED_WEIGHTS[cat]["_oof_mae"] for cat in SCORED_CATS}
block_mae_opt = float(np.mean(list(cat_maes_opt.values())))

oof_lines = [
    f"{cat_maes_opt['all_drugs']:.4f}",
    f"{cat_maes_opt['all_opioids']:.4f}",
    f"{cat_maes_opt['all_stimulants']:.4f}",
    f"{block_mae_opt:.4f}",
]

for dest in [Path("expert_william_v12.csv"), GDRIVE / "expert_william_v12.csv"]:
    final_sub.to_csv(dest, index=False)
for dest in [Path("oof_mae_william_v12.txt"), GDRIVE / "oof_mae_william_v12.txt"]:
    dest.write_text("\n".join(oof_lines) + "\n")

if TIME_EVAL:
    ta_lines = [f"{TIME_EVAL[cat]['anchored']['W']:.4f}" for cat in SCORED_CATS]
    ta_lines.append(f"{TIME_EVAL['block']['anchored']['W']:.4f}")
    for dest in [Path("oof_time_aware_william_v12.txt"), GDRIVE / "oof_time_aware_william_v12.txt"]:
        dest.write_text("\n".join(ta_lines) + "\n")
    print("✓ oof_time_aware_william_v12.txt written (weighted 0.3·S1+0.7·S2, anchored)")

print("✓ expert_william_v12.csv written (local + Drive)")
print("✓ oof_mae_william_v12.txt written (local + Drive, GroupKFold OOF — legacy comparability)")

# RULES.md mandatory submission pattern: re-read sample_submission, assign the
# predictions aligned to its row_ids, write the two required columns.
sub = pd.read_csv(SUBMISSION)
pred_map = final_sub.set_index("row_id")["rate_per_10000_ed_visits"]
sub["rate_per_10000_ed_visits"] = sub["row_id"].map(pred_map)
assert pd.api.types.is_integer_dtype(sub["row_id"])           # row_id (int)
assert sub["rate_per_10000_ed_visits"].notna().all()
assert np.isfinite(sub["rate_per_10000_ed_visits"]).all()     # float, finite
out_path = Path("/kaggle/working/submission.csv") if IS_KAGGLE else Path("submission.csv")
sub[["row_id", "rate_per_10000_ed_visits"]].to_csv(out_path, index=False)
print(f"✓ {out_path} written")
print("\n=== OOF MAE Summary (v12, optimized weights) ===")
print(f"  all_drugs      {cat_maes_opt['all_drugs']:.4f}")
print(f"  all_opioids    {cat_maes_opt['all_opioids']:.4f}")
print(f"  all_stimulants {cat_maes_opt['all_stimulants']:.4f}")
print(f"  block-avg      {block_mae_opt:.4f}")
print(f"\n  (initial-weight OOF: {oof_saved['block_mae']:.4f})")
print("\n✅ v12 pipeline complete.")

# ── MoE integration: emit the standard expert filename ──
final_sub.to_csv("expert_william.csv", index=False)
print("\u2713 expert_william.csv written (MoE standard name)")
