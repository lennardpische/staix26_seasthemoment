"""Feature engineering — transformer expert (pipeline 2).

Produces three modality arrays per split (numeric, text, image) and, for train,
a targets array with the three scoring-category rates.  Everything is fit on
train and applied to val; no leakage.

The feature matrix is pivoted wide: one row per (period_id × jurisdiction).
The three target values become columns [all_drugs, all_opioids, all_stimulants]
and NaN cells represent suppressed rows that are masked out during training loss.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer

CENSUS_DIVISION = {
    "CT": 1, "ME": 1, "MA": 1, "NH": 1, "RI": 1, "VT": 1,
    "NJ": 2, "NY": 2, "PA": 2,
    "IL": 3, "IN": 3, "MI": 3, "OH": 3, "WI": 3,
    "IA": 4, "KS": 4, "MN": 4, "MO": 4, "NE": 4, "ND": 4, "SD": 4,
    "DC": 5, "DE": 5, "FL": 5, "GA": 5, "MD": 5, "NC": 5, "SC": 5, "VA": 5, "WV": 5,
    "AL": 6, "KY": 6, "MS": 6, "TN": 6,
    "AR": 7, "LA": 7, "OK": 7, "TX": 7,
    "AZ": 8, "CO": 8, "ID": 8, "MT": 8, "NV": 8, "NM": 8, "UT": 8, "WY": 8,
    "AK": 9, "CA": 9, "HI": 9, "OR": 9, "WA": 9,
}

SCORING_CATEGORIES = ["all_drugs", "all_opioids", "all_stimulants"]
AUX_CATEGORIES = ["heroin", "fentanyl", "cocaine", "methamphetamine", "benzodiazepine"]

_RAW_NUMERICS = [
    "unemployment_rate", "temp_avg_f", "precip_in",
    "gtrends_overdose", "gtrends_fentanyl", "gtrends_naloxone",
    "gtrends_opioid", "gtrends_methamphetamine",
]
_ENGINEERED = [
    "log_labor_force", "census_division", "period_rank",
    "gtrends_fentanyl_x_opioid", "gtrends_overdose_x_naloxone",
    # Real date features — populated when period_id_map.json is present
    "year", "month", "quarter", "month_sin", "month_cos", "months_since_covid",
]

N_TEXT = 32
N_IMAGE = 64
IMG_SIZE = 32  # resize to 32×32 RGB before PCA


def _load_cached_embeddings(path: str, keys: pd.DataFrame) -> np.ndarray | None:
    """Load pre-computed embeddings from a CSV saved by src/vectorize.py.

    Joins on (period_id, jurisdiction) so row order always matches keys,
    regardless of the order they were saved. Returns None if file missing.
    """
    p = Path(path)
    if not p.exists():
        return None
    df = pd.read_csv(p)
    emb_cols = [c for c in df.columns if c not in ("period_id", "jurisdiction")]
    merged = keys[["period_id", "jurisdiction"]].merge(df, on=["period_id", "jurisdiction"], how="left")
    result = merged[emb_cols].values.astype(np.float32)
    if np.isnan(result).any():
        # Some rows missing from cache — can't use it
        return None
    return result


def _base_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["census_division"] = out["jurisdiction"].map(CENSUS_DIVISION).fillna(0).astype(int)
    out["log_labor_force"] = np.log1p(out["labor_force"].clip(lower=0).fillna(0))
    out["gtrends_fentanyl_x_opioid"] = out["gtrends_fentanyl"] * out["gtrends_opioid"]
    out["gtrends_overdose_x_naloxone"] = out["gtrends_overdose"] * out["gtrends_naloxone"]
    return out


def _lag_lookup_for(train_long: pd.DataFrame, category: str) -> pd.DataFrame:
    """Build (jurisdiction, period_rank) → (lag1, lag2) from the training target."""
    prefix = category.replace("all_", "")
    t = (
        train_long[train_long["overdose_category"] == category]
        [["jurisdiction", "period_rank", "rate_per_10000_ed_visits"]]
        .sort_values(["jurisdiction", "period_rank"])
        .copy()
    )
    t["lag_1"] = t.groupby("jurisdiction")["rate_per_10000_ed_visits"].shift(1)
    t["lag_2"] = t.groupby("jurisdiction")["rate_per_10000_ed_visits"].shift(2)
    return (
        t[["jurisdiction", "period_rank", "lag_1", "lag_2"]]
        .rename(columns={"lag_1": f"{prefix}_lag1", "lag_2": f"{prefix}_lag2"})
    )


def _attach_lags(df: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    """Backward-asof merge: attach lag values from the most recent period_rank
    that is <= the row's own period_rank, per jurisdiction.

    Uses index-aligned assignment so the result stays in df's original row order.
    """
    lag_cols = [c for c in lookup.columns if c not in ("jurisdiction", "period_rank")]
    ll = lookup.sort_values("period_rank")
    df_sorted = df[["jurisdiction", "period_rank"]].sort_values("period_rank")
    merged = pd.merge_asof(
        df_sorted, ll, on="period_rank", by="jurisdiction", direction="backward",
    )
    result = df.copy()
    for c in lag_cols:
        result[c] = merged[c]  # pandas index-aligns; no positional mismatch
    return result


def _pivot_targets(train_long: pd.DataFrame) -> pd.DataFrame:
    sc = train_long[train_long["overdose_category"].isin(SCORING_CATEGORIES)]
    pivot = sc.pivot_table(
        index=["period_id", "jurisdiction"],
        columns="overdose_category",
        values="rate_per_10000_ed_visits",
        aggfunc="first",
    ).reset_index()
    pivot.columns.name = None
    for c in SCORING_CATEGORIES:
        if c not in pivot.columns:
            pivot[c] = np.nan
    return pivot[["period_id", "jurisdiction"] + SCORING_CATEGORIES]


def _pivot_aux(train_long: pd.DataFrame) -> pd.DataFrame:
    aux = train_long[train_long["overdose_category"].isin(AUX_CATEGORIES)]
    if aux.empty:
        return pd.DataFrame(columns=["period_id", "jurisdiction"])
    pivot = aux.pivot_table(
        index=["period_id", "jurisdiction"],
        columns="overdose_category",
        values="rate_per_10000_ed_visits",
        aggfunc="first",
    ).reset_index()
    pivot.columns.name = None
    return pivot.rename(columns={c: f"aux_{c}" for c in AUX_CATEGORIES if c in pivot.columns})


class FeaturePipeline:
    """Fit on train_long; produce numeric/text/image arrays for transformer."""

    def __init__(self, n_text: int = N_TEXT, n_image: int = N_IMAGE):
        self.n_text = n_text
        self.n_image = n_image
        self.tfidf: TfidfVectorizer | None = None
        self.text_svd: TruncatedSVD | None = None
        self.img_pca: PCA | None = None
        self.num_imputer: SimpleImputer | None = None
        self.numeric_cols: list[str] = []
        self._lag_lookups: list[pd.DataFrame] = []

    def fit_transform(
        self,
        train_long: pd.DataFrame,
        val_cov: pd.DataFrame,
        data_root: Path | None = None,
    ) -> tuple[dict, dict]:
        """Return (train_data, val_data) dicts.

        Each dict contains:
            numeric  ndarray (N, n_num)   float32
            text     ndarray (N, n_text)  float32
            image    ndarray (N, n_img)   float32
            targets  ndarray (N, 3)       float32  — val has all-NaN
            keys     DataFrame            period_id, jurisdiction, period_rank
        """
        KEY = ["period_id", "jurisdiction"]
        COV_COLS = KEY + [
            "unemployment_rate", "labor_force", "temp_avg_f", "precip_in",
            "gtrends_overdose", "gtrends_fentanyl", "gtrends_naloxone",
            "gtrends_opioid", "gtrends_methamphetamine",
            "state_doh_release", "period_rank",
            # Date features from period_id_map.json (NaN when map absent)
            "year", "month", "quarter", "month_sin", "month_cos", "months_since_covid",
        ]

        # One row per (period_id × jurisdiction) — covariates are identical across categories
        tc = (
            train_long
            .drop_duplicates(subset=KEY)
            [[c for c in COV_COLS if c in train_long.columns]]
            .reset_index(drop=True)
        )
        vc = val_cov[[c for c in COV_COLS if c in val_cov.columns]].copy().reset_index(drop=True)
        for c in COV_COLS:
            if c not in vc.columns:
                vc[c] = np.nan

        tc = _base_features(tc)
        vc = _base_features(vc)

        # Targets (wide)
        targets_wide = _pivot_targets(train_long)
        tc = tc.merge(targets_wide, on=KEY, how="left")

        # Aux category rates as extra covariates
        aux_pivot = _pivot_aux(train_long)
        aux_cols = [c for c in aux_pivot.columns if c.startswith("aux_")]
        if aux_cols:
            tc = tc.merge(aux_pivot, on=KEY, how="left")
            vc = vc.merge(aux_pivot, on=KEY, how="left")

        # Jurisdiction historical mean + std per scoring category (fit on train only)
        jur_stats = (
            train_long[train_long["overdose_category"].isin(SCORING_CATEGORIES)]
            .dropna(subset=["rate_per_10000_ed_visits"])
            .groupby(["jurisdiction", "overdose_category"])["rate_per_10000_ed_visits"]
            .agg(["mean", "std"])
            .reset_index()
        )
        jur_pivot = jur_stats.pivot(index="jurisdiction", columns="overdose_category",
                                    values=["mean", "std"])
        jur_pivot.columns = [f"jur_{stat}_{cat}" for stat, cat in jur_pivot.columns]
        jur_pivot = jur_pivot.reset_index()
        jur_cols = [c for c in jur_pivot.columns if c != "jurisdiction"]
        tc = tc.merge(jur_pivot, on="jurisdiction", how="left")
        vc = vc.merge(jur_pivot, on="jurisdiction", how="left")

        # Lag features
        lag_cols: list[str] = []
        if train_long["period_rank"].nunique() > 1:
            for cat in SCORING_CATEGORIES:
                ll = _lag_lookup_for(train_long, cat)
                self._lag_lookups.append(ll)
                tc = _attach_lags(tc, ll)
                vc = _attach_lags(vc, ll)
                prefix = cat.replace("all_", "")
                lag_cols += [f"{prefix}_lag1", f"{prefix}_lag2"]

        self.numeric_cols = _RAW_NUMERICS + _ENGINEERED + aux_cols + jur_cols + lag_cols

        # Numeric imputation
        num_tr = tc[self.numeric_cols].values.astype(np.float32)
        num_va = vc[self.numeric_cols].values.astype(np.float32)
        self.num_imputer = SimpleImputer(strategy="median")
        num_tr = self.num_imputer.fit_transform(num_tr).astype(np.float32)
        num_va = self.num_imputer.transform(num_va).astype(np.float32)

        # Text: TF-IDF + TruncatedSVD (fit on train only)
        tr_text = tc["state_doh_release"].fillna("").tolist()
        va_text = vc["state_doh_release"].fillna("").tolist()
        self.tfidf = TfidfVectorizer(max_features=5000, sublinear_tf=True, ngram_range=(1, 2))
        tr_tfidf = self.tfidf.fit_transform(tr_text)
        va_tfidf = self.tfidf.transform(va_text)
        n_comp = min(self.n_text, tr_tfidf.shape[1] - 1, tr_tfidf.shape[0] - 1)
        self.text_svd = TruncatedSVD(n_components=n_comp, random_state=42)
        text_tr = self.text_svd.fit_transform(tr_tfidf).astype(np.float32)
        text_va = self.text_svd.transform(va_tfidf).astype(np.float32)

        # Text: try cached sentence-transformer embeddings first, fall back to TF-IDF+SVD
        cache_text_tr = _load_cached_embeddings("embeddings/text_train.csv", tc[KEY])
        cache_text_va = _load_cached_embeddings("embeddings/text_val.csv", vc[KEY])
        if cache_text_tr is not None and cache_text_va is not None:
            text_tr = cache_text_tr.astype(np.float32)
            text_va = cache_text_va.astype(np.float32)
        # (TF-IDF path already ran above and set text_tr / text_va — override if cache hit)

        # Image: try cached EfficientNet embeddings first, fall back to pixel PCA
        cache_img_tr = _load_cached_embeddings("embeddings/img_train.csv", tc[KEY])
        cache_img_va = _load_cached_embeddings("embeddings/img_val.csv", vc[KEY])
        if cache_img_tr is not None and cache_img_va is not None:
            img_tr = cache_img_tr.astype(np.float32)
            img_va = cache_img_va.astype(np.float32)
        elif data_root is not None:
            from .data_loader import load_images_for
            train_images = load_images_for(data_root, "train", tc[["jurisdiction", "period_id"]], IMG_SIZE)
            val_images = load_images_for(data_root, "val", vc[["jurisdiction", "period_id"]], IMG_SIZE)
            n_ic = min(self.n_image, train_images.shape[1] - 1, len(train_images) - 1)
            self.img_pca = PCA(n_components=n_ic, random_state=42)
            img_tr = self.img_pca.fit_transform(train_images).astype(np.float32)
            img_va = self.img_pca.transform(val_images).astype(np.float32)
        else:
            img_tr = np.zeros((len(tc), self.n_image), dtype=np.float32)
            img_va = np.zeros((len(vc), self.n_image), dtype=np.float32)

        train_keys = tc[KEY + ["period_rank"]].reset_index(drop=True)
        val_keys = vc[KEY + ["period_rank"]].reset_index(drop=True)
        targets = tc[SCORING_CATEGORIES].values.astype(np.float32)

        train_data = dict(numeric=num_tr, text=text_tr, image=img_tr, targets=targets, keys=train_keys)
        val_data = dict(
            numeric=num_va, text=text_va, image=img_va,
            targets=np.full((len(vc), 3), np.nan, dtype=np.float32),
            keys=val_keys,
        )
        return train_data, val_data

    @property
    def n_numeric(self) -> int:
        return len(self.numeric_cols)

    @property
    def n_text_out(self) -> int:
        return self.text_svd.n_components if self.text_svd else self.n_text

    @property
    def n_image_out(self) -> int:
        return int(self.img_pca.n_components_) if self.img_pca else self.n_image
