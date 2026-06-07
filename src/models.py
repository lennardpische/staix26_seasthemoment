"""FT-Transformer model — STAI-X 2026 transformer expert (pipeline 2).

Architecture: CLS-token transformer with three modality projection tokens
(numeric, text, image) and three category-specific regression heads.
Falls back to sklearn MLPRegressor when PyTorch is unavailable or the
90-minute wall-clock budget is exhausted.
"""

from __future__ import annotations

import time
import warnings
import numpy as np
from sklearn.model_selection import GroupKFold

CATEGORIES = ["all_drugs", "all_opioids", "all_stimulants"]
N_FOLDS = 5
RANDOM_STATE = 42

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    warnings.warn("PyTorch not found — transformer expert will use MLP fallback.", stacklevel=1)


# ── FT-Transformer ──────────────────────────────────────────────────────────

if _HAS_TORCH:

    class _ModalityProj(nn.Module):
        def __init__(self, in_dim: int, d_model: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
            )

        def forward(self, x):
            return self.net(x)

    class FTTransformer(nn.Module):
        """CLS + 3 modality tokens → transformer encoder → 3 regression heads."""

        def __init__(
            self,
            n_num: int,
            n_text: int,
            n_img: int,
            d_model: int = 128,
            n_heads: int = 4,
            n_layers: int = 2,
            d_ff: int = 256,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.cls = nn.Parameter(torch.empty(1, 1, d_model))
            nn.init.normal_(self.cls, std=0.02)
            self.proj_num = _ModalityProj(n_num, d_model)
            self.proj_text = _ModalityProj(n_text, d_model)
            self.proj_img = _ModalityProj(n_img, d_model)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
                dropout=dropout, batch_first=True, activation="gelu",
            )
            self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
            self.drop = nn.Dropout(dropout)
            self.heads = nn.ModuleList([nn.Linear(d_model, 1) for _ in CATEGORIES])

        def forward(self, xn, xt, xi):
            B = xn.size(0)
            tokens = torch.cat([
                self.cls.expand(B, -1, -1),
                self.proj_num(xn).unsqueeze(1),
                self.proj_text(xt).unsqueeze(1),
                self.proj_img(xi).unsqueeze(1),
            ], dim=1)                               # (B, 4, d_model)
            z = self.drop(self.enc(tokens)[:, 0])   # CLS output
            return torch.cat([h(z) for h in self.heads], dim=1)  # (B, 3)

    def _masked_huber(pred, target, delta: float = 1.0):
        """Huber loss averaged over non-NaN (pred, target) pairs."""
        mask = ~torch.isnan(target)
        if not mask.any():
            return pred.sum() * 0.0
        p, t = pred[mask], target[mask]
        err = torch.abs(p - t)
        return torch.where(err <= delta, 0.5 * err ** 2, delta * (err - 0.5 * delta)).mean()

    def _fit_fold(
        Xn: np.ndarray, Xt: np.ndarray, Xi: np.ndarray, Y: np.ndarray,
        tr_idx: np.ndarray, va_idx: np.ndarray,
        epochs: int = 100, patience: int = 15, lr: float = 3e-4, batch: int = 64,
    ) -> tuple[np.ndarray, "FTTransformer"]:
        def _t(arr, idx):
            return torch.tensor(arr[idx], dtype=torch.float32)

        Xn_tr, Xt_tr, Xi_tr, Y_tr = _t(Xn, tr_idx), _t(Xt, tr_idx), _t(Xi, tr_idx), _t(Y, tr_idx)
        Xn_va, Xt_va, Xi_va = _t(Xn, va_idx), _t(Xt, va_idx), _t(Xi, va_idx)
        Y_va = torch.tensor(Y[va_idx], dtype=torch.float32)

        loader = DataLoader(TensorDataset(Xn_tr, Xt_tr, Xi_tr, Y_tr), batch_size=batch, shuffle=True)

        model = FTTransformer(n_num=Xn.shape[1], n_text=Xt.shape[1], n_img=Xi.shape[1])
        opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        best_loss, best_w, no_imp = float("inf"), None, 0
        for _ in range(epochs):
            model.train()
            for xn, xt, xi, yb in loader:
                opt.zero_grad()
                _masked_huber(model(xn, xt, xi), yb).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()

            model.eval()
            with torch.no_grad():
                vl = _masked_huber(model(Xn_va, Xt_va, Xi_va), Y_va).item()
            if vl < best_loss - 1e-6:
                best_loss = vl
                best_w = {k: v.clone() for k, v in model.state_dict().items()}
                no_imp = 0
            else:
                no_imp += 1
                if no_imp >= patience:
                    break

        if best_w:
            model.load_state_dict(best_w)
        model.eval()
        with torch.no_grad():
            oof = model(Xn_va, Xt_va, Xi_va).cpu().numpy()
        return oof, model


# ── MLP fallback ─────────────────────────────────────────────────────────────

def _fit_fold_mlp(
    Xn: np.ndarray, Xt: np.ndarray, Xi: np.ndarray, Y: np.ndarray,
    tr_idx: np.ndarray, va_idx: np.ndarray,
    Xn_v: np.ndarray, Xt_v: np.ndarray, Xi_v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.neural_network import MLPRegressor

    X_tr = np.concatenate([Xn[tr_idx], Xt[tr_idx], Xi[tr_idx]], axis=1)
    X_va = np.concatenate([Xn[va_idx], Xt[va_idx], Xi[va_idx]], axis=1)
    X_val = np.concatenate([Xn_v, Xt_v, Xi_v], axis=1)

    oof = np.full((len(va_idx), 3), np.nan, dtype=np.float32)
    val_preds = np.zeros((len(Xn_v), 3), dtype=np.float32)

    for i in range(3):
        mask = ~np.isnan(Y[tr_idx, i])
        if mask.sum() < 5:
            continue
        m = MLPRegressor(hidden_layer_sizes=(256, 256), max_iter=200, random_state=RANDOM_STATE)
        m.fit(X_tr[mask], Y[tr_idx][mask, i])
        oof[:, i] = m.predict(X_va)
        val_preds[:, i] = m.predict(X_val)

    return oof, val_preds


# ── Public API ────────────────────────────────────────────────────────────────

def train_and_predict(
    train_data: dict,
    val_data: dict,
    verbose: bool = True,
    deadline_secs: float = 90 * 60,
) -> tuple[np.ndarray, np.ndarray]:
    """GroupKFold training loop.

    Returns:
        val_preds  (N_val, 3)   — ensemble of fold predictions, clipped ≥ 0
        oof_preds  (N_train, 3) — out-of-fold predictions for CV evaluation
    """
    t0 = time.time()

    Xn, Xt, Xi = train_data["numeric"], train_data["text"], train_data["image"]
    Y = train_data["targets"]
    groups = train_data["keys"]["period_rank"].fillna(0).astype(int).values
    Xn_v, Xt_v, Xi_v = val_data["numeric"], val_data["text"], val_data["image"]

    gkf = GroupKFold(n_splits=N_FOLDS)
    oof = np.full_like(Y, np.nan)
    val_fold_preds: list[np.ndarray] = []
    use_torch = _HAS_TORCH

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(Xn, Y, groups)):
        if verbose:
            elapsed = time.time() - t0
            print(f"  Fold {fold + 1}/{N_FOLDS}  ({elapsed:.0f}s elapsed) ...", end=" ", flush=True)

        use_fallback = (not use_torch) or (time.time() - t0 >= deadline_secs)

        if not use_fallback:
            try:
                oof_va, model = _fit_fold(Xn, Xt, Xi, Y, tr_idx, va_idx)
                oof[va_idx] = oof_va
                model.eval()
                with torch.no_grad():
                    vp = model(
                        torch.tensor(Xn_v, dtype=torch.float32),
                        torch.tensor(Xt_v, dtype=torch.float32),
                        torch.tensor(Xi_v, dtype=torch.float32),
                    ).cpu().numpy()
                val_fold_preds.append(vp)
                if verbose:
                    print("transformer", flush=True)
                continue
            except Exception as exc:
                warnings.warn(f"Fold {fold + 1} transformer failed ({exc!r}); switching to MLP.")
                use_torch = False

        oof_va, vp = _fit_fold_mlp(Xn, Xt, Xi, Y, tr_idx, va_idx, Xn_v, Xt_v, Xi_v)
        oof[va_idx] = oof_va
        val_fold_preds.append(vp)
        if verbose:
            print("MLP fallback", flush=True)

    val_preds = np.clip(np.nanmean(val_fold_preds, axis=0), 0, None)

    if verbose:
        maes = []
        for i, cat in enumerate(CATEGORIES):
            mask = ~np.isnan(Y[:, i]) & ~np.isnan(oof[:, i])
            if mask.sum() > 0:
                mae = float(np.abs(oof[mask, i] - Y[mask, i]).mean())
                maes.append(mae)
                print(f"  OOF MAE [{cat}]: {mae:.4f}")
        if maes:
            print(f"  Mean OOF MAE: {np.mean(maes):.4f}")

    return val_preds, oof
