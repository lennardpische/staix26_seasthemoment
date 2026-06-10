"""FT-Transformer — STAI-X 2026 transformer expert (pipeline 2).

A100 / GPU-optimised build:
  • d_model = 256, n_heads = 8, n_layers = 6, d_ff = 1024
  • Pre-norm (norm_first) for stable deep training
  • BF16 mixed precision on CUDA, fp32 fallback on CPU
  • CosineAnnealingWarmRestarts scheduler (restarts at 100 / 200 / 400 epochs)
  • Gaussian noise augmentation on numeric features during training
  • MLP fallback when PyTorch unavailable or 90-min wall-clock hit

Expects pre-vectorized embeddings from src/vectorize.py for best quality;
works with TF-IDF/PCA fallback features from src/features.py otherwise.
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
    warnings.warn("PyTorch not found — using MLP fallback.", stacklevel=1)


# ── Architecture ─────────────────────────────────────────────────────────────

if _HAS_TORCH:

    class _ModalityProj(nn.Module):
        """Project one modality vector into the shared d_model space."""
        def __init__(self, in_dim: int, d_model: int, dropout: float = 0.1):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, d_model * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model),
                nn.LayerNorm(d_model),
            )

        def forward(self, x):
            return self.net(x)


    class FTTransformer(nn.Module):
        """CLS + 3 modality tokens → 6-layer pre-norm transformer → 3 regression heads.

        Architecture per PIPELINE_TRANSFORMER.md (A100 build):
          4 tokens: [CLS, numeric, text, image]
          6 encoder layers, d_model=256, 8 heads, d_ff=1024
          Pre-norm (norm_first=True) for stable deep training
          Outputs: (batch, 3) — one scalar per scoring category
        """

        def __init__(
            self,
            n_num: int,
            n_text: int,
            n_img: int,
            d_model: int = 256,
            n_heads: int = 8,
            n_layers: int = 6,
            d_ff: int = 1024,
            dropout: float = 0.2,
        ):
            super().__init__()
            self.cls = nn.Parameter(torch.empty(1, 1, d_model))
            nn.init.trunc_normal_(self.cls, std=0.02)

            self.proj_num  = _ModalityProj(n_num,   d_model, dropout)
            self.proj_text = _ModalityProj(n_text,  d_model, dropout)
            self.proj_img  = _ModalityProj(n_img,   d_model, dropout)

            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,   # pre-norm: more stable at depth ≥ 4
            )
            self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers,
                                             enable_nested_tensor=False)
            self.post_norm = nn.LayerNorm(d_model)
            self.drop = nn.Dropout(dropout)
            self.heads = nn.ModuleList([nn.Linear(d_model, 1) for _ in CATEGORIES])

        def forward(self, xn, xt, xi):
            B = xn.size(0)
            tokens = torch.cat([
                self.cls.expand(B, -1, -1),
                self.proj_num(xn).unsqueeze(1),
                self.proj_text(xt).unsqueeze(1),
                self.proj_img(xi).unsqueeze(1),
            ], dim=1)                                        # (B, 4, d_model)
            z = self.post_norm(self.enc(tokens)[:, 0])      # CLS after encoder
            z = self.drop(z)
            return torch.cat([h(z) for h in self.heads], dim=1)  # (B, 3)


    # ── Loss & augmentation ──────────────────────────────────────────────────

    # Head weights: all_drugs gets 2× because it dominates block-MAE and has 4× the variance
    _HEAD_WEIGHTS = [2.0, 1.0, 1.0]

    def _masked_huber(pred, target, delta: float = 1.0):
        """Weighted Huber loss per head, NaN-masked. all_drugs weighted 2×."""
        total = pred.sum() * 0.0
        for i, w in enumerate(_HEAD_WEIGHTS):
            mask = ~torch.isnan(target[:, i])
            if not mask.any():
                continue
            p, t = pred[mask, i], target[mask, i]
            err = torch.abs(p - t)
            head_loss = torch.where(err <= delta, 0.5 * err ** 2, delta * (err - 0.5 * delta)).mean()
            total = total + w * head_loss
        return total


    def _add_noise(x: torch.Tensor, sigma: float = 0.02) -> torch.Tensor:
        """Gaussian noise augmentation for numeric features during training."""
        return x + torch.randn_like(x) * sigma


    # ── Single fold training ─────────────────────────────────────────────────

    def _fit_fold(
        Xn: np.ndarray, Xt: np.ndarray, Xi: np.ndarray, Y: np.ndarray,
        tr_idx: np.ndarray, va_idx: np.ndarray,
        device: str,
        epochs: int = 500,
        patience: int = 50,
        lr: float = 3e-4,
        weight_decay: float = 1e-3,
        batch: int = 128,
        noise_sigma: float = 0.02,
    ) -> tuple[np.ndarray, "FTTransformer"]:

        amp_dtype = torch.bfloat16 if device == "cuda" else torch.float32
        use_amp = device == "cuda"

        def _t(arr, idx=None):
            a = arr if idx is None else arr[idx]
            return torch.tensor(a, dtype=torch.float32).to(device)

        Xn_tr, Xt_tr, Xi_tr, Y_tr = _t(Xn, tr_idx), _t(Xt, tr_idx), _t(Xi, tr_idx), _t(Y, tr_idx)
        Xn_va, Xt_va, Xi_va, Y_va = _t(Xn, va_idx), _t(Xt, va_idx), _t(Xi, va_idx), _t(Y, va_idx)

        loader = DataLoader(
            TensorDataset(Xn_tr, Xt_tr, Xi_tr, Y_tr),
            batch_size=batch, shuffle=True, drop_last=False,
        )

        model = FTTransformer(n_num=Xn.shape[1], n_text=Xt.shape[1], n_img=Xi.shape[1]).to(device)
        opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        # Warm restarts: epoch 100 → 200 → 400 (T_mult=2 doubles the period each restart)
        sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=100, T_mult=2, eta_min=lr * 0.01)
        scaler = torch.amp.GradScaler(enabled=use_amp)

        best_loss, best_w, no_imp = float("inf"), None, 0

        for epoch in range(epochs):
            model.train()
            for xn, xt, xi, yb in loader:
                opt.zero_grad()
                with torch.amp.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                    loss = _masked_huber(model(_add_noise(xn, noise_sigma), xt, xi), yb)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            sched.step()

            model.eval()
            with torch.no_grad(), torch.amp.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                vl = _masked_huber(model(Xn_va, Xt_va, Xi_va), Y_va).item()

            if vl < best_loss - 1e-6:
                best_loss, best_w, no_imp = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
            else:
                no_imp += 1
                if no_imp >= patience:
                    break

        if best_w:
            model.load_state_dict(best_w)
        model.eval()
        with torch.no_grad():
            oof = model(Xn_va, Xt_va, Xi_va).cpu().float().numpy()
        return oof, model


# ── MLP fallback ─────────────────────────────────────────────────────────────

def _fit_fold_mlp(
    Xn, Xt, Xi, Y, tr_idx, va_idx, Xn_v, Xt_v, Xi_v,
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
        m = MLPRegressor(hidden_layer_sizes=(256, 256), max_iter=300, random_state=RANDOM_STATE)
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
    checkpoint_dir: str = "checkpoints",
) -> tuple[np.ndarray, np.ndarray]:
    """GroupKFold (n=5) training with per-fold checkpointing.

    Completed folds are saved to checkpoint_dir and skipped on re-run.
    Delete checkpoints/ to force a full retrain.

    Returns:
        val_preds  (N_val, 3)   — fold-ensemble predictions, clipped ≥ 0
        oof_preds  (N_train, 3) — out-of-fold predictions for CV reporting
    """
    from pathlib import Path
    ckpt = Path(checkpoint_dir)
    ckpt.mkdir(exist_ok=True)

    t0 = time.time()
    device = "cuda" if (_HAS_TORCH and __import__("torch").cuda.is_available()) else "cpu"
    if verbose:
        print(f"  Device: {device}")
        print(f"  Checkpoints: {ckpt.resolve()}")

    Xn, Xt, Xi = train_data["numeric"], train_data["text"], train_data["image"]
    Y = train_data["targets"]
    groups = train_data["keys"]["period_rank"].fillna(0).astype(int).values
    Xn_v, Xt_v, Xi_v = val_data["numeric"], val_data["text"], val_data["image"]

    gkf = GroupKFold(n_splits=N_FOLDS)
    oof = np.full_like(Y, np.nan)
    val_fold_preds: list[np.ndarray] = []
    use_torch = _HAS_TORCH

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(Xn, Y, groups)):
        ckpt_file = ckpt / f"fold_{fold}.npz"

        # ── Resume: load completed fold from disk ────────────────────────────
        if ckpt_file.exists():
            data = np.load(ckpt_file)
            oof[data["va_idx"]] = data["oof_va"]
            val_fold_preds.append(data["val_preds"])
            if verbose:
                print(f"  Fold {fold + 1}/{N_FOLDS} — loaded from checkpoint ✓", flush=True)
            continue

        if verbose:
            print(f"  Fold {fold + 1}/{N_FOLDS}  ({time.time() - t0:.0f}s) ...", end=" ", flush=True)

        use_fallback = (not use_torch) or (time.time() - t0 >= deadline_secs)

        if not use_fallback:
            try:
                oof_va, model = _fit_fold(Xn, Xt, Xi, Y, tr_idx, va_idx, device)
                oof[va_idx] = oof_va
                model.eval()
                with __import__("torch").no_grad():
                    vp = model(
                        __import__("torch").tensor(Xn_v, dtype=__import__("torch").float32).to(device),
                        __import__("torch").tensor(Xt_v, dtype=__import__("torch").float32).to(device),
                        __import__("torch").tensor(Xi_v, dtype=__import__("torch").float32).to(device),
                    ).cpu().float().numpy()
                # Save model weights alongside predictions
                __import__("torch").save(model.state_dict(), ckpt / f"fold_{fold}_weights.pt")
                if verbose:
                    print("transformer", flush=True)
            except Exception as exc:
                warnings.warn(f"Fold {fold + 1} transformer failed ({exc!r}); switching to MLP.")
                use_torch = False

        if use_fallback or not use_torch:
            oof_va, vp = _fit_fold_mlp(Xn, Xt, Xi, Y, tr_idx, va_idx, Xn_v, Xt_v, Xi_v)
            oof[va_idx] = oof_va
            if verbose:
                print("MLP fallback", flush=True)

        val_fold_preds.append(vp)

        # ── Save completed fold ───────────────────────────────────────────────
        np.savez(ckpt_file, va_idx=va_idx, oof_va=oof_va, val_preds=vp)
        if verbose:
            print(f"    → saved to {ckpt_file}", flush=True)

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
        print(f"  Total time: {time.time() - t0:.0f}s")

    return val_preds, oof
