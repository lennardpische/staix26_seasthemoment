"""Pre-compute and cache text + image embeddings for the transformer pipeline.

Run this once before training:
    python -m src.vectorize [--data-root PATH] [--output-dir embeddings]

Saves per-split embedding CSVs keyed on (period_id, jurisdiction):
    embeddings/text_train.csv  — 768-dim sentence-transformer embeddings
    embeddings/text_val.csv
    embeddings/img_train.csv   — 1408-dim EfficientNet-B2 features
    embeddings/img_val.csv

features.py loads these automatically when present; falls back to TF-IDF/PCA otherwise.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TEXT_MODEL = "all-mpnet-base-v2"   # 768-dim, strong semantic quality
IMAGE_MODEL = "efficientnet_b2"    # 1408-dim after global avg pool
IMAGE_SIZE = 224


def vectorize_text(texts: list[str], model_name: str = TEXT_MODEL, batch_size: int = 64) -> np.ndarray:
    """Encode a list of strings with a sentence-transformer. Shape: (N, 768)."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype(np.float32)


def vectorize_images(
    root: Path,
    split: str,
    keys: pd.DataFrame,
    model_name: str = IMAGE_MODEL,
    batch_size: int = 64,
    img_size: int = IMAGE_SIZE,
) -> np.ndarray:
    """Extract EfficientNet features for all (jurisdiction, period_id) pairs.

    Returns shape (N, feature_dim). Rows without an image get a zero vector.
    """
    import timm
    import torch
    from PIL import Image
    from torchvision import transforms

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(model_name, pretrained=True, num_classes=0)  # remove classifier
    model = model.to(device).eval()

    cfg = timm.data.resolve_model_data_config(model)
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg["mean"], std=cfg["std"]),
    ])

    def _load(jurisdiction: str, period_id: str) -> torch.Tensor | None:
        path = root / split / "images" / "mat_density" / f"{jurisdiction}_{period_id}.png"
        try:
            return tf(Image.open(path).convert("RGB"))
        except Exception:
            return None

    results = []
    batch_imgs, batch_indices = [], []
    feat_dim: int | None = None

    for i, row in enumerate(keys.itertuples(index=False)):
        img = _load(row.jurisdiction, row.period_id)
        if img is not None:
            batch_imgs.append(img)
            batch_indices.append(i)

        if len(batch_imgs) == batch_size or (i == len(keys) - 1 and batch_imgs):
            with torch.no_grad(), torch.amp.autocast(device_type=device, dtype=torch.float16):
                feats = model(torch.stack(batch_imgs).to(device)).cpu().float().numpy()
            if feat_dim is None:
                feat_dim = feats.shape[1]
            for idx, feat in zip(batch_indices, feats):
                results.append((idx, feat))
            batch_imgs, batch_indices = [], []

    if feat_dim is None:
        feat_dim = 1408  # efficientnet_b2 default
    out = np.zeros((len(keys), feat_dim), dtype=np.float32)
    for idx, feat in results:
        out[idx] = feat
    return out


def _save_embeddings(keys: pd.DataFrame, embeddings: np.ndarray, path: Path) -> None:
    n_dims = embeddings.shape[1]
    cols = [f"e{i}" for i in range(n_dims)]
    df = pd.concat([
        keys[["period_id", "jurisdiction"]].reset_index(drop=True),
        pd.DataFrame(embeddings, columns=cols),
    ], axis=1)
    df.to_csv(path, index=False)
    print(f"  Saved {path}  {df.shape}")


def run(
    data_root: Path | None = None,
    output_dir: Path | str = "embeddings",
    text_model: str = TEXT_MODEL,
    image_model: str = IMAGE_MODEL,
) -> None:
    from .data_loader import _find_data_root, load_train, load_val

    root = data_root or _find_data_root()
    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    print("Loading data...")
    train_long = load_train(root)
    val_cov = load_val(root)

    KEY = ["period_id", "jurisdiction"]
    train_keys = train_long.drop_duplicates(subset=KEY)[KEY].reset_index(drop=True)
    val_keys = val_cov[KEY].drop_duplicates().reset_index(drop=True)

    # ── Text embeddings ──────────────────────────────────────────────────────
    print(f"\nText embeddings ({text_model})...")
    train_text = train_long.drop_duplicates(subset=KEY).set_index(KEY)["state_doh_release"].reindex(
        pd.MultiIndex.from_frame(train_keys)
    ).fillna("").tolist()
    val_text = val_cov.drop_duplicates(subset=KEY).set_index(KEY)["state_doh_release"].reindex(
        pd.MultiIndex.from_frame(val_keys)
    ).fillna("").tolist()

    print("  Encoding train text...")
    text_tr = vectorize_text(train_text, text_model)
    print("  Encoding val text...")
    text_va = vectorize_text(val_text, text_model)
    _save_embeddings(train_keys, text_tr, out / "text_train.csv")
    _save_embeddings(val_keys, text_va, out / "text_val.csv")

    # ── Image embeddings ─────────────────────────────────────────────────────
    print(f"\nImage embeddings ({image_model})...")
    print("  Encoding train images...")
    img_tr = vectorize_images(root, "train", train_keys, image_model)
    print("  Encoding val images...")
    img_va = vectorize_images(root, "val", val_keys, image_model)
    _save_embeddings(train_keys, img_tr, out / "img_train.csv")
    _save_embeddings(val_keys, img_va, out / "img_val.csv")

    print("\nVectorization complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=str, default="embeddings")
    parser.add_argument("--text-model", type=str, default=TEXT_MODEL)
    parser.add_argument("--image-model", type=str, default=IMAGE_MODEL)
    args = parser.parse_args()
    run(args.data_root, args.output_dir, args.text_model, args.image_model)
