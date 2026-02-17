import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Optional TensorFlow import (Streamlit Cloud may not have it)
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers
    HAS_TF = True
except Exception:
    HAS_TF = False


# -------------------------------
# Constants / configuration
# -------------------------------

DEFAULT_METALS = [
    "gold", "silver", "copper", "nickel", "zinc", "tin", "aluminum",
    "steel", "stainless", "iron", "platinum", "palladium", "lead",
    "brass", "bronze", "cupro-nickel", "manganese", "chromium",
    "cobalt", "unknown", "none"
]

# Single-folder paths (save beside this file)
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = str(BASE_DIR / "coin_metal_model.keras")
META_PATH  = str(BASE_DIR / "coin_metal_model_meta.json")


@dataclass
class TrainConfig:
    """Config object expected by app_coin.py."""
    epochs: int = 60
    metals: Optional[List[str]] = None
    learning_rate: float = 0.002
    val_split: float = 0.15


# -------------------------------
# Geometry / physics helpers
# -------------------------------

def compute_density_gcm3(
    mass_g: float,
    diameter_mm: float,
    thickness_mm: float,
    hole_mm: float = 0.0
) -> float:
    """Compute density in g/cm^3 (clamped to [0, 40])."""
    try:
        mass_g = float(mass_g or 0.0)
        diameter_mm = float(diameter_mm or 0.0)
        thickness_mm = float(thickness_mm or 0.0)
        hole_mm = float(hole_mm or 0.0)

        if mass_g <= 0 or diameter_mm <= 0 or thickness_mm <= 0:
            return 0.0

        R = diameter_mm / 2.0
        r = max(0.0, hole_mm) / 2.0

        # mm^3
        volume_mm3 = math.pi * (R * R - r * r) * thickness_mm
        if volume_mm3 <= 0:
            return 0.0

        # cm^3
        volume_cm3 = volume_mm3 / 1000.0
        density = mass_g / volume_cm3

        if not math.isfinite(density):
            return 0.0

        return float(min(max(density, 0.0), 40.0))
    except Exception:
        return 0.0


# -------------------------------
# Training data preparation
# -------------------------------

def build_training_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Builds a clean, NaN-free training dataframe from the coin DB.
    Uses metal_1..3 + pct_1..pct_3 as labels.
    """
    df = df.copy()
    # CRITICAL: ensure contiguous 0..N-1 index so numpy indexing is safe
    df = df.reset_index(drop=True)

    metals = DEFAULT_METALS.copy()
    metal_to_idx = {m: i for i, m in enumerate(metals)}

    # ---- Feature engineering ----
    densities = []
    for _, r in df.iterrows():
        dens = compute_density_gcm3(
            r.get("mass_g", 0.0),
            r.get("diameter_mm", 0.0),
            r.get("thickness_mm", 0.0),
            r.get("hole_mm", 0.0),
        )
        densities.append(dens)

    df["density"] = densities
    df["year"] = pd.to_numeric(df.get("year_start", 0), errors="coerce").fillna(0).astype(int)

    df["color_is_gold"] = df.get("color_hint", "").astype(str).str.contains("gold", case=False).astype(float)
    df["color_is_silver"] = df.get("color_hint", "").astype(str).str.contains("silver", case=False).astype(float)
    df["color_is_copper"] = df.get("color_hint", "").astype(str).str.contains("brown|copper", case=False).astype(float)

    df["denom_len"] = df.get("denomination", "").astype(str).str.slice(0, 64).str.len().astype(float)

    feature_cols = [
        "mass_g",
        "diameter_mm",
        "thickness_mm",
        "hole_mm",
        "density",
        "year",
        "color_is_gold",
        "color_is_silver",
        "color_is_copper",
        "denom_len",
    ]

    Xdf = df[feature_cols].copy()
    Xdf = Xdf.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X = np.nan_to_num(Xdf.values, nan=0.0, posinf=40.0, neginf=0.0)

    # ---- Labels ----
    Y = np.zeros((len(df), len(metals)), dtype=np.float32)

    # enumerate ensures i is 0..N-1 even if df index was weird before reset
    for i, (_, r) in enumerate(df.iterrows()):
        for m_col, p_col in [("metal_1", "pct_1"), ("metal_2", "pct_2"), ("metal_3", "pct_3")]:
            m = str(r.get(m_col, "unknown")).strip().lower()
            try:
                p = float(r.get(p_col, 0.0) or 0.0)
            except Exception:
                p = 0.0

            if m in metal_to_idx and p > 0:
                Y[i, metal_to_idx[m]] += p

    Y = np.nan_to_num(Y, nan=0.0)
    row_sums = Y.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    Y /= row_sums

    train_df = pd.DataFrame(X, columns=feature_cols)
    train_df["_Y"] = list(Y)
    return train_df, metals


# -------------------------------
# Model
# -------------------------------

def _build_model(input_dim: int, output_dim: int, learning_rate: float) -> "models.Model":
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(output_dim, activation="softmax"),
    ])
    model.compile(
        optimizer=optimizers.Adam(learning_rate=float(learning_rate)),
        loss="categorical_crossentropy",
    )
    return model


def train_model(train_df: pd.DataFrame, cfg: Union[TrainConfig, int, None] = None) -> Dict:
    """
    Trains the Keras model and writes:
      - coin_metal_model.keras
      - coin_metal_model_meta.json
    to the same folder as nn.py.

    Supports being called as:
      train_model(train_df, TrainConfig(...))
      train_model(train_df, epochs_int)
      train_model(train_df)
    """
    if not HAS_TF:
        raise RuntimeError("TensorFlow not available (install tensorflow to enable training).")

    # Normalize cfg
    if cfg is None:
        cfg = TrainConfig()
    elif isinstance(cfg, int):
        cfg = TrainConfig(epochs=int(cfg))
    elif not isinstance(cfg, TrainConfig):
        # be forgiving if something dict-like got passed
        try:
            cfg = TrainConfig(**dict(cfg))  # type: ignore
        except Exception:
            cfg = TrainConfig()

    epochs = int(cfg.epochs or 60)
    epochs = max(1, epochs)

    # Extract arrays
    X = np.vstack(train_df.drop(columns=["_Y"]).values)
    Y = np.vstack(train_df["_Y"].values)

    model = _build_model(X.shape[1], Y.shape[1], learning_rate=cfg.learning_rate)

    hist = model.fit(
        X, Y,
        epochs=epochs,
        validation_split=float(cfg.val_split),
        verbose=0
    )

    model.save(MODEL_PATH)

    meta = {
        "metals": list(cfg.metals) if cfg.metals else DEFAULT_METALS,
        "feat_cols": list(train_df.drop(columns=["_Y"]).columns),
        "final_loss": float(hist.history["loss"][-1]),
        "final_val_loss": float(hist.history.get("val_loss", [float("nan")])[-1]),
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return {
        "final_loss": meta["final_loss"],
        "final_val_loss": meta["final_val_loss"],
        "model_path": MODEL_PATH,
        "meta_path": META_PATH,
        "epochs": epochs,
    }


# -------------------------------
# Prediction
# -------------------------------

def predict_metal_probs(
    mass_g: float,
    diameter_mm: float,
    thickness_mm: float,
    hole_mm: float = 0.0,
    year: Optional[int] = None,
    denomination: Optional[str] = None,
    color_hint: Optional[str] = None,
) -> Dict[str, float]:
    """
    Loads saved model + meta if available and predicts metal probabilities.
    If TensorFlow isn't available or the model files don't exist, returns {"unknown": 1.0}.
    """
    if not HAS_TF:
        return {"unknown": 1.0}

    # If model hasn't been trained/saved yet
    if not Path(MODEL_PATH).exists() or not Path(META_PATH).exists():
        return {"unknown": 1.0}

    model = tf.keras.models.load_model(MODEL_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    density = compute_density_gcm3(mass_g, diameter_mm, thickness_mm, hole_mm)

    x = {
        "mass_g": float(mass_g),
        "diameter_mm": float(diameter_mm),
        "thickness_mm": float(thickness_mm),
        "hole_mm": float(hole_mm),
        "density": float(density),
        "year": float(year or 0),
        "color_is_gold": float("gold" in (color_hint or "").lower()),
        "color_is_silver": float("silver" in (color_hint or "").lower()),
        "color_is_copper": float(any(k in (color_hint or "").lower() for k in ["brown", "copper"])),
        "denom_len": float(len(str(denomination or "")[:64])),
    }

    feat_cols = meta.get("feat_cols") or []
    if not feat_cols:
        return {"unknown": 1.0}

    X = np.array([[x.get(c, 0.0) for c in feat_cols]], dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=40.0, neginf=0.0)

    probs = model.predict(X, verbose=0)[0]
    probs = np.nan_to_num(probs, nan=0.0)

    s = float(probs.sum())
    if s <= 0:
        return {"unknown": 1.0}

    probs = probs / s
    metals = meta.get("metals") or DEFAULT_METALS
    return {m: float(p) for m, p in zip(metals, probs)}
