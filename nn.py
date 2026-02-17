import math
import json
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

# Optional TensorFlow import
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

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = str(BASE_DIR / "coin_metal_model.keras")
META_PATH  = str(BASE_DIR / "coin_metal_model_meta.json")




# -------------------------------
# Geometry / physics helpers
# -------------------------------

def compute_density_gcm3(
    mass_g: float,
    diameter_mm: float,
    thickness_mm: float,
    hole_mm: float = 0.0
) -> float:
    """
    Computes volumetric density in g/cm^3 for a coin approximated
    as a cylinder with an optional central hole.

    All NaN / invalid values are converted to 0.0.
    """
    try:
        if mass_g <= 0 or diameter_mm <= 0 or thickness_mm <= 0:
            return 0.0

        R = diameter_mm / 2.0
        r = max(0.0, hole_mm / 2.0)

        if r >= R:
            return 0.0

        volume_mm3 = math.pi * (R * R - r * r) * thickness_mm
        if volume_mm3 <= 0:
            return 0.0

        volume_cm3 = volume_mm3 / 1000.0
        density = mass_g / volume_cm3

        if not math.isfinite(density):
            return 0.0

        # Clamp extreme values
        return float(min(max(density, 0.0), 40.0))
    except Exception:
        return 0.0


# -------------------------------
# Training data preparation
# -------------------------------

def build_training_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Builds a clean, NaN-free training dataframe from the coin DB.
    Uses metal_1..3 + pct_1..3 as labels.
    """

    df = df.copy()

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

    X = df[feature_cols].copy()
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X = np.nan_to_num(X.values, nan=0.0, posinf=40.0, neginf=0.0)

    # ---- Labels ----
    Y = np.zeros((len(df), len(metals)), dtype=np.float32)

    for i, r in df.iterrows():
        for m_col, p_col in [("metal_1", "pct_1"), ("metal_2", "pct_2"), ("metal_3", "pct_3")]:
            m = str(r.get(m_col, "unknown")).lower()
            p = float(r.get(p_col, 0.0) or 0.0)

            if m in metal_to_idx and p > 0:
                Y[i, metal_to_idx[m]] += p

    Y = np.nan_to_num(Y, nan=0.0)
    row_sums = Y.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    Y /= row_sums

    train_df = pd.DataFrame(X, columns=feature_cols)
    train_df["_Y"] = list(Y)

    meta = {
        "metals": metals,
        "feat_cols": feature_cols,
    }

    return train_df, metals


# -------------------------------
# Model
# -------------------------------

def _build_model(input_dim: int, output_dim: int) -> "models.Model":
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(output_dim, activation="softmax"),
    ])
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.002),
        loss="categorical_crossentropy",
    )
    return model


def train_model(train_df: pd.DataFrame, epochs: int = 60) -> Dict:
    if not HAS_TF:
        raise RuntimeError("TensorFlow not available")

    X = np.vstack(train_df.drop(columns=["_Y"]).values)
    Y = np.vstack(train_df["_Y"].values)

    model = _build_model(X.shape[1], Y.shape[1])
    hist = model.fit(X, Y, epochs=epochs, validation_split=0.15, verbose=0)

    model.save(MODEL_PATH)
    with open(META_PATH, "w") as f:
        json.dump(
            {
                "metals": DEFAULT_METALS,
                "feat_cols": list(train_df.drop(columns=["_Y"]).columns),
                "final_loss": float(hist.history["loss"][-1]),
                "final_val_loss": float(hist.history["val_loss"][-1]),
            },
            f,
            indent=2,
        )

    return {
        "final_loss": hist.history["loss"][-1],
        "final_val_loss": hist.history["val_loss"][-1],
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

    if not HAS_TF:
        return {"unknown": 1.0}

    model = tf.keras.models.load_model(MODEL_PATH)
    with open(META_PATH, "r") as f:
        meta = json.load(f)

    density = compute_density_gcm3(mass_g, diameter_mm, thickness_mm, hole_mm)

    x = {
        "mass_g": mass_g,
        "diameter_mm": diameter_mm,
        "thickness_mm": thickness_mm,
        "hole_mm": hole_mm,
        "density": density,
        "year": float(year or 0),
        "color_is_gold": float("gold" in (color_hint or "").lower()),
        "color_is_silver": float("silver" in (color_hint or "").lower()),
        "color_is_copper": float(any(k in (color_hint or "").lower() for k in ["brown", "copper"])),
        "denom_len": float(len(str(denomination or "")[:64])),
    }

    X = np.array([[x[c] for c in meta["feat_cols"]]], dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=40.0, neginf=0.0)

    probs = model.predict(X, verbose=0)[0]
    probs = np.nan_to_num(probs, nan=0.0)

    s = probs.sum()
    if s <= 0:
        return {"unknown": 1.0}

    probs = probs / s
    return {m: float(p) for m, p in zip(meta["metals"], probs)}
