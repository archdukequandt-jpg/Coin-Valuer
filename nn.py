import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Optional TensorFlow import (some deploy targets won't have it)
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
KERAS_MODEL_PATH = BASE_DIR / "coin_metal_model.keras"
NP_MODEL_PATH    = BASE_DIR / "coin_metal_model_np.npz"
META_PATH        = BASE_DIR / "coin_metal_model_meta.json"


@dataclass
class TrainConfig:
    """Config object expected by app_coin.py."""
    epochs: int = 60
    metals: Optional[List[str]] = None
    learning_rate: float = 0.05      # used by numpy fallback; TF uses its own default unless overridden
    val_split: float = 0.15
    l2: float = 1e-4                 # numpy fallback regularization


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
    df = df.reset_index(drop=True)  # ensure 0..N-1 indexing for numpy arrays

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
# Numpy fallback model
# -------------------------------

def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    s = np.sum(e, axis=1, keepdims=True)
    s[s == 0] = 1.0
    return e / s

def _cross_entropy_soft_labels(P: np.ndarray, Y: np.ndarray) -> float:
    eps = 1e-9
    return float(-np.mean(np.sum(Y * np.log(P + eps), axis=1)))

def _train_numpy_softmax_regression(
    X: np.ndarray,
    Y: np.ndarray,
    epochs: int,
    lr: float,
    l2: float,
    seed: int = 7
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Multinomial logistic regression (softmax regression) with soft labels.
    """
    rng = np.random.default_rng(seed)
    n, d = X.shape
    k = Y.shape[1]

    # Simple standardization helps stability (persist stats in meta)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    Xn = (X - mu) / sd

    W = rng.normal(0, 0.01, size=(d, k)).astype(np.float32)
    b = np.zeros((1, k), dtype=np.float32)

    best_loss = float("inf")
    for _ in range(max(1, int(epochs))):
        logits = Xn @ W + b
        P = _softmax(logits)
        loss = _cross_entropy_soft_labels(P, Y) + float(l2) * float(np.sum(W * W))

        # gradients
        # dL/dlogits = (P - Y)/n
        G = (P - Y) / float(n)
        dW = Xn.T @ G + 2.0 * float(l2) * W
        db = np.sum(G, axis=0, keepdims=True)

        W -= float(lr) * dW
        b -= float(lr) * db

        best_loss = min(best_loss, loss)

    metrics = {"final_loss": float(loss), "best_loss": float(best_loss)}
    return (W, b, mu.astype(np.float32), sd.astype(np.float32), metrics)


def _predict_numpy_softmax_regression(
    X: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    mu: np.ndarray,
    sd: np.ndarray,
) -> np.ndarray:
    Xn = (X - mu) / sd
    P = _softmax(Xn @ W + b)
    return P


# -------------------------------
# TF model
# -------------------------------

def _build_tf_model(input_dim: int, output_dim: int, learning_rate: float) -> "models.Model":
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


# -------------------------------
# Train
# -------------------------------

def train_model(train_df: pd.DataFrame, cfg: Union[TrainConfig, int, None] = None) -> Dict:
    """
    Trains and saves a metal-composition classifier.
    - If TensorFlow is available, trains a small Keras model and saves coin_metal_model.keras
    - Otherwise, trains a lightweight numpy softmax regression and saves coin_metal_model_np.npz

    Always writes coin_metal_model_meta.json with the feature columns and metal labels.
    """
    # Normalize cfg
    if cfg is None:
        cfg = TrainConfig()
    elif isinstance(cfg, int):
        cfg = TrainConfig(epochs=int(cfg))
    elif not isinstance(cfg, TrainConfig):
        try:
            cfg = TrainConfig(**dict(cfg))  # type: ignore
        except Exception:
            cfg = TrainConfig()

    epochs = max(1, int(cfg.epochs or 60))

    X = np.vstack(train_df.drop(columns=["_Y"]).values).astype(np.float32)
    Y = np.vstack(train_df["_Y"].values).astype(np.float32)

    # Choose metals list in meta
    metals = list(cfg.metals) if cfg.metals else DEFAULT_METALS

    if HAS_TF:
        model = _build_tf_model(X.shape[1], Y.shape[1], learning_rate=cfg.learning_rate)
        hist = model.fit(X, Y, epochs=epochs, validation_split=float(cfg.val_split), verbose=0)
        model.save(str(KERAS_MODEL_PATH))

        meta = {
            "backend": "tensorflow",
            "metals": metals,
            "feat_cols": list(train_df.drop(columns=["_Y"]).columns),
            "final_loss": float(hist.history["loss"][-1]),
            "final_val_loss": float(hist.history.get("val_loss", [float("nan")])[-1]),
        }
        META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return {
            "backend": meta["backend"],
            "final_loss": meta["final_loss"],
            "final_val_loss": meta["final_val_loss"],
            "model_path": str(KERAS_MODEL_PATH),
            "meta_path": str(META_PATH),
            "epochs": epochs,
        }

    # ---- numpy fallback ----
    W, b, mu, sd, m = _train_numpy_softmax_regression(
        X=X,
        Y=Y,
        epochs=epochs,
        lr=float(cfg.learning_rate),
        l2=float(cfg.l2),
    )
    np.savez_compressed(str(NP_MODEL_PATH), W=W, b=b, mu=mu, sd=sd)

    meta = {
        "backend": "numpy",
        "metals": metals,
        "feat_cols": list(train_df.drop(columns=["_Y"]).columns),
        "final_loss": float(m.get("final_loss", float("nan"))),
        "final_val_loss": None,  # not computed for numpy fallback
        "note": "TensorFlow not available; trained numpy softmax regression fallback.",
    }
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return {
        "backend": meta["backend"],
        "final_loss": meta["final_loss"],
        "final_val_loss": meta["final_val_loss"],
        "model_path": str(NP_MODEL_PATH),
        "meta_path": str(META_PATH),
        "epochs": epochs,
    }


# -------------------------------
# Predict
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
    Predict metal probabilities.
    - Prefer TF model if present & loadable.
    - Else use numpy fallback model if present.
    - Else return {"unknown": 1.0}.
    """
    if not META_PATH.exists():
        return {"unknown": 1.0}

    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    feat_cols = meta.get("feat_cols") or []
    metals = meta.get("metals") or DEFAULT_METALS
    if not feat_cols:
        return {"unknown": 1.0}

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
    X = np.array([[x.get(c, 0.0) for c in feat_cols]], dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=40.0, neginf=0.0)

    # 1) TensorFlow path
    if HAS_TF and KERAS_MODEL_PATH.exists():
        try:
            model = tf.keras.models.load_model(str(KERAS_MODEL_PATH))
            probs = model.predict(X, verbose=0)[0]
            probs = np.nan_to_num(probs, nan=0.0)
            s = float(probs.sum())
            if s <= 0:
                return {"unknown": 1.0}
            probs = probs / s
            return {m: float(p) for m, p in zip(metals, probs)}
        except Exception:
            # fall through to numpy if TF model can't load
            pass

    # 2) numpy fallback
    if NP_MODEL_PATH.exists():
        try:
            data = np.load(str(NP_MODEL_PATH))
            W = data["W"]
            b = data["b"]
            mu = data["mu"]
            sd = data["sd"]
            probs = _predict_numpy_softmax_regression(X, W=W, b=b, mu=mu, sd=sd)[0]
            probs = np.nan_to_num(probs, nan=0.0)
            s = float(probs.sum())
            if s <= 0:
                return {"unknown": 1.0}
            probs = probs / s
            return {m: float(p) for m, p in zip(metals, probs)}
        except Exception:
            return {"unknown": 1.0}

    return {"unknown": 1.0}
