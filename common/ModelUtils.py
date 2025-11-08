# common/model_utils.py
# Utilities for the edge client: data loading, model building, metadata & anomaly scoring
# Dependencies: tensorflow>=2.12, numpy

import os
import json
import time
import math
import numpy as np
from typing import Tuple, Dict, Any

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except Exception as e:
    raise RuntimeError(
        "TensorFlow is required for model_utils. Install via `pip install tensorflow`."
    ) from e


# -----------------------------
# Config & helpers
# -----------------------------

DEFAULT_INPUT_DIM = int(os.getenv("ATOF_INPUT_DIM", "32"))
DEFAULT_HIDDEN = int(os.getenv("ATOF_HIDDEN_UNITS", "64"))
DEFAULT_NOISE_STD = float(os.getenv("ATOF_DP_NOISE_STD", "0.0"))  # set >0 to simulate local DP
DEFAULT_SEED = int(os.getenv("ATOF_SEED", "42"))

rng = np.random.default_rng(DEFAULT_SEED)


def _ensure_2d(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x.reshape(-1, 1)
    return x


# -----------------------------
# Synthetic data loader (replace with real CPS feeds)
# -----------------------------
class LocalDataset:
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = _ensure_2d(x.astype(np.float32))
        self.y = _ensure_2d(y.astype(np.float32))


def load_local_data(n_samples: int = 2048,
                    input_dim: int = DEFAULT_INPUT_DIM,
                    anomaly_ratio: float = 0.05) -> LocalDataset:
    """
    Generates CPS-like encrypted traffic metadata + sensor residuals.
    Replace with your own loader (e.g., from files, Kafka, MQTT).
    """
    # benign: Gaussian around 0
    x_normal = rng.normal(0.0, 1.0, size=(int(n_samples * (1 - anomaly_ratio)), input_dim))
    y_normal = np.zeros((x_normal.shape[0], 1), dtype=np.float32)

    # anomalous: shifted/variance-changed distribution
    x_anom = rng.normal(2.5, 1.2, size=(n_samples - x_normal.shape[0], input_dim))
    y_anom = np.ones((x_anom.shape[0], 1), dtype=np.float32)

    x = np.concatenate([x_normal, x_anom], axis=0)
    y = np.concatenate([y_normal, y_anom], axis=0)

    # shuffle
    idx = rng.permutation(n_samples)
    x, y = x[idx], y[idx]
    return LocalDataset(x, y)


# -----------------------------
# Model: small autoencoder + head
# -----------------------------
def _build_autoencoder(input_dim: int, hidden: int) -> keras.Model:
    inp = keras.Input(shape=(input_dim,), name="inp")
    z = layers.Dense(hidden, activation="relu")(inp)
    z = layers.Dense(hidden // 2, activation="relu")(z)
    bottleneck = layers.Dense(hidden // 4, activation="relu", name="bottleneck")(z)
    z = layers.Dense(hidden // 2, activation="relu")(bottleneck)
    z = layers.Dense(hidden, activation="relu")(z)
    out = layers.Dense(input_dim, activation=None, name="recon")(z)
    model = keras.Model(inp, out, name="autoencoder")
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model


def _build_detector_head(input_dim: int, hidden: int) -> keras.Model:
    inp = keras.Input(shape=(input_dim,), name="inp")
    z = layers.Dense(hidden, activation="relu")(inp)
    z = layers.Dense(hidden // 2, activation="relu")(z)
    out = layers.Dense(1, activation="sigmoid", name="p_anom")(z)
    model = keras.Model(inp, out, name="detector")
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy")
    return model


class EdgeCompositeModel:
    """
    Wraps an autoencoder (for reconstruction-based anomaly signals) and a
    small classifier head. get_weights/set_weights expose a single flat vector
    so the aggregator logic remains simple.
    """
    def __init__(self, input_dim: int = DEFAULT_INPUT_DIM, hidden: int = DEFAULT_HIDDEN):
        self.ae = _build_autoencoder(input_dim, hidden)
        self.det = _build_detector_head(input_dim, hidden)
        self._shapes = [w.shape for w in self.ae.get_weights() + self.det.get_weights()]
        self._sizes = [int(np.prod(s)) for s in self._shapes]
        self._total = sum(self._sizes)

    def _flatten(self, weights_list):
        return np.concatenate([w.flatten() for w in weights_list]).astype(np.float32)

    def _unflatten(self, flat: np.ndarray):
        outs = []
        cursor = 0
        for shape, size in zip(self._shapes, self._sizes):
            part = flat[cursor: cursor + size]
            outs.append(part.reshape(shape))
            cursor += size
        return outs

    def get_weights(self) -> np.ndarray:
        wl = self.ae.get_weights() + self.det.get_weights()
        return self._flatten(wl)

    def set_weights(self, flat: np.ndarray):
        wl = self._unflatten(np.asarray(flat, dtype=np.float32))
        # split back to ae/det
        n_ae = len(self.ae.get_weights())
        self.ae.set_weights(wl[:n_ae])
        self.det.set_weights(wl[n_ae:])

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int = 1, batch_size: int = 32):
        # train AE for reconstruction
        self.ae.fit(x, x, epochs=epochs, batch_size=batch_size, verbose=0)
        # train detector on labels (if labels exist; otherwise generate pseudo-labels)
        if y is None or y.size == 0:
            # pseudo-labels by top-k recon error
            recon = self.ae.predict(x, verbose=0)
            err = np.mean((recon - x) ** 2, axis=1)
            thresh = np.percentile(err, 90)
            y_hat = (err >= thresh).astype(np.float32).reshape(-1, 1)
            self.det.fit(x, y_hat, epochs=max(1, epochs // 2), batch_size=batch_size, verbose=0)
        else:
            self.det.fit(x, y, epochs=max(1, epochs // 2), batch_size=batch_size, verbose=0)

    def anomaly_score(self, x: np.ndarray) -> float:
        # combine reconstruction error and detector probability
        recon = self.ae.predict(x, verbose=0)
        rec_err = float(np.mean((recon - x) ** 2))
        p = float(np.mean(self.det.predict(x, verbose=0)))
        # normalized combo
        return float((rec_err / (1.0 + rec_err)) * 0.5 + p * 0.5)


def build_model(input_dim: int = DEFAULT_INPUT_DIM,
                hidden: int = DEFAULT_HIDDEN) -> EdgeCompositeModel:
    return EdgeCompositeModel(input_dim=input_dim, hidden=hidden)


# -----------------------------
# Metadata & DP / Secure-agg stubs
# -----------------------------
def _l2_norm(v: np.ndarray) -> float:
    return float(np.sqrt(np.sum(v * v)))


def _expected_norm(dim: int) -> float:
    # heuristic baseline for gradient magnitude expectation
    return float(math.sqrt(dim))


def _clip_by_norm(v: np.ndarray, max_norm: float) -> np.ndarray:
    n = np.linalg.norm(v) + 1e-8
    if n <= max_norm:
        return v
    return v * (max_norm / n)


def _add_local_dp_noise(v: np.ndarray, noise_std: float = DEFAULT_NOISE_STD) -> np.ndarray:
    if noise_std <= 0.0:
        return v
    return v + rng.normal(0.0, noise_std, size=v.shape).astype(np.float32)


def compute_metadata(delta: np.ndarray,
                     new_weights: np.ndarray,
                     local_data: "LocalDataset",
                     clip_norm: float = 1.0,
                     apply_local_dp: bool = DEFAULT_NOISE_STD > 0.0) -> Dict[str, Any]:
    """
    Compute metadata sent with an update. Applies optional clipping and local DP.
    Returns a dict that EdgeClient can serialize as JSON.
    """
    t0 = time.time()
    delta = delta.astype(np.float32)
    clipped = _clip_by_norm(delta, clip_norm)
    if apply_local_dp:
        clipped = _add_local_dp_noise(clipped, DEFAULT_NOISE_STD)

    update_norm = _l2_norm(clipped)
    exp_norm = _expected_norm(clipped.size)

    # very rough reliability: smaller norm & faster training imply more reliable
    latency_ms = int((time.time() - t0) * 1000)
    reliability = float(1.0 / (1.0 + (update_norm / (exp_norm + 1e-6)) + (latency_ms / 1000.0)))

    # quick anomaly score on the local batch
    # (in practice, the caller should pass a model to compute anomaly_score; we mimic via statistics here)
    batch = local_data.x[:min(256, local_data.x.shape[0])]
    batch_var = float(np.var(batch))
    anomaly_score = float(1.0 / (1.0 + math.exp(- (batch_var - 1.0))))  # squashed variance proxy

    md = {
        "update_norm": update_norm,
        "expected_norm": exp_norm,
        "latency_ms": latency_ms,
        "update_reliability": reliability,
        "anomaly_score": anomaly_score,
        "vector_size": int(clipped.size),
    }
    return md


# -----------------------------
# Convenience: training step used by EdgeClient
# -----------------------------
def train_one_round(global_weights: np.ndarray,
                    local_data: LocalDataset,
                    epochs: int = 1,
                    batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Set global → train locally → return (delta, new_weights, metadata)
    """
    model = build_model(input_dim=local_data.x.shape[1])
    model.set_weights(global_weights)
    model.fit(local_data.x, local_data.y, epochs=epochs, batch_size=batch_size)

    new_w = model.get_weights()
    delta = (new_w - global_weights).astype(np.float32)
    md = compute_metadata(delta, new_w, local_data)
    return delta, new_w, md
