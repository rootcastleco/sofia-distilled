"""NumPy signal classifier and reproducible synthetic waveform specification."""

from __future__ import annotations

import hashlib
import json
from itertools import pairwise
from pathlib import Path

import numpy as np

LABELS = ("healthy", "imbalance", "misalignment", "bearing_impulses", "rubbing")
FEATURE_NAMES = (
    "log_rms",
    "log_peak",
    "kurtosis",
    "crest_factor",
    "shaft_energy",
    "second_harmonic_energy",
    "third_harmonic_energy",
    "fourth_harmonic_energy",
    "high_frequency_energy",
    "spectral_flatness",
    "spectral_entropy",
    "envelope_kurtosis",
    "log_envelope_rms",
    "zero_crossing_rate",
)
SCHEMA = "sofia.synthetic-vibration.v1"


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(axis=-1, keepdims=True)
    ex = np.exp(z)
    return ex / ex.sum(axis=-1, keepdims=True)


def extract_features(signal: np.ndarray, sample_rate: float, shaft_hz: float) -> np.ndarray:
    """14 dimensionless/log features; amplitude input must be acceleration in g."""
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 1 or x.size < 128 or not np.isfinite(x).all():
        raise ValueError("signal must be a finite 1-D waveform with at least 128 samples")
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("sample_rate must be positive and finite")
    if not np.isfinite(shaft_hz) or not 0 < shaft_hz < sample_rate / 10:
        raise ValueError("shaft_hz must be positive and below sample_rate / 10")
    x = x - x.mean()
    eps = 1e-12
    rms = np.sqrt(np.mean(x * x))
    peak = np.max(np.abs(x))
    if rms < 1e-9:
        raise ValueError("constant or zero-energy signals cannot be classified")
    power = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
    freq = np.fft.rfftfreq(x.size, 1 / sample_rate)
    total = power.sum() + eps
    width = max(2 * sample_rate / x.size, shaft_hz * 0.12)
    bands = [power[np.abs(freq - k * shaft_hz) < width].sum() / total for k in range(1, 5)]
    dist = power / total
    analytic_filter = np.zeros(x.size)
    analytic_filter[0] = 1
    analytic_filter[1 : (x.size + 1) // 2] = 2
    if x.size % 2 == 0:
        analytic_filter[x.size // 2] = 1
    envelope = np.abs(np.fft.ifft(np.fft.fft(x) * analytic_filter))
    env_center = envelope - envelope.mean()
    env_var = np.mean(env_center**2)
    return np.asarray(
        [
            np.log(rms + eps),
            np.log(peak + eps),
            np.mean(x**4) / (rms**4 + eps),
            peak / rms,
            *bands,
            power[freq > sample_rate * 0.2].sum() / total,
            np.exp(np.mean(np.log(power + eps))) / (power.mean() + eps),
            -np.sum(dist * np.log(dist + eps)) / np.log(power.size),
            np.mean(env_center**4) / (env_var**2 + eps),
            np.log(np.sqrt(np.mean(envelope**2)) + eps),
            np.mean(x[1:] * x[:-1] < 0),
        ],
        dtype=np.float64,
    )


def waveform(
    label: int,
    rng: np.random.Generator,
    *,
    shaft_hz: float,
    sample_rate: float = 2048,
    size: int = 1024,
) -> np.ndarray:
    """Analytical toy scenarios, not a simulation validated against real machinery."""
    if label not in range(len(LABELS)):
        raise ValueError("unknown scenario")
    t = np.arange(size) / sample_rate
    phase = rng.uniform(0, 2 * np.pi, 4)
    fundamental = rng.uniform(0.15, 0.35)
    second = rng.uniform(0.01, 0.04)
    third = rng.uniform(0.005, 0.02)
    if label == 1:
        fundamental *= rng.uniform(3.0, 6.0)
    if label == 2:
        second = fundamental * rng.uniform(0.9, 1.8)
        third = fundamental * rng.uniform(0.3, 0.8)
    x = (
        fundamental * np.sin(2 * np.pi * shaft_hz * t + phase[0])
        + second * np.sin(4 * np.pi * shaft_hz * t + phase[1])
        + third * np.sin(6 * np.pi * shaft_hz * t + phase[2])
    )
    x += rng.normal(0, rng.uniform(0.005, 0.025), size)
    if label == 3:
        period = sample_rate / (shaft_hz * rng.uniform(4.4, 5.2))
        for start in np.arange(rng.uniform(0, period), size, period):
            lag = np.arange(size - int(start)) / sample_rate
            pulse = rng.uniform(0.7, 1.8) * np.exp(-lag * rng.uniform(180, 350))
            x[int(start) :] += pulse * np.sin(2 * np.pi * rng.uniform(500, 750) * lag)
    if label == 4:
        x += rng.uniform(0.2, 0.5) * np.sin(2 * np.pi * shaft_hz * t + phase[3]) ** 3
        x += rng.normal(0, rng.uniform(0.10, 0.22), size)
    return x * rng.uniform(0.8, 1.2)


class EdgeModel:
    """Load verified, pickle-free weights and classify a signal or feature matrix."""

    def __init__(self, folder: str | Path):
        folder = Path(folder)
        self.config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        if self.config.get("feature_schema") != SCHEMA:
            raise ValueError("unsupported feature schema")
        if self.config.get("feature_names") != list(FEATURE_NAMES):
            raise ValueError("feature ordering mismatch")
        path = folder / "model.npz"
        if hashlib.sha256(path.read_bytes()).hexdigest() != self.config["weights_sha256"]:
            raise ValueError("weights failed SHA-256 verification")
        with np.load(path, allow_pickle=False) as archive:
            self.weights = {k: archive[k].copy() for k in archive.files}
        dims = self.config["dimensions"]
        if dims[0] != len(FEATURE_NAMES) or dims[-1] != len(LABELS):
            raise ValueError("invalid input/output dimensions")
        for i, (a, b) in enumerate(pairwise(dims)):
            if self.weights[f"w{i}"].shape != (a, b) or self.weights[f"b{i}"].shape != (b,):
                raise ValueError("invalid weight dimensions")
        for name in ("mean", "scale"):
            if self.weights[name].shape != (len(FEATURE_NAMES),):
                raise ValueError("invalid scaler dimensions")
        if not all(np.isfinite(v).all() for v in self.weights.values()):
            raise ValueError("non-finite model parameters")
        if (self.weights["scale"] <= 0).any():
            raise ValueError("invalid feature scale")

    def predict_features(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float64)
        if x.ndim not in (1, 2) or x.shape[-1] != len(FEATURE_NAMES):
            raise ValueError("expected 14 features per sample")
        if not np.isfinite(x).all():
            raise ValueError("features must be finite")
        h = (x - self.weights["mean"]) / self.weights["scale"]
        for i in range(len(self.config["dimensions"]) - 1):
            h = h @ self.weights[f"w{i}"] + self.weights[f"b{i}"]
            if i < len(self.config["dimensions"]) - 2:
                h = np.maximum(h, 0)
        return softmax(h)

    def predict_signal(self, signal: np.ndarray, sample_rate: float, shaft_hz: float) -> dict:
        features = extract_features(signal, sample_rate, shaft_hz)
        p = self.predict_features(features)
        z = np.abs((features - self.weights["mean"]) / self.weights["scale"])
        return {
            "label": LABELS[int(p.argmax())],
            "probabilities": dict(zip(LABELS, p.tolist())),
            "max_probability": float(p.max()),
            "out_of_training_range": bool((z > 6).any()),
            "feature_schema": SCHEMA,
            "status": "experimental_synthetic_only",
        }
