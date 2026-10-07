"""Classically evaluated product-rotation kernels and centred ridge learning.

Method reference: Ayribas, Quantum Artificial Intelligence with Verifiable Kernels (2026).
All calculations here use classical hardware; no quantum advantage is asserted.
"""

from __future__ import annotations

import numpy as np


def matrix(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not x.size or not np.isfinite(x).all():
        raise ValueError("expected a nonempty finite matrix")
    return x


def product_kernel(x, z, scale=1.0):
    x, z = matrix(x), matrix(z)
    if x.shape[1] != z.shape[1] or not np.isfinite(scale) or scale <= 0:
        raise ValueError("dimensions must match and scale must be positive")
    # The direct classical formula avoids constructing an exponential feature vector.
    out = np.ones((len(x), len(z)))
    for j in range(x.shape[1]):
        out *= np.cos(scale * (x[:, j, None] - z[None, :, j]) / 2) ** 2
    return out


def product_states(x, scale=1.0):
    x = matrix(x)
    if x.shape[1] > 10 or not np.isfinite(scale) or scale <= 0:
        raise ValueError("state audit supports at most ten qubits and positive scale")
    out = np.ones((len(x), 1))
    for j in range(x.shape[1]):
        local = np.stack([np.cos(scale * x[:, j] / 2), np.sin(scale * x[:, j] / 2)], axis=1)
        out = (out[:, :, None] * local[:, None, :]).reshape(len(x), -1)
    return out


def trig_features(x, scale=1.0):
    x = matrix(x)
    if x.shape[1] > 8 or not np.isfinite(scale) or scale <= 0:
        raise ValueError("explicit feature audit supports at most eight dimensions")
    out = np.ones((len(x), 1))
    for j in range(x.shape[1]):
        local = np.stack(
            [np.ones(len(x)), np.cos(scale * x[:, j]), np.sin(scale * x[:, j])], axis=1
        ) / np.sqrt(2)
        out = (out[:, :, None] * local[:, None, :]).reshape(len(x), -1)
    return out


def rbf_kernel(x, z, gamma=1.0):
    x, z = matrix(x), matrix(z)
    if x.shape[1] != z.shape[1] or not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("invalid RBF inputs")
    d2 = np.maximum((x * x).sum(1)[:, None] + (z * z).sum(1)[None, :] - 2 * x @ z.T, 0)
    return np.exp(-gamma * d2)


def depolarize(kernel, strength, dimension):
    if not np.isfinite(strength) or not 0 <= strength <= 1 or dimension < 1:
        raise ValueError("invalid global depolarization parameters")
    a = (1 - strength) ** 2
    return a * np.asarray(kernel) + (1 - a) / dimension


def swap_estimate(kernel, shots, rng, *, symmetric=False):
    k = matrix(kernel)
    if not isinstance(shots, (int, np.integer)) or shots <= 0 or np.abs(k).max() > 1 + 1e-12:
        raise ValueError("invalid shot budget or overlap")
    if symmetric:
        if k.shape[0] != k.shape[1] or not np.allclose(k, k.T):
            raise ValueError("symmetric estimate requires symmetric square matrix")
        i, j = np.triu_indices(len(k))
        counts = rng.binomial(shots, np.clip((1 + k[i, j]) / 2, 0, 1))
        out = np.zeros_like(k)
        out[i, j] = 2 * counts / shots - 1
        out[j, i] = out[i, j]
        return out
    return 2 * rng.binomial(shots, np.clip((1 + k) / 2, 0, 1)) / shots - 1


class CenteredRidge:
    """Multioutput ridge with training-only centring and an unpenalized intercept."""

    def fit(self, kernel, y, penalty=0.01, *, positive_eigenspace=False):
        k, y = matrix(kernel), np.asarray(y, dtype=np.float64)
        if k.shape != (len(y), len(y)) or not np.allclose(k, k.T):
            raise ValueError("invalid training Gram matrix")
        if not np.isfinite(y).all() or not np.isfinite(penalty) or penalty <= 0:
            raise ValueError("invalid ridge targets or penalty")
        self.mean = k.mean(axis=0)
        self.overall = k.mean()
        self.ymean = y.mean(axis=0)
        centered = k - self.mean[:, None] - self.mean[None, :] + self.overall
        yc = y - self.ymean
        if positive_eigenspace:
            values, vectors = np.linalg.eigh(centered)
            keep = values > 1e-10 * max(1, np.max(np.abs(values)))
            self.alpha = vectors[:, keep] @ (
                (vectors[:, keep].T @ yc) / (values[keep, None] + len(y) * penalty)
                if yc.ndim == 2
                else (vectors[:, keep].T @ yc) / (values[keep] + len(y) * penalty)
            )
        else:
            self.alpha = np.linalg.solve(centered + len(y) * penalty * np.eye(len(y)), yc)
        return self

    def predict(self, test_kernel):
        q = matrix(test_kernel)
        if q.shape[1] != len(self.mean):
            raise ValueError("test block must reference the training anchors")
        centered = q - self.mean[None, :] - q.mean(axis=1)[:, None] + self.overall
        return self.ymean + centered @ self.alpha


def balanced_accuracy(y, pred):
    classes = np.unique(y)
    return float(np.mean([np.mean(pred[y == c] == c) for c in classes]))
