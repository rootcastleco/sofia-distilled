"""Train a NumPy teacher, KL-distilled student and supervised-only control."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from itertools import pairwise
from pathlib import Path

import numpy as np

from .edge import FEATURE_NAMES, LABELS, SCHEMA, extract_features, softmax, waveform


def initialize(dims, rng):
    params = {}
    for i, (a, b) in enumerate(pairwise(dims)):
        params[f"w{i}"] = rng.normal(0, np.sqrt(2 / a), (a, b))
        params[f"b{i}"] = np.zeros(b)
    return params


def forward(params, x):
    acts = [x]
    for i in range(len(params) // 2):
        h = acts[-1] @ params[f"w{i}"] + params[f"b{i}"]
        acts.append(np.maximum(h, 0) if i < len(params) // 2 - 1 else h)
    return acts


def train(params, x, y, xv, yv, *, epochs, seed, targets=None, alpha=0.7, temperature=2.0):
    rng = np.random.default_rng(seed)
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(val) for k, val in params.items()}
    onehot = np.eye(len(LABELS))[y]
    best_loss, best, step, history = float("inf"), None, 0, []
    for epoch in range(epochs):
        order = rng.permutation(len(x))
        for batch in range(0, len(x), 128):
            idx = order[batch : batch + 128]
            acts = forward(params, x[idx])
            grad = softmax(acts[-1]) - onehot[idx]
            if targets is not None:
                grad = (1 - alpha) * grad + alpha * temperature * (
                    softmax(acts[-1], temperature) - targets[idx]
                )
            grad /= len(idx)
            grads = {}
            for i in reversed(range(len(params) // 2)):
                grads[f"w{i}"] = acts[i].T @ grad
                grads[f"b{i}"] = grad.sum(axis=0)
                grad = grad @ params[f"w{i}"].T
                if i:
                    grad *= acts[i] > 0
            step += 1
            for key in params:
                m[key] = 0.9 * m[key] + 0.1 * grads[key]
                v[key] = 0.999 * v[key] + 0.001 * grads[key] ** 2
                params[key] -= (
                    0.002
                    * (m[key] / (1 - 0.9**step))
                    / (np.sqrt(v[key] / (1 - 0.999**step)) + 1e-8)
                )
        p = softmax(forward(params, xv)[-1])
        loss = float(-np.log(np.maximum(p[np.arange(len(yv)), yv], 1e-12)).mean())
        history.append({"epoch": epoch + 1, "validation_cross_entropy": loss})
        if loss < best_loss:
            best_loss, best = loss, {k: val.copy() for k, val in params.items()}
        if (epoch + 1) % 20 == 0:
            print(f"epoch {epoch + 1}: validation CE={loss:.5f}", flush=True)
    return best, history


def dataset(n, seed, *, rpm_range=(18, 42)):
    rng = np.random.default_rng(seed)
    x, labels = [], []
    for index in range(n):
        label = index % len(LABELS)
        shaft = rng.uniform(*rpm_range)
        x.append(extract_features(waveform(label, rng, shaft_hz=shaft), 2048, shaft))
        labels.append(label)
    return np.stack(x), np.asarray(labels)


def metrics(params, x, y):
    p = softmax(forward(params, x)[-1])
    pred = p.argmax(axis=1)
    confusion = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    np.add.at(confusion, (y, pred), 1)
    f1 = []
    for i in range(len(LABELS)):
        tp = confusion[i, i]
        f1.append(float(2 * tp / max(1, confusion[i].sum() + confusion[:, i].sum())))
    return {
        "accuracy": float((pred == y).mean()),
        "macro_f1": float(np.mean(f1)),
        "cross_entropy": float(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-12)).mean()),
        "confusion_matrix": confusion.tolist(),
        "per_class_f1": dict(zip(LABELS, f1)),
    }


def export(folder, params, dims, mean, scale, role):
    folder.mkdir(parents=True, exist_ok=True)
    weights = folder / "model.npz"
    np.savez(
        weights,
        **{k: v.astype(np.float32) for k, v in params.items()},
        mean=mean.astype(np.float32),
        scale=scale.astype(np.float32),
    )
    config = {
        "model_type": "sofia-edge-mlp",
        "version": "0.1.0",
        "role": role,
        "feature_schema": SCHEMA,
        "feature_names": list(FEATURE_NAMES),
        "labels": list(LABELS),
        "dimensions": dims,
        "input_unit": "g",
        "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "parameter_count": sum(v.size for v in params.values()),
        "training_data": "analytical synthetic waveforms only",
        "sample_rate_hz": 2048,
        "window_size": 1024,
        "training_shaft_hz": [18, 42],
    }
    (folder / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/edge"))
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--samples", type=int, default=6000)
    args = parser.parse_args()
    if args.samples < 100 or args.epochs < 1:
        parser.error("samples >= 100 and epochs >= 1 required")
    started = time.perf_counter()
    xt, yt = dataset(args.samples, 42)
    xv, yv = dataset(1000, 43)
    xs, ys = dataset(1500, 44)
    xo, yo = dataset(1500, 45, rpm_range=(43, 65))
    mean, scale = xt.mean(axis=0), np.maximum(xt.std(axis=0), 1e-6)
    normalized = [(x - mean) / scale for x in (xt, xv, xs, xo)]
    xt, xv, xs, xo = normalized
    print("Training teacher (14 -> 128 -> 64 -> 5)", flush=True)
    teacher, ht = train(
        initialize([14, 128, 64, 5], np.random.default_rng(46)),
        xt,
        yt,
        xv,
        yv,
        epochs=args.epochs,
        seed=47,
    )
    targets = softmax(forward(teacher, xt)[-1], 2.0)
    print("Training distilled student (14 -> 16 -> 5)", flush=True)
    initial = initialize([14, 16, 5], np.random.default_rng(48))
    student, hs = train(
        {k: v.copy() for k, v in initial.items()},
        xt,
        yt,
        xv,
        yv,
        epochs=args.epochs,
        seed=49,
        targets=targets,
    )
    print("Training supervised-only student control", flush=True)
    control, hc = train(initial, xt, yt, xv, yv, epochs=args.epochs, seed=49)
    config = export(args.output, student, [14, 16, 5], mean, scale, "distilled-student")
    teacher_cfg = export(args.output / "teacher", teacher, [14, 128, 64, 5], mean, scale, "teacher")
    export(args.output / "control", control, [14, 16, 5], mean, scale, "supervised-control")
    np.savez(
        args.output / "evaluation.npz",
        features=xs * scale + mean,
        labels=ys,
        ood_features=xo * scale + mean,
        ood_labels=yo,
    )
    report = {
        "dataset": "sofia.synthetic-vibration.v1",
        "seed": 42,
        "split": {"train": args.samples, "validation": 1000, "test": 1500, "ood_test": 1500},
        "split_method": "independently generated waveforms; separate RNG seeds; train-only scaling",
        "ood_shift": "shaft frequency 43-65 Hz versus 18-42 Hz during training",
        "epochs": args.epochs,
        "temperature": 2.0,
        "distillation_alpha": 0.7,
        "checkpoint_selection": "minimum validation hard-label cross-entropy",
        "teacher": metrics(teacher, xs, ys),
        "student": metrics(student, xs, ys),
        "supervised_control": metrics(control, xs, ys),
        "ood_teacher": metrics(teacher, xo, yo),
        "ood_student": metrics(student, xo, yo),
        "teacher_agreement": float(
            (forward(teacher, xs)[-1].argmax(1) == forward(student, xs)[-1].argmax(1)).mean()
        ),
        "student_parameters": config["parameter_count"],
        "teacher_parameters": teacher_cfg["parameter_count"],
        "parameter_compression_ratio": teacher_cfg["parameter_count"] / config["parameter_count"],
        "elapsed_seconds": time.perf_counter() - started,
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "limitations": [
            "No real sensor measurements or independent field validation",
            "Softmax probabilities are not calibrated confidence",
            "Toy fault formulas do not establish physical diagnostic accuracy",
        ],
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output / "training_history.json").write_text(
        json.dumps({"teacher": ht, "student": hs, "control": hc}, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
