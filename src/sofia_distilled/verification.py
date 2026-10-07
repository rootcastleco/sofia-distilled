"""Independent numerical identities, paired kernel controls and finite-shot study."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .kernels import (
    CenteredRidge,
    balanced_accuracy,
    depolarize,
    product_kernel,
    product_states,
    rbf_kernel,
    swap_estimate,
    trig_features,
)


def ci(values, rng):
    values = np.asarray(values)
    bootstrap = values[rng.integers(0, len(values), (2000, len(values)))].mean(1)
    return {
        "mean": float(values.mean()),
        "pointwise_95_percent_bootstrap_ci": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
    }


def identity_audit():
    rng = np.random.default_rng(20260911)
    errors = {
        "state_overlap": 0.0,
        "explicit_classical_features": 0.0,
        "shared_terminal_unitary": 0.0,
        "density_depolarization": 0.0,
        "matched_regularization_predictions": 0.0,
    }
    for dim in (1, 2, 4, 6, 8):
        x = rng.uniform(-np.pi, np.pi, (12, dim))
        k = product_kernel(x, x)
        states = product_states(x)
        phi = trig_features(x)
        unitary, _ = np.linalg.qr(rng.normal(size=(2**dim, 2**dim)))
        transformed = states @ unitary
        errors["state_overlap"] = max(
            errors["state_overlap"], float(np.max(np.abs(k - (states @ states.T) ** 2)))
        )
        errors["explicit_classical_features"] = max(
            errors["explicit_classical_features"], float(np.max(np.abs(k - phi @ phi.T)))
        )
        errors["shared_terminal_unitary"] = max(
            errors["shared_terminal_unitary"],
            float(np.max(np.abs(k - (transformed @ transformed.T) ** 2))),
        )
        if dim <= 4:
            rho = states[:, :, None] * states[:, None, :]
            for p in (0.0, 0.5, 0.75, 0.9, 1.0):
                noisy = (1 - p) * rho + p * np.eye(2**dim) / 2**dim
                direct = np.einsum("aij,bji->ab", noisy, noisy)
                errors["density_depolarization"] = max(
                    errors["density_depolarization"],
                    float(np.max(np.abs(direct - depolarize(k, p, 2**dim)))),
                )
        y = rng.normal(size=len(x))
        z = rng.uniform(-np.pi, np.pi, (8, dim))
        q = product_kernel(z, x)
        reference = CenteredRidge().fit(k, y, 0.01).predict(q)
        for p in (0.25, 0.5, 0.75, 0.9):
            prediction = (
                CenteredRidge()
                .fit(depolarize(k, p, 2**dim), y, (1 - p) ** 2 * 0.01)
                .predict(depolarize(q, p, 2**dim))
            )
            errors["matched_regularization_predictions"] = max(
                errors["matched_regularization_predictions"],
                float(np.max(np.abs(reference - prediction))),
            )
    return {
        "tested_dimensions": [1, 2, 4, 6, 8],
        "max_absolute_errors": errors,
        "tolerance": 1e-10,
        "passed": all(v < 1e-10 for v in errors.values()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/verification"))
    parser.add_argument("--replicates", type=int, default=30)
    args = parser.parse_args()
    if args.replicates < 2:
        parser.error("at least two independent replicates required")
    if (args.output / "report.json").exists():
        parser.error("output already contains results; choose a fresh output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "seed": 20261008,
        "replicates": args.replicates,
        "tasks": ["linear", "parity", "periodic"],
        "label_flip_probability": 0.05,
        "split": [96, 64, 256],
        "dimension": 4,
        "scales": [0.25, 0.5, 1, 2, 4],
        "rbf_gamma": [0.125, 0.5, 2, 8, 32],
        "penalties": [1, 0.1, 0.01, 0.001, 0.0001],
        "noise_strengths": [0, 0.5, 0.75],
        "shots": [128, 512, 2048, 8192],
        "source": "https://www.academia.edu/175377730/Quantum_Artificial_Intelligence_with_Verifiable_Kernels",
        "scope": "independent adapted implementation; not a reproduction of original seed-level archive",
    }
    raw = json.dumps(protocol, indent=2).encode()
    (args.output / "protocol.json").write_bytes(raw)
    audit = identity_audit()
    if not audit["passed"]:
        raise RuntimeError("identity audit failed")
    results, noise_results, retained = [], [], {}
    for task_id, task in enumerate(protocol["tasks"]):
        for rep in range(args.replicates):
            rng = np.random.default_rng(np.random.SeedSequence([protocol["seed"], task_id, rep]))
            x = rng.uniform(-np.pi, np.pi, (416, 4))
            functions = [
                x[:, 0] + 0.6 * x[:, 1] - 0.4 * x[:, 2],
                x[:, 0] * x[:, 1],
                np.sin(x[:, 0]) * np.sin(x[:, 1]) + 0.5 * np.cos(x[:, 2]),
            ]
            y = np.where(functions[task_id] >= 0, 1.0, -1.0)
            y[rng.random(len(y)) < 0.05] *= -1
            retained[f"{task}_{rep}_x"], retained[f"{task}_{rep}_y"] = x, y
            for kind in ("linear", "rbf", "product"):
                best, chosen = -1.0, None
                candidates = (
                    [1]
                    if kind == "linear"
                    else (protocol["rbf_gamma"] if kind == "rbf" else protocol["scales"])
                )

                def kernel(a, b, s, kind=kind):
                    return (
                        a @ b.T / np.pi**2
                        if kind == "linear"
                        else rbf_kernel(a / np.pi, b / np.pi, s)
                        if kind == "rbf"
                        else product_kernel(a, b, s)
                    )

                for s in candidates:
                    k, q = kernel(x[:96], x[:96], s), kernel(x[96:160], x[:96], s)
                    for penalty in protocol["penalties"]:
                        model = CenteredRidge().fit(k, y[:96], penalty)
                        score = balanced_accuracy(y[96:160], np.where(model.predict(q) >= 0, 1, -1))
                        if score > best:
                            best, chosen = score, (s, penalty)
                s, penalty = chosen
                scores = (
                    CenteredRidge()
                    .fit(kernel(x[:160], x[:160], s), y[:160], penalty)
                    .predict(kernel(x[160:], x[:160], s))
                )
                retained[f"{task}_{rep}_{kind}_scores"] = scores
                results.append(
                    {
                        "task": task,
                        "replicate": rep,
                        "model": kind,
                        "scale": s,
                        "penalty": penalty,
                        "validation_balanced_accuracy": best,
                        "test_balanced_accuracy": balanced_accuracy(
                            y[160:], np.where(scores >= 0, 1, -1)
                        ),
                    }
                )
            if task == "periodic":
                k, q = product_kernel(x[:160], x[:160]), product_kernel(x[160:], x[:160])
                ideal = CenteredRidge().fit(k, y[:160], 0.01).predict(q)
                for p in protocol["noise_strengths"]:
                    a = (1 - p) ** 2
                    kp, qp = depolarize(k, p, 16), depolarize(q, p, 16)
                    exact = CenteredRidge().fit(kp, y[:160], a * 0.01).predict(qp)
                    for shots in protocol["shots"]:
                        shot_rng = np.random.default_rng(
                            np.random.SeedSequence([protocol["seed"], rep, round(p * 100), shots])
                        )
                        kh = swap_estimate(kp, shots, shot_rng, symmetric=True)
                        qh = swap_estimate(qp, shots, shot_rng)
                        predicted = (
                            CenteredRidge()
                            .fit(kh, y[:160], a * 0.01, positive_eigenspace=True)
                            .predict(qh)
                        )
                        retained[f"periodic_{rep}_p{p}_shots{shots}"] = predicted
                        noise_results.append(
                            {
                                "replicate": rep,
                                "p": p,
                                "shots": shots,
                                "exact_matched_max_error": float(np.max(np.abs(exact - ideal))),
                                "ideal_balanced_accuracy": balanced_accuracy(
                                    y[160:], np.where(ideal >= 0, 1, -1)
                                ),
                                "shot_balanced_accuracy": balanced_accuracy(
                                    y[160:], np.where(predicted >= 0, 1, -1)
                                ),
                            }
                        )
            print(f"verification {task}: replicate {rep + 1}/{args.replicates}", flush=True)
    rng = np.random.default_rng(20261014)
    summary = {}
    for task in protocol["tasks"]:
        summary[task] = {
            model: ci(
                [
                    r["test_balanced_accuracy"]
                    for r in results
                    if r["task"] == task and r["model"] == model
                ],
                rng,
            )
            for model in ("linear", "rbf", "product")
        }
        differences = [
            next(
                r["test_balanced_accuracy"]
                for r in results
                if r["task"] == task and r["replicate"] == rep and r["model"] == "product"
            )
            - next(
                r["test_balanced_accuracy"]
                for r in results
                if r["task"] == task and r["replicate"] == rep and r["model"] == "rbf"
            )
            for rep in range(args.replicates)
        ]
        summary[task]["paired_product_minus_rbf"] = ci(differences, rng)
    finite = [
        {
            "p": p,
            "shots": shots,
            **ci(
                [
                    r["shot_balanced_accuracy"]
                    for r in noise_results
                    if r["p"] == p and r["shots"] == shots
                ],
                rng,
            ),
        }
        for p in protocol["noise_strengths"]
        for shots in protocol["shots"]
    ]
    np.savez_compressed(args.output / "datasets_and_predictions.npz", **retained)
    report = {
        "protocol_sha256": hashlib.sha256(raw).hexdigest(),
        "identity_audit": audit,
        "classification": summary,
        "finite_shot": finite,
        "replicates": args.replicates,
        "quantum_hardware": False,
        "interpretation": "Any product-kernel performance is shared by its exact classical equivalent; no quantum advantage claim",
        "scope": protocol["scope"],
        "intervals": "pointwise percentile bootstrap over independent simulator replicates; 2000 resamples; no significance testing",
    }
    for name, content in (
        ("report.json", report),
        ("classification_rows.json", results),
        ("noise_rows.json", noise_results),
    ):
        (args.output / name).write_text(json.dumps(content, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
