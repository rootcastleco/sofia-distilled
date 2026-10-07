"""Verified product-kernel teacher -> tiny neural Edge student."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .edge import LABELS, softmax
from .edge_training import dataset, export, forward, initialize, metrics, train
from .kernels import CenteredRidge, balanced_accuracy, product_kernel, rbf_kernel

SCALES = (0.25, 0.5, 1.0, 2.0, 4.0)
PENALTIES = (1.0, 0.1, 0.01, 0.001, 0.0001)


def select_teacher(x, y, xv, yv, kernel):
    target = np.eye(len(LABELS))[y]
    best, choice = -1.0, None
    for scale in SCALES:
        k, q = kernel(x, x, scale), kernel(xv, x, scale)
        for penalty in PENALTIES:
            model = CenteredRidge().fit(k, target, penalty)
            score = balanced_accuracy(yv, model.predict(q).argmax(1))
            if score > best:
                best, choice = score, (scale, penalty)
    return choice, best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/edge-kernel"))
    parser.add_argument("--epochs", type=int, default=120)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "method": "product-rotation centred kernel ridge teacher + neural KL distillation",
        "source": "https://www.academia.edu/175377730/Quantum_Artificial_Intelligence_with_Verifiable_Kernels",
        "scales": SCALES,
        "ridge_penalties": PENALTIES,
        "epochs": args.epochs,
        "seed": 20261008,
        "teacher_fit": 600,
        "teacher_validation": 200,
        "student_train": 6000,
        "student_validation": 1000,
        "test": 1500,
        "ood_test": 1500,
        "selection": "teacher: first best validation balanced accuracy; student: minimum validation CE",
        "neural_temperature": 2.0,
        "kernel_score_temperature": 0.5,
        "alpha": 0.7,
        "quantum_hardware": False,
    }
    protocol_bytes = json.dumps(protocol, indent=2).encode()
    (args.output / "protocol.json").write_bytes(protocol_bytes)
    xt, yt = dataset(600, 20261008)
    xv, yv = dataset(200, 20261009)
    xu, yu = dataset(6000, 20261010)
    xuv, yuv = dataset(1000, 20261011)
    xs, ys = dataset(1500, 20261012)
    xo, yo = dataset(1500, 20261013, rpm_range=(43, 65))
    mean, scale = xt.mean(0), np.maximum(xt.std(0), 1e-6)
    norm = lambda x: (x - mean) / scale
    # A fixed arctangent feature encoding avoids unbounded angle aliasing.
    encode = lambda x: 2 * np.arctan(norm(x))
    te, ve = encode(xt), encode(xv)
    choice, val_score = select_teacher(te, yt, ve, yv, product_kernel)
    rbf_choice, rbf_val = select_teacher(te, yt, ve, yv, rbf_kernel)
    print(f"Product teacher selected={choice}, validation BA={val_score:.4f}", flush=True)
    anchors = np.concatenate([te, ve])
    target = np.eye(len(LABELS))[np.concatenate([yt, yv])]
    s, penalty = choice
    k = product_kernel(anchors, anchors, s)
    teacher = CenteredRidge().fit(k, target, penalty)
    q = lambda x: teacher.predict(product_kernel(encode(x), anchors, s))
    teacher_logits = q(xu) / 0.5
    targets = softmax(teacher_logits, 2.0)
    initial = initialize([14, 16, 5], np.random.default_rng(20261014))
    student, history = train(
        {k: v.copy() for k, v in initial.items()},
        norm(xu),
        yu,
        norm(xuv),
        yuv,
        epochs=args.epochs,
        seed=20261015,
        targets=targets,
    )
    control, hc = train(initial, norm(xu), yu, norm(xuv), yuv, epochs=args.epochs, seed=20261015)
    export(args.output, student, [14, 16, 5], mean, scale, "product-kernel-distilled-student")
    export(args.output / "control", control, [14, 16, 5], mean, scale, "supervised-control")
    np.savez(
        args.output / "kernel_teacher.npz",
        anchors=anchors,
        alpha=teacher.alpha,
        kernel_mean=teacher.mean,
        kernel_overall=teacher.overall,
        target_mean=teacher.ymean,
        mean=mean,
        scale=scale,
        kernel_scale=s,
        ridge_penalty=penalty,
    )
    np.savez(
        args.output / "dataset.npz",
        teacher_train=xt,
        teacher_train_y=yt,
        teacher_validation=xv,
        teacher_validation_y=yv,
        student_train=xu,
        student_train_y=yu,
        student_validation=xuv,
        student_validation_y=yuv,
        test=xs,
        test_y=ys,
        ood_test=xo,
        ood_test_y=yo,
        test_teacher_scores=q(xs),
        test_student_logits=forward(student, norm(xs))[-1],
        test_control_logits=forward(control, norm(xs))[-1],
    )
    rs, rp = rbf_choice
    rbf = CenteredRidge().fit(rbf_kernel(anchors, anchors, rs), target, rp)
    rbf_pred = rbf.predict(rbf_kernel(encode(xs), anchors, rs)).argmax(1)
    report = {
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "teacher_selected_scale": s,
        "teacher_selected_penalty": penalty,
        "teacher_validation_balanced_accuracy": val_score,
        "teacher_test_balanced_accuracy": balanced_accuracy(ys, q(xs).argmax(1)),
        "teacher_ood_balanced_accuracy": balanced_accuracy(yo, q(xo).argmax(1)),
        "rbf_selected": list(rbf_choice),
        "rbf_validation_balanced_accuracy": rbf_val,
        "rbf_test_balanced_accuracy": balanced_accuracy(ys, rbf_pred),
        "student": metrics(student, norm(xs), ys),
        "supervised_control": metrics(control, norm(xs), ys),
        "ood_student": metrics(student, norm(xo), yo),
        "student_teacher_agreement": float(
            (forward(student, norm(xs))[-1].argmax(1) == q(xs).argmax(1)).mean()
        ),
        "student_parameters": sum(v.size for v in student.values()),
        "teacher_anchor_count": len(anchors),
        "quantum_hardware": False,
        "limitations": [
            "Synthetic scenarios only; no field diagnostic validation",
            "Kernel has an exact efficient classical equivalent",
            "One model-training seed; no inferential benefit claim for distillation",
            "Scores/softmax are not calibrated confidence",
        ],
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output / "training_history.json").write_text(
        json.dumps({"student": history, "supervised_control": hc}), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
