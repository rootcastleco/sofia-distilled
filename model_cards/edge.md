---
language:
- en
license: apache-2.0
tags:
- sofia
- numpy
- knowledge-distillation
- quantum-inspired
- verifiable-kernels
- synthetic-data
- vibration-analysis
- experimental
---

# Sofia Edge Distilled v0.1

A **trained 325-parameter NumPy classifier** for five synthetic vibration scenarios: healthy, imbalance, misalignment, bearing impulses and rubbing. The teacher is centred kernel ridge regression using a product-rotation kernel with an exact efficient classical evaluation.

Developed by Rootcastle Engineering & Innovation. [Source and reproducible training](https://github.com/rootcastleco/sofia-distilled).

## Training

14 statistical/spectral features feed a 14 → 16 → 5 ReLU MLP. The teacher uses 600 fit examples and 200 validation examples; its 25-fit scale/penalty search is matched by an RBF control. After selection it is refitted on the 800 training/validation anchors. The student is trained for 120 epochs on 6,000 separately generated waveforms, with 1,000 student-validation examples for checkpoint selection.

The student objective combines hard-label cross-entropy and temperature-2 teacher-target KL. A supervised-only student with identical initialization, data, architecture and optimization budget is retained as a control. Train-only standardization precedes fixed arctangent angle encoding for the teacher.

## Held-out results

| Model | Synthetic test accuracy / balanced accuracy |
| --- | ---: |
| Product-kernel teacher | 99.933% |
| Matched-budget RBF teacher | 99.933% |
| Distilled student | 99.933% |
| Supervised-only student | 100.000% |

The balanced test set has 1,500 examples. The distilled student reaches 100.000% on a separately generated shaft-frequency shift of 43–65 Hz, versus 18–42 Hz during training. These results concern easy analytical formulas, not machine diagnosis. The control outperforms distillation on the in-distribution accuracy measure; no distillation quality advantage is claimed.

## Use

Download this repository, then run its standalone loader:

```python
import numpy as np
from inference import EdgeModel

model = EdgeModel(".")
signal = np.load("acceleration.npy", allow_pickle=False)
result = model.predict_signal(signal, sample_rate=2048, shaft_hz=25)
print(result)
```

Acceleration values must be in g. Training windows have 1,024 samples at 2,048 Hz. The loader validates feature order, dimensions, finiteness and the weight SHA-256 before inference.

## Artifacts and provenance

- `model.npz`: final float32 student parameters and training-only scaler.
- `config.json`: exact ordered feature schema and weight digest.
- `kernel_teacher.npz`: anchors, ridge coefficients, centring statistics and selected hyperparameters.
- `dataset.npz`: all fit/validation/test feature matrices, labels and held-out scores.
- `metrics.json`, `protocol.json`, `training_history.json`: measured results and run configuration.
- `verification_report.json`: independent kernel and noise control experiments.

The method follows [Quantum Artificial Intelligence with Verifiable Kernels](https://www.academia.edu/175377730/Quantum_Artificial_Intelligence_with_Verifiable_Kernels). Calculations use classical CPU hardware. Read `VERIFICATION.md` for exact scope.

## Limitations

No measured machine data, field validation or calibrated confidence. Softmax values are scores, not diagnostic certainty. The range check is a feature-distance heuristic. The model cannot validate arbitrary sensor placement, machines or units. It is intended for research and demonstration; it has no machinery actuation API. Only one Edge training seed was run. No quantum processor or quantum computational advantage is claimed.
