# Verification and evidence scope

## Method source and scope

[Quantum Artificial Intelligence with Verifiable Kernels](https://www.academia.edu/175377730/Quantum_Artificial_Intelligence_with_Verifiable_Kernels), Batuhan Ayribas, September 2026, informs the kernel controls. This repository contains an independently written adaptation. Its new random seeds and outputs are not the manuscript's original supplementary archive.

All experiments here use classical numerical computation. A feature geometry described using quantum states can have a useful classical representation. Predictive scores alone do not establish quantum advantage.

## Evidence map

| Claim | Implementation | Retained evidence |
| --- | --- | --- |
| Product kernel equals a state-overlap calculation | `product_kernel`, `product_states` | `identity_audit`, parameterized tests |
| Product kernel equals explicit classical features | `trig_features` | maximum discrepancy across dimensions 1, 2, 4, 6, 8 |
| A shared terminal unitary preserves overlap | independent orthogonal matrices applied to state vectors | numerical discrepancy |
| Global depolarization produces the specified affine kernel | independent density matrices | density-versus-affine discrepancy |
| Matched regularization preserves exact centred ridge predictions | `CenteredRidge`, `depolarize` | prediction discrepancy, separate exact and shot results |
| Finite-shot measurement does not inherit exact invariance | binomial SWAP simulation, positive training eigenspace | 30 replicates at each noise/shot condition |
| Kernel teacher can be compressed into an Edge MLP | `kernel_training` | held-out predictions, supervised-only student control |
| Chat representation alignment actually appears in the loss | `chat_training` | training configuration and validation kernel MSE |

## Experiment protocol

Each of three small classification tasks uses 30 independent train/validation/test datasets. Product and RBF models use 25 validation fits each; linear uses five. Hyperparameters are selected on validation labels and then refitted on the training/validation union. Test labels are reserved for final reporting. Intervals are pointwise percentile bootstraps over the simulator replicates, with 2,000 resamples. No multiplicity-adjusted significance claim is made.

Finite-shot conditions use a separate fixed product model, a global state-depolarization channel, an ideal SWAP measurement estimator, and positive-eigenspace repair fitted using the training matrix. Negative overlap estimates are retained. No shot data are described as real device measurements.

The protocol is saved and hashed before computation. `datasets_and_predictions.npz` retains inputs, labels and score arrays. `classification_rows.json` and `noise_rows.json` retain seed-level metrics.

## Measured checks in the first local run

| Identity | Maximum absolute discrepancy |
| --- | ---: |
| State overlap versus direct classical kernel | 1.55e-15 |
| Explicit classical features versus direct kernel | 1.89e-15 |
| Shared terminal unitary | 1.78e-15 |
| Density-matrix global depolarization | 8.88e-16 |
| Matched-regularization prediction | 1.37e-13 |

These finite checks support the implementations at tested inputs; they do not replace mathematical proofs or certify a physical device. Original numerical reports, rather than rounded values in this document, are authoritative.

## Model-specific boundaries

Edge uses newly generated vibration formulas rather than recorded machinery data. Its angle encoding is a fixed `2 * atan` transformation after training-only feature standardization. Product-kernel and RBF teacher selection use identical scale/penalty budgets. The 325-parameter student combines hard-label cross-entropy with soft teacher targets. These targets are softened ridge scores, not calibrated probabilities.

Chat uses pretrained Qwen blocks, a frozen shared embedding/head and trainable LoRA adapters. Its kernel term compares four-dimensional sketches of response representations against eight training-only teacher landmarks. The sketch and angle map are fixed. This is a heuristic representation-alignment objective. The manuscript's exact ridge-noise identity applies to its specified kernel learner, not to the transformer or its generated answers.

The Chat test set contains different topic families from the training and validation sets. It uses original reference answers, with teacher logits computed on the reference sequence. A lower reference cross-entropy on this small bank is not a factuality, coding or safety benchmark. The references were authored with AI assistance and have not been externally reviewed. Broad quality and the separate effect of the kernel term require additional data, multiple training seeds, independent references and controlled ablations.
