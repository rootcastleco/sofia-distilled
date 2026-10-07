# Sofia Distilled

Two experimental models with trained weights, reproducible distillation, and explicit scientific controls:

- **Sofia Edge**: a 325-parameter NumPy vibration classifier distilled from a verified product-rotation kernel ridge teacher.
- **Sofia Chat**: a smaller Qwen-derived English technical assistant, trained with teacher-logit distillation and product-kernel representation alignment.

Developed by Rootcastle Engineering & Innovation. This is a new model-training repository associated with [Sofia Engine](https://github.com/rootcastleco/sofia-ai).

| Artifact | Distribution |
| --- | --- |
| Source, tests and experiment data | [GitHub](https://github.com/rootcastleco/sofia-distilled) |
| Edge model | [Hugging Face](https://huggingface.co/rootcastleengineering/sofia-edge-distilled-v0.1) |
| Chat model | [Hugging Face](https://huggingface.co/rootcastleengineering/sofia-chat-distilled-v0.1) |

## Scientific principle

The training design follows [Quantum Artificial Intelligence with Verifiable Kernels](https://www.academia.edu/175377730/Quantum_Artificial_Intelligence_with_Verifiable_Kernels) by Batuhan Ayribas (2026). The product kernel is evaluated classically and checked against independently constructed state vectors and trigonometric features. Shared final-unitary invariance, global-depolarization identities and matched ridge predictions are numerically audited. Finite-shot SWAP estimates are tested separately with training-only positive-eigenspace repair.

The released models run on ordinary CPU/GPU hardware. No quantum processor was used and no quantum computational advantage is claimed. The Chat alignment objective is a new engineering adaptation of that kernel geometry, not a result proved by the manuscript.

## Install

Python 3.11 or later:

```bash
git clone https://github.com/rootcastleco/sofia-distilled.git
cd sofia-distilled
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -e '.[dev,hub]'
```

For Chat, install a PyTorch build suitable for your GPU from the [official installer](https://pytorch.org/get-started/locally/), then:

```bash
pip install -e '.[chat]'
```

The initial Chat training run uses PyTorch 2.7.1 with CUDA 11.8 on an RTX 2060. It uses LoRA to keep optimizer memory small and merges the selected adapter into ordinary Transformers weights.

## Run the models

The Edge weights and evidence are included in this repository:

```bash
sofia-distilled edge --demo-class 3
sofia-distilled edge --signal acceleration.npy --sample-rate 2048 --shaft-hz 25
```

Signal values must be acceleration in **g**. The synthetic training specification uses 1,024-sample windows at 2,048 Hz. `max_probability` is an uncalibrated softmax score; the range check is a simple heuristic.

Chat weights are distributed through Hugging Face, avoiding large binaries in Git history:

```bash
sofia-distilled chat --model rootcastleengineering/sofia-chat-distilled-v0.1 --prompt "Explain spectral leakage briefly."
```

Download Chat locally if you want the combined workflow:

```bash
hf download rootcastleengineering/sofia-chat-distilled-v0.1 --local-dir artifacts/chat
sofia-distilled assist --demo-class 3
```

The combined command returns structured Edge evidence and a separately marked, unverified Chat explanation. It exposes no machinery control interface.

## Reproduce training

```bash
# Set OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1 for CPU experiment reproducibility.
python -m sofia_distilled.kernel_training --output artifacts/edge-kernel --epochs 120
python -m sofia_distilled.chat_training --output artifacts/chat --layers 18 --epochs 4
python -m sofia_distilled.verification --output reproduced-verification --replicates 30
pytest -q
```

`verification` refuses to overwrite a completed report. Both teacher targets and original Chat prompts are retained; cached targets are checked against compatible training configuration. The teacher is pinned to Qwen revision `7ae557604adf67be50417f59c2c2f167def9a775`.

## Evidence and limitations

See [verification](docs/VERIFICATION.md), [Edge model card](model_cards/edge.md), [Chat model card](model_cards/chat.md), and machine-readable reports under `artifacts/`.

The signal dataset is synthetic. High accuracy on these simple formulas does not establish field diagnostic accuracy. The supervised-only Edge control is retained and can outperform the distilled model. Chat uses 48 original reference answers across 192 prompts; the teacher provides soft-token targets. Reference perplexity on this small bank does not establish independent factual accuracy. Sofia Chat inherits Qwen pretraining; this project does not claim foundation-model pretraining from random initialization.

The released 18-layer Chat checkpoint has approximately 405 million parameters (18.11% fewer than the teacher). Held-out reference perplexity improves from 10,327.21 after pruning to 84.14 after distillation, but actual held-out generations still contain factual errors. It is an experimental checkpoint for auditing and further research, rather than a dependable engineering adviser.

## Publish a verified bundle

```bash
python scripts/publish.py edge
python scripts/publish.py chat
# After a local `hf auth login`:
python scripts/publish.py edge --upload
python scripts/publish.py chat --upload
```

The staging script requires completed training outputs and selects named files. Model weights are `.npz` or `.safetensors`, with SHA-256 manifests. License: Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE) for Qwen attribution.
