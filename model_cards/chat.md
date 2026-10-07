---
language:
- en
license: apache-2.0
base_model:
- Qwen/Qwen2.5-0.5B-Instruct
base_model_relation: distillation
library_name: transformers
pipeline_tag: text-generation
tags:
- sofia
- knowledge-distillation
- qwen2
- technical-assistant
- verifiable-kernels
- experimental
---

# Sofia Chat Distilled v0.1

An experimental English technical assistant built by pruning and distilling **Qwen2.5-0.5B-Instruct**. Selected teacher blocks initialize a 12-layer student. The student retains the teacher's embeddings and tokenizer. LoRA learns from teacher output distributions and a product-rotation kernel alignment objective; the selected adapter is merged into standard Transformers weights.

Developed by Rootcastle Engineering & Innovation. [Source and reproducible training](https://github.com/rootcastleco/sofia-distilled).

This model inherits Qwen pretraining. It is not a foundation model pretrained from random initialization, and the narrow pilot dataset does not establish general language or engineering competence.

## Data and objective

192 original English prompts cover engineering, signal processing, Python, ML and verifiable-kernel concepts. Each of 48 topics has an original concise reference answer shared across four prompt styles. The teacher supplies token probabilities and representation targets on those reference sequences. Entire topic families are assigned to train/validation/test partitions, with 160/16/16 prompt examples. Validation selects the adapter checkpoint; test topics are reserved for reporting. References were written with AI assistance for this release and have not undergone external expert review.

Training uses assistant-token hard cross-entropy, temperature-scaled KL over teacher top-64 tokens plus an aggregated tail bucket, and a kernel alignment penalty. The latter compares fixed four-dimensional sketches of response hidden states against eight teacher landmarks chosen only from training samples. It is an engineering extension inspired by [Quantum Artificial Intelligence with Verifiable Kernels](https://www.academia.edu/175377730/Quantum_Artificial_Intelligence_with_Verifiable_Kernels); the manuscript does not prove a language-model benefit for this extension.

Teacher revision: `7ae557604adf67be50417f59c2c2f167def9a775`. Default run: 12 selected layers, rank-16 LoRA, four epochs, learning rate 0.0002, sequence length 192, seed 42. Exact configuration and measured metrics accompany the checkpoint.

## Evaluation

`metrics.json` reports response cross-entropy, perplexity and kernel-alignment MSE for the pruned baseline and the selected distilled model. `test_generations.json` exposes actual held-out reference and student answers. Token likelihood on this narrow reference bank is not an independent factuality, coding or safety benchmark. No broad quality advantage or quantum advantage is claimed.

## Use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

repo = "rootcastleengineering/sofia-chat-distilled-v0.1"
tokenizer = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, use_safetensors=True)
messages = [
    {"role": "system", "content": "You are Sofia, a concise English technical assistant."},
    {"role": "user", "content": "Explain why synthetic fault data require field validation."},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt")
with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=128, do_sample=False,
                            pad_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(output[0, inputs.input_ids.shape[-1]:], skip_special_tokens=True))
```

## Limitations and attribution

Small original reference corpus, one training seed, no independent expert answer verification and no broad benchmark. Answers may be false, repetitive or incomplete. Pruning can reduce general language quality. The trained context was short; upstream positional configuration is not proof of long-context competence. Kernel alignment is an explicit classical regularizer, not a quantum language-model architecture. Neither model performs machinery control.

Apache-2.0; Qwen weights and tokenizer remain subject to their upstream Apache-2.0 attribution. See `LICENSE` and `NOTICE`. Final weights use safetensors, with SHA-256 digests and pinned upstream provenance.
