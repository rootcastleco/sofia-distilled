"""Verify and finish an already trained/exported Chat checkpoint after a tooling error.

No optimization happens here. Metrics are recomputed from the cached original references,
and the ordinary Transformers checkpoint is reloaded before qualitative evaluation.
"""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

from sofia_distilled.chat_data import SYSTEM, prompt_bank
from sofia_distilled.chat_training import REVISION, TEACHER, evaluate


def main():
    output = Path("artifacts/chat")
    config_path = output / "config.json"
    config = json.loads(config_path.read_text())
    # Transformers 4.57 materializes a per-layer list absent from older Qwen configs.
    config["layer_types"] = ["full_attention"] * config["num_hidden_layers"]
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    run = json.loads((output / "training_config.json").read_text())
    torch.manual_seed(run["seed"])
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(output, local_files_only=True)
    rows = prompt_bank()
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "validation"]
    test_rows = [r for r in rows if r["split"] == "test"]
    cache = output / "teacher_cache"
    anchors = torch.tensor(np.load(output / "kernel_landmarks.npy"), device=device)
    teacher = AutoModelForCausalLM.from_pretrained(
        TEACHER,
        revision=REVISION,
        torch_dtype=dtype,
        local_files_only=True,
        attn_implementation="sdpa",
    )
    teacher_params = sum(p.numel() for p in teacher.parameters())
    selected = np.linspace(0, len(teacher.model.layers) - 1, run["layers"]).round().astype(int)
    baseline = copy.deepcopy(teacher)
    baseline.model.layers = torch.nn.ModuleList([baseline.model.layers[i] for i in selected])
    for index, layer in enumerate(baseline.model.layers):
        layer.self_attn.layer_idx = index
    baseline.config.num_hidden_layers = run["layers"]
    baseline.config.layer_types = ["full_attention"] * run["layers"]
    baseline.config.use_cache = False
    del teacher
    baseline.to(device).eval()
    baseline_val = evaluate(baseline, val_rows, cache, device, run["temperature"], anchors)
    baseline_test = evaluate(baseline, test_rows, cache, device, run["temperature"], anchors)
    del baseline
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    model = (
        AutoModelForCausalLM.from_pretrained(
            output,
            torch_dtype=dtype,
            local_files_only=True,
            use_safetensors=True,
            attn_implementation="sdpa",
        )
        .to(device)
        .eval()
    )
    final_val = evaluate(model, val_rows, cache, device, run["temperature"], anchors)
    final_test = evaluate(model, test_rows, cache, device, run["temperature"], anchors)
    history = json.loads((output / "training_history.json").read_text())
    best = min(history, key=lambda r: r["response_cross_entropy"])
    if abs(final_val["response_cross_entropy"] - best["response_cross_entropy"]) > 0.1:
        raise RuntimeError("Merged checkpoint differs materially from selected adapter")
    examples = []
    with torch.inference_mode():
        for row in test_rows[::4]:
            text = tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM}, {"role": "user", "content": row["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(text, return_tensors="pt").to(device)
            generated = model.generate(
                **inputs, max_new_tokens=128, do_sample=False, pad_token_id=tokenizer.eos_token_id
            )
            examples.append(
                {
                    "prompt": row["prompt"],
                    "reference_response": row["response"],
                    "student_response": tokenizer.decode(
                        generated[0, inputs.input_ids.shape[-1] :], skip_special_tokens=True
                    ),
                }
            )
    nll, tokens = 0.0, 0
    for row in test_rows:
        with np.load(cache / f"{row['id']}.npz", allow_pickle=False) as d:
            nll += float(d["teacher_nll"])
            tokens += int((d["labels"][1:] != -100).sum())
    with safe_open(output / "best_adapter" / "adapter_model.safetensors", framework="np") as f:
        # safe_open exposes keys(), and is not a dictionary or an iterable container.
        trainable = sum(np.prod(f.get_slice(key).get_shape()) for key in f.keys())  # noqa: SIM118
    student_params = sum(p.numel() for p in model.parameters())
    report = {
        "status": "experimental-pilot",
        "teacher": TEACHER,
        "teacher_revision": REVISION,
        "initialization": "selected pretrained teacher layers, embeddings, final norm and tied head",
        "selected_teacher_layers_zero_based": selected.tolist(),
        "teacher_parameters": teacher_params,
        "student_parameters": student_params,
        "parameter_compression_ratio": teacher_params / student_params,
        "trainable_lora_parameters": int(trainable),
        "split": {"train": len(train_rows), "validation": len(val_rows), "test": len(test_rows)},
        "split_method": "disjoint original topic families; four prompt styles per reference topic",
        "baseline_validation": baseline_val,
        "baseline_test": baseline_test,
        "final_validation": final_val,
        "final_test": final_test,
        "teacher_test": {
            "response_cross_entropy": nll / tokens,
            "perplexity": math.exp(nll / tokens),
        },
        "best_epoch": best["epoch"],
        "optimization_steps": history[-1]["steps"],
        "distillation": "0.3 assistant-token CE + 0.7 T^2 KL(top-64 plus tail) + 0.1 product-kernel alignment",
        "kernel_method": "fixed 4-D Gaussian sketch, 2*atan encoding, 8 training-only landmarks, rotation scale 0.5",
        "kernel_beta": run["kernel_beta"],
        "temperature": run["temperature"],
        "top_k": run["top_k"],
        "architecture_selection": {
            "candidates": [12, 18],
            "metric": "validation reference cross-entropy",
            "candidate_12_cross_entropy": 6.182507134065395,
            "selected_layers": 18,
        },
        "quantum_hardware": False,
        "principle_source": "https://www.academia.edu/175377730/Quantum_Artificial_Intelligence_with_Verifiable_Kernels",
        "elapsed_seconds_including_export_recovery": time.time()
        - (output / "training_config.json").stat().st_mtime,
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "numpy": np.__version__,
            "device": torch.cuda.get_device_name() if device == "cuda" else "CPU",
        },
        "limitations": [
            "48 original AI-assisted reference answers; no external expert review",
            "Narrow reference likelihood does not establish factual, coding or safety quality",
            "No independent language benchmark or kernel-specific ablation",
            "Inherited Qwen pretraining; not foundation pretraining from random initialization",
            "Classical kernel alignment extension; no quantum advantage claim",
        ],
        "export_verification": "merged safetensors reloaded through ordinary Transformers; validation metrics recomputed",
    }
    for name, value in (("metrics.json", report), ("test_generations.json", examples)):
        (output / name).write_text(json.dumps(value, indent=2), encoding="utf-8")
    sums = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in output.iterdir()
        if p.is_file() and p.name != "SHA256SUMS.json"
    }
    (output / "SHA256SUMS.json").write_text(json.dumps(sums, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(json.dumps(examples, indent=2), flush=True)


if __name__ == "__main__":
    main()
