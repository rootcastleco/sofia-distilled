"""Prune Qwen layers and distill teacher logits with LoRA on one small GPU.

The shipped checkpoint is a derivative of Qwen, not a model pretrained from zero.
Teacher targets are cached on disk so the teacher can leave GPU memory before training.
KL uses the teacher's top-k tokens plus one aggregated tail-probability bucket.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np

from .chat_data import SYSTEM, prompt_bank

TEACHER = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


def kernel_projection(hidden_size, device):
    import torch

    rng = np.random.default_rng(20260911)
    return torch.tensor(
        rng.normal(0, 1 / np.sqrt(hidden_size), (hidden_size, 4)),
        device=device,
        dtype=torch.float32,
    )


def rotation_kernel_torch(angles, anchors):
    import torch

    return torch.cos(0.5 * (angles[None, :] - anchors) / 2).square().prod(dim=-1)


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def response_mask(tokenizer, prompt, response, max_length):
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    prefix = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(
        [*messages, {"role": "assistant", "content": response}], tokenize=True
    )
    # Mask assistant answer AND its end-of-turn token; never train on prompt positions.
    full = full[:max_length]
    labels = [-100] * min(len(prefix), len(full)) + full[len(prefix) :]
    if sum(v != -100 for v in labels) < 2:
        raise ValueError("max_length leaves no assistant tokens")
    return full, labels


def prepare_cache(teacher, tokenizer, output, device, args):
    import torch
    import torch.nn.functional as F

    cache = output / "teacher_cache"
    cache.mkdir(parents=True, exist_ok=True)
    rows = prompt_bank()
    for index, row in enumerate(rows):
        path = cache / f"{row['id']}.npz"
        if path.exists():
            continue
        with torch.inference_mode():
            response = row["response"]
            ids, labels = response_mask(tokenizer, row["prompt"], response, args.max_length)
            tokens = torch.tensor([ids], device=device)
            mask = torch.tensor(labels[1:], device=device) != -100
            outputs = teacher(tokens, use_cache=False, output_hidden_states=True)
            logits = outputs.logits[0, :-1][mask].float()
            hidden = outputs.hidden_states[-1][0, :-1][mask].float().mean(dim=0)
            angles = 2 * torch.atan(hidden @ kernel_projection(hidden.shape[-1], device))
            logp = F.log_softmax(logits / args.temperature, dim=-1)
            top_logp, top_ids = torch.topk(logp, args.top_k, dim=-1)
            tail = (1 - top_logp.exp().sum(-1)).clamp(min=1e-8)
            target_ids = torch.tensor(labels[1:], device=device)[mask]
            teacher_nll = F.cross_entropy(logits, target_ids, reduction="sum").item()
        np.savez_compressed(
            path,
            ids=np.asarray(ids, dtype=np.int32),
            labels=np.asarray(labels, dtype=np.int32),
            top_ids=top_ids.cpu().numpy().astype(np.int32),
            top_prob=top_logp.exp().cpu().numpy().astype(np.float32),
            tail_prob=tail.cpu().numpy().astype(np.float32),
            teacher_nll=np.asarray(teacher_nll),
            teacher_angles=angles.cpu().numpy(),
            response=np.asarray(response),
        )
        print(f"Teacher targets {index + 1}/{len(rows)} ({row['split']})", flush=True)
    pairs = []
    for row in rows:
        with np.load(cache / f"{row['id']}.npz", allow_pickle=False) as data:
            pairs.append(
                {
                    **row,
                    "response": str(data["response"]),
                    "teacher": TEACHER,
                    "teacher_revision": REVISION,
                }
            )
    (output / "dataset.jsonl").write_text(
        "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pairs), encoding="utf-8"
    )
    return rows


def load_sample(cache, row, device):
    import torch

    with np.load(cache / f"{row['id']}.npz", allow_pickle=False) as d:
        return {
            k: torch.from_numpy(d[k].copy()).to(device)
            for k in ("ids", "labels", "top_ids", "top_prob", "tail_prob", "teacher_angles")
        }


def losses(model, sample, temperature, anchors):
    import torch
    import torch.nn.functional as F

    ids = sample["ids"].long()[None]
    labels = sample["labels"].long()[1:]
    mask = labels != -100
    outputs = model(ids, use_cache=False, output_hidden_states=True)
    logits = outputs.logits[0, :-1][mask].float()
    ce = F.cross_entropy(logits, labels[mask])
    logp = F.log_softmax(logits / temperature, dim=-1)
    selected = logp.gather(-1, sample["top_ids"].long())
    student_tail = (1 - selected.exp().sum(-1)).clamp(min=1e-8)
    p = sample["top_prob"].float().clamp(min=1e-12)
    tail = sample["tail_prob"].float().clamp(min=1e-12)
    kl = (
        (p * (p.log() - selected)).sum(-1) + tail * (tail.log() - student_tail.log())
    ).mean() * temperature**2
    hidden = outputs.hidden_states[-1][0, :-1][mask].float().mean(dim=0)
    angles = 2 * torch.atan(hidden @ kernel_projection(hidden.shape[-1], hidden.device))
    reference = rotation_kernel_torch(sample["teacher_angles"].float(), anchors)
    geometry = F.mse_loss(rotation_kernel_torch(angles, anchors), reference)
    return ce, kl, geometry, int(mask.sum())


def evaluate(model, rows, cache, device, temperature, anchors):
    import torch

    model.eval()
    nll, kd, geo, count = 0.0, 0.0, 0.0, 0
    with torch.inference_mode():
        for row in rows:
            sample = load_sample(cache, row, device)
            ce, kl, geometry, n = losses(model, sample, temperature, anchors)
            nll += ce.item() * n
            kd += kl.item() * n
            geo += geometry.item() * n
            count += n
    return {
        "response_cross_entropy": nll / count,
        "perplexity": math.exp(nll / count),
        "coarsened_kl_times_temperature_squared": kd / count,
        "product_kernel_alignment_mse": geo / count,
        "response_tokens": count,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/chat"))
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kernel-beta", type=float, default=0.1)
    args = parser.parse_args()
    if not 1 <= args.layers < 24 or args.epochs < 1 or not 0 <= args.alpha <= 1:
        parser.error("layers must be 1-23, epochs >= 1 and alpha in [0,1]")
    if args.temperature <= 0 or not 1 <= args.top_k < 151936:
        parser.error("positive temperature and top-k in [1,151935] required")
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    run_config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    run_config.update(
        teacher=TEACHER,
        teacher_revision=REVISION,
        system_prompt=SYSTEM,
        data_version="original-references-v2",
    )
    config_path = args.output / "training_config.json"
    if config_path.exists():
        old = json.loads(config_path.read_text(encoding="utf-8"))
        for key in ("teacher_revision", "max_length", "temperature", "top_k", "data_version"):
            if old.get(key) != run_config[key]:
                raise ValueError(
                    f"cached targets incompatible with changed {key}; use a new output"
                )
    write_json(config_path, run_config)
    print(f"Device={device}, dtype={dtype}; loading pinned teacher", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(TEACHER, revision=REVISION)
    teacher = AutoModelForCausalLM.from_pretrained(
        TEACHER, revision=REVISION, torch_dtype=dtype, attn_implementation="sdpa"
    )
    teacher.eval().to(device)
    rows = prepare_cache(teacher, tokenizer, args.output, device, args)
    teacher_params = sum(p.numel() for p in teacher.parameters())
    # Copy selected blocks and retain teacher embeddings, output head and final RMSNorm.
    selected_layers = np.linspace(0, len(teacher.model.layers) - 1, args.layers).round().astype(int)
    student = copy.deepcopy(teacher.cpu())
    student.model.layers = torch.nn.ModuleList([student.model.layers[i] for i in selected_layers])
    for i, layer in enumerate(student.model.layers):
        layer.self_attn.layer_idx = i
    student.config.num_hidden_layers = args.layers
    student.config.max_window_layers = args.layers
    student.config.use_cache = False
    del teacher
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    student_params = sum(p.numel() for p in student.parameters())
    student.to(device)
    cache = args.output / "teacher_cache"
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "validation"]
    test_rows = [r for r in rows if r["split"] == "test"]
    anchors = torch.stack(
        [
            load_sample(cache, row, device)["teacher_angles"]
            for row in train_rows[:: max(1, len(train_rows) // 8)][:8]
        ]
    ).float()
    np.save(args.output / "kernel_landmarks.npy", anchors.cpu().numpy())
    baseline_val = evaluate(student, val_rows, cache, device, args.temperature, anchors)
    baseline_test = evaluate(student, test_rows, cache, device, args.temperature, anchors)
    print(f"Pruned baseline validation: {baseline_val}", flush=True)
    student = get_peft_model(
        student,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=16,
            lora_alpha=32,
            lora_dropout=0.0,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
    )
    trainable = [p for p in student.parameters() if p.requires_grad]
    trainable_params = sum(p.numel() for p in trainable)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    history, best, best_epoch, steps = [], float("inf"), 0, 0
    for epoch in range(args.epochs):
        student.train()
        shuffled = train_rows.copy()
        random.Random(args.seed + epoch).shuffle(shuffled)
        for row in shuffled:
            sample = load_sample(cache, row, device)
            optimizer.zero_grad(set_to_none=True)
            ce, kl, geometry, _ = losses(student, sample, args.temperature, anchors)
            loss = (1 - args.alpha) * ce + args.alpha * kl + args.kernel_beta * geometry
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
            optimizer.step()
            steps += 1
            if steps % 16 == 0:
                print(
                    f"epoch={epoch + 1} step={steps} CE={ce.item():.4f} KD={kl.item():.4f}",
                    flush=True,
                )
        val = evaluate(student, val_rows, cache, device, args.temperature, anchors)
        history.append({"epoch": epoch + 1, "steps": steps, **val})
        print(f"Validation epoch {epoch + 1}: {val}", flush=True)
        if val["response_cross_entropy"] < best:
            best = val["response_cross_entropy"]
            best_epoch = epoch + 1
            student.save_pretrained(args.output / "best_adapter", safe_serialization=True)
    # Load validation-selected adapter, then merge into ordinary Qwen2 weights.
    from peft import PeftModel

    base = student.unload()
    student = PeftModel.from_pretrained(base, args.output / "best_adapter").merge_and_unload()
    final_val = evaluate(student, val_rows, cache, device, args.temperature, anchors)
    final_test = evaluate(student, test_rows, cache, device, args.temperature, anchors)
    student.config.use_cache = True
    student.generation_config.do_sample = False
    student.save_pretrained(args.output, safe_serialization=True)
    tokenizer.save_pretrained(args.output)
    # Reload saved artifacts for qualitative test samples, verifying ordinary Transformers loading.
    del student, base, optimizer, trainable
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.output, torch_dtype=dtype, attn_implementation="sdpa"
        )
        .eval()
        .to(device)
    )
    examples = []
    data_rows = {
        r["id"]: r
        for r in [
            json.loads(line)
            for line in (args.output / "dataset.jsonl").read_text(encoding="utf-8").splitlines()
        ]
    }
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
                    "reference_response": data_rows[row["id"]]["response"],
                    "student_response": tokenizer.decode(
                        generated[0, inputs.input_ids.shape[-1] :], skip_special_tokens=True
                    ),
                }
            )
    teacher_nll, teacher_tokens = 0.0, 0
    for row in test_rows:
        with np.load(cache / f"{row['id']}.npz", allow_pickle=False) as d:
            teacher_nll += float(d["teacher_nll"])
            teacher_tokens += int((d["labels"][1:] != -100).sum())
    report = {
        "status": "experimental-pilot",
        "teacher": TEACHER,
        "teacher_revision": REVISION,
        "initialization": "selected pretrained teacher layers, embeddings, final norm and tied head",
        "selected_teacher_layers_zero_based": selected_layers.tolist(),
        "teacher_parameters": teacher_params,
        "student_parameters": student_params,
        "parameter_compression_ratio": teacher_params / student_params,
        "trainable_lora_parameters": trainable_params,
        "split": {"train": len(train_rows), "validation": len(val_rows), "test": len(test_rows)},
        "split_method": "disjoint original topic families; four prompt styles per topic",
        "baseline_validation": baseline_val,
        "baseline_test": baseline_test,
        "final_validation": final_val,
        "final_test": final_test,
        "teacher_test": {
            "response_cross_entropy": teacher_nll / teacher_tokens,
            "perplexity": math.exp(teacher_nll / teacher_tokens),
        },
        "best_epoch": best_epoch,
        "optimization_steps": steps,
        "distillation": "(1-alpha) assistant-token CE + alpha T^2 KL(top-k plus tail) + beta product-kernel alignment",
        "kernel_method": "4-D fixed Gaussian projection of mean response hidden state; 2*atan encoding; 8 training-only landmarks; rotation scale 0.5",
        "kernel_beta": args.kernel_beta,
        "quantum_hardware": False,
        "principle_source": "https://www.academia.edu/175377730/Quantum_Artificial_Intelligence_with_Verifiable_Kernels",
        "temperature": args.temperature,
        "top_k": args.top_k,
        "elapsed_seconds": time.perf_counter() - started,
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name() if device == "cuda" else "CPU",
        },
        "limitations": [
            "192 original prompts with 48 original reference answers; narrow pilot corpus",
            "Reference answers were authored for this release, not externally reviewed benchmark labels",
            "No broad language, coding, safety or industrial deployment benchmark",
            "Inherited Qwen pretraining, not foundation pretraining from scratch",
            "Kernel alignment is an engineering extension, not a theorem about LLM quality",
            "Pruning can reduce general language quality; inspect published test generations",
        ],
    }
    write_json(args.output / "metrics.json", report)
    write_json(args.output / "training_history.json", history)
    write_json(args.output / "test_generations.json", examples)
    hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in args.output.iterdir()
        if p.is_file() and p.name != "SHA256SUMS.json"
    }
    write_json(args.output / "SHA256SUMS.json", hashes)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
