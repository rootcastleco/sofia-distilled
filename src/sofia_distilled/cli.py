"""Local inference for Edge, Chat and an advisory combined workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .chat_data import SYSTEM
from .edge import EdgeModel, waveform


def generate(model_path, prompt, max_new_tokens=160):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit('Install chat extras: pip install -e ".[chat]"') from exc
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, use_safetensors=True, trust_remote_code=False
    ).to(device)
    model.eval()
    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)
    if inputs.input_ids.shape[1] + max_new_tokens > model.config.max_position_embeddings:
        raise ValueError("prompt exceeds model context window")
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0, inputs.input_ids.shape[-1] :], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Experimental Sofia distilled models")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("edge", "assist"):
        p = sub.add_parser(command)
        p.add_argument("--edge-model", type=Path, default=Path("artifacts/edge-kernel"))
        p.add_argument("--signal", type=Path, help="1-D .npy acceleration waveform, unit g")
        p.add_argument("--sample-rate", type=float, default=2048)
        p.add_argument("--shaft-hz", type=float, default=25)
        p.add_argument("--demo-class", type=int, choices=range(5), default=3)
        if command == "assist":
            p.add_argument("--chat-model", default="artifacts/chat")
    chat = sub.add_parser("chat")
    chat.add_argument("--model", default="artifacts/chat")
    chat.add_argument("--prompt", required=True)
    args = parser.parse_args()
    if args.command == "chat":
        print(generate(args.model, args.prompt))
        return
    signal = (
        np.load(args.signal, allow_pickle=False)
        if args.signal
        else waveform(
            args.demo_class,
            np.random.default_rng(99),
            shaft_hz=args.shaft_hz,
            sample_rate=args.sample_rate,
        )
    )
    evidence = EdgeModel(args.edge_model).predict_signal(signal, args.sample_rate, args.shaft_hz)
    output = {"edge_evidence": evidence}
    if args.command == "assist":
        prompt = (
            "Explain this experimental synthetic-only classifier result in English. "
            "Do not present it as a confirmed machine diagnosis. State that physical inspection "
            "is needed. Result: " + json.dumps(evidence)
        )
        output["unverified_chat_explanation"] = generate(args.chat_model, prompt)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
