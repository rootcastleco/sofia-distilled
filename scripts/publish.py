"""Stage explicitly selected trained model files, then optionally upload to HF."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def stage(kind):
    source = ROOT / "artifacts" / ("edge-kernel" if kind == "edge" else "chat")
    destination = ROOT / "dist" / f"hf-{kind}"
    required = ["config.json", "metrics.json", "training_history.json"]
    required += (
        ["model.npz", "kernel_teacher.npz", "dataset.npz", "protocol.json"]
        if kind == "edge"
        else [
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
            "special_tokens_map.json",
            "added_tokens.json",
            "vocab.json",
            "merges.txt",
            "generation_config.json",
            "dataset.jsonl",
            "training_config.json",
            "test_generations.json",
            "kernel_landmarks.npy",
        ]
    )
    destination.mkdir(parents=True, exist_ok=True)
    for filename in required:
        path = source / filename
        if not path.is_file():
            raise FileNotFoundError(f"Training incomplete: missing {path}")
        shutil.copy2(path, destination / filename)
    for filename in ("LICENSE", "NOTICE"):
        shutil.copy2(ROOT / filename, destination / filename)
    shutil.copy2(ROOT / "model_cards" / f"{kind}.md", destination / "README.md")
    shutil.copy2(ROOT / "docs" / "VERIFICATION.md", destination / "VERIFICATION.md")
    shutil.copy2(
        ROOT / "artifacts" / "verification" / "report.json",
        destination / "verification_report.json",
    )
    shutil.copy2(
        ROOT / "artifacts" / "verification" / "reproducibility.json",
        destination / "reproducibility.json",
    )
    if kind == "edge":
        # A standalone NumPy loader is also included in the model repo.
        shutil.copy2(ROOT / "src" / "sofia_distilled" / "edge.py", destination / "inference.py")
    sums = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in destination.iterdir()
        if p.is_file() and p.name != "SHA256SUMS.json"
    }
    (destination / "SHA256SUMS.json").write_text(json.dumps(sums, indent=2), encoding="utf-8")
    return destination


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["edge", "chat"])
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--owner", default="rootcastleengineering")
    args = parser.parse_args()
    folder = stage(args.kind)
    print(f"Verified staging files: {folder}")
    if args.upload:
        from huggingface_hub import HfApi, get_token

        if not get_token():
            raise SystemExit("Hugging Face authentication required: hf auth login")
        api = HfApi()
        repo_id = f"{args.owner}/sofia-{args.kind}-distilled-v0.1"
        api.create_repo(repo_id, repo_type="model", exist_ok=True)
        commit = api.upload_folder(
            folder_path=folder,
            repo_id=repo_id,
            commit_message="Publish trained Sofia v0.1 pilot with verifiable kernel evidence",
        )
        print(commit)
        remote = api.model_info(repo_id, files_metadata=True)
        local_hashes = json.loads((folder / "SHA256SUMS.json").read_text())
        names = {s.rfilename for s in remote.siblings}
        if not set(local_hashes) <= names:
            raise RuntimeError("remote files incomplete")
        print(f"Published and verified https://huggingface.co/{repo_id} at {remote.sha}")


if __name__ == "__main__":
    main()
