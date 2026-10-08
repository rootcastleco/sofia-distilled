"""Verify a public Hugging Face release against its local staging manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


def verify(kind, owner="rootcastleengineering", download_weights=False):
    folder = Path(__file__).resolve().parent.parent / "dist" / f"hf-{kind}"
    manifest = json.loads((folder / "SHA256SUMS.json").read_text())
    repo = f"{owner}/sofia-{kind}-distilled-v0.1"
    info = HfApi().model_info(repo, files_metadata=True)
    siblings = {item.rfilename: item for item in info.siblings}
    remote_manifest = Path(hf_hub_download(repo, "SHA256SUMS.json", revision=info.sha))
    if json.loads(remote_manifest.read_text()) != manifest:
        raise RuntimeError("Remote manifest differs from the staged release")
    modes = {}
    for name, expected in manifest.items():
        item = siblings[name]
        if name == "model.safetensors" and not download_weights:
            if item.lfs is None or item.lfs.sha256 != expected:
                raise RuntimeError("Remote model content digest differs")
            modes[name] = "Hub content-addressed SHA-256 metadata"
        else:
            downloaded = Path(hf_hub_download(repo, name, revision=info.sha))
            if hashlib.sha256(downloaded.read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"Remote file differs: {name}")
            modes[name] = "downloaded bytes"
    result = {"repo": repo, "revision": info.sha, "verified_files": len(manifest), "modes": modes}
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["edge", "chat"])
    parser.add_argument("--owner", default="rootcastleengineering")
    parser.add_argument("--download-weights", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.kind, args.owner, args.download_weights)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
