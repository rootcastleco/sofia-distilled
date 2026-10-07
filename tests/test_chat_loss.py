"""Scientific checks for masking and the differentiable classical kernel term."""

import importlib.util

import numpy as np
import pytest

from sofia_distilled.kernels import product_kernel

HAS_CHAT = importlib.util.find_spec("torch") and importlib.util.find_spec("transformers")
pytestmark = pytest.mark.skipif(not HAS_CHAT, reason="optional chat dependencies")


def test_torch_kernel_matches_independent_numpy_and_has_gradients():
    import torch

    from sofia_distilled.chat_training import rotation_kernel_torch

    rng = np.random.default_rng(4)
    a, z = rng.normal(size=4), rng.normal(size=(8, 4))
    angles = torch.tensor(a, dtype=torch.float64, requires_grad=True)
    anchors = torch.tensor(z, dtype=torch.float64)
    result = rotation_kernel_torch(angles, anchors)
    np.testing.assert_allclose(result.detach().numpy(), product_kernel(a[None], z, 0.5)[0])
    result.sum().backward()
    assert torch.isfinite(angles.grad).all()
    assert angles.grad.abs().sum() > 0


def test_assistant_mask_excludes_prompt_and_includes_end_turn():
    from huggingface_hub import try_to_load_from_cache
    from transformers import AutoTokenizer

    from sofia_distilled.chat_training import REVISION, TEACHER, response_mask

    if not isinstance(try_to_load_from_cache(TEACHER, "tokenizer.json", revision=REVISION), str):
        pytest.skip("pinned teacher tokenizer is not locally cached")
    tokenizer = AutoTokenizer.from_pretrained(TEACHER, revision=REVISION, local_files_only=True)
    ids, labels = response_mask(tokenizer, "What is RMS?", "RMS is sqrt(mean(x**2)).", 192)
    assert len(ids) == len(labels)
    active = [i for i, value in enumerate(labels) if value != -100]
    assert min(active) > 10
    assert all(labels[i] == ids[i] for i in active)
    assert tokenizer.eos_token_id in [labels[i] for i in active]
