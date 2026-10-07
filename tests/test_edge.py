import json
from pathlib import Path

import numpy as np
import pytest

from sofia_distilled.edge import EdgeModel, extract_features, waveform
from sofia_distilled.edge_training import export, forward, initialize


def test_reload_predictions_and_corruption(tmp_path):
    rng = np.random.default_rng(42)
    weights = initialize([14, 16, 5], rng)
    export(tmp_path, weights, [14, 16, 5], np.zeros(14), np.ones(14), "test")
    model = EdgeModel(tmp_path)
    x = rng.normal(size=(8, 14))
    from sofia_distilled.edge import softmax

    np.testing.assert_allclose(
        model.predict_features(x), softmax(forward(weights, x)[-1]), rtol=1e-5
    )
    (tmp_path / "model.npz").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA-256"):
        EdgeModel(tmp_path)


def test_feature_schema_rejected(tmp_path):
    config = {"feature_schema": "unknown"}
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="schema"):
        EdgeModel(tmp_path)


@pytest.mark.parametrize("signal", [np.zeros(1024), np.full(1024, np.nan), np.zeros(5)])
def test_invalid_signals_rejected(signal):
    with pytest.raises(ValueError):
        extract_features(signal, 2048, 25)


def test_heldout_saved_student():
    path = Path("artifacts/edge-kernel")
    if not path.exists():
        pytest.skip("run kernel_training first")
    model = EdgeModel(path)
    with np.load(path / "dataset.npz", allow_pickle=False) as data:
        predictions = model.predict_features(data["test"])
        assert np.mean(predictions.argmax(1) == data["test_y"]) > 0.95
        np.testing.assert_allclose(predictions.sum(1), 1)
    report = model.predict_signal(waveform(2, np.random.default_rng(98), shaft_hz=25), 2048, 25)
    assert report["label"] == "misalignment"
