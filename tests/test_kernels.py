import numpy as np
import pytest

from sofia_distilled.kernels import (
    CenteredRidge,
    depolarize,
    product_kernel,
    product_states,
    swap_estimate,
    trig_features,
)
from sofia_distilled.verification import identity_audit


@pytest.mark.parametrize("dim", [1, 2, 4, 6, 8])
def test_three_independent_kernel_implementations(dim):
    x = np.random.default_rng(dim).normal(size=(7, dim))
    expected = product_kernel(x, x, 0.5)
    states, features = product_states(x, 0.5), trig_features(x, 0.5)
    np.testing.assert_allclose(expected, (states @ states.T) ** 2, atol=5e-14)
    np.testing.assert_allclose(expected, features @ features.T, atol=5e-14)
    assert np.linalg.eigvalsh(expected).min() > -1e-12


def test_identity_and_noise_audit():
    assert identity_audit()["passed"]


def test_finite_shot_moments_and_negative_estimates():
    rng = np.random.default_rng(8)
    samples = swap_estimate(np.full((100000, 1), 0.3), 128, rng)
    assert abs(samples.mean() - 0.3) < 0.002
    assert abs(samples.var() - (1 - 0.3**2) / 128) < 0.0002
    zero = swap_estimate(np.zeros((1000, 1)), 16, rng)
    assert (zero < 0).any()


def test_symmetric_shots_include_noisy_diagonal():
    k = depolarize(np.eye(8), 0.75, 16)
    estimated = swap_estimate(k, 128, np.random.default_rng(42), symmetric=True)
    np.testing.assert_array_equal(estimated, estimated.T)
    assert not np.allclose(np.diag(estimated), 1)


def test_complete_depolarization_predicts_intercept():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(10, 4))
    y = rng.normal(size=(10, 3))
    k = depolarize(product_kernel(x, x), 1, 16)
    q = np.full((4, 10), 1 / 16)
    np.testing.assert_allclose(CenteredRidge().fit(k, y).predict(q), np.tile(y.mean(0), (4, 1)))


def test_positive_eigenspace_out_of_sample_and_intercept():
    k = np.array([[1.0, -0.9, 0.2], [-0.9, 1, 0.3], [0.2, 0.3, 0.1]])
    y = np.array([1, -1, 1])
    q = np.array([[0.3, 0.4, 0.2]])
    model = CenteredRidge().fit(k, y, positive_eigenspace=True)
    # Adding a constant to every kernel entry must cancel through training-only centring.
    shifted = CenteredRidge().fit(k + 0.7, y, positive_eigenspace=True)
    np.testing.assert_allclose(model.predict(q), shifted.predict(q + 0.7), atol=1e-12)


@pytest.mark.parametrize("bad", [np.array([1, 2]), np.array([[np.nan]]), np.empty((0, 2))])
def test_bad_kernel_inputs_rejected(bad):
    with pytest.raises(ValueError):
        product_kernel(bad, bad)


@pytest.mark.parametrize("p", [-0.1, 1.1, float("nan")])
def test_bad_noise_rejected(p):
    with pytest.raises(ValueError):
        depolarize(np.eye(2), p, 4)
