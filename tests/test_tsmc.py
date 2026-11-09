import numpy as np

from prism.tsmc import (
    _gradients,
    _kernel,
    _transform,
    fit_sensitivity_contrast,
    predict_sensitivity_contrast,
)


def make_problem():
    rng = np.random.default_rng(23)

    def features(size):
        return {key: rng.normal(size=(size, width)) for key, width in (
            ("low", 4), ("high", 4), ("raw", 7)
        )}

    return features(20), np.array([0] * 14 + [1] * 6), features(30), features(9)


def test_input_gradients_match_finite_differences():
    train, labels, reference, query = make_problem()
    model = fit_sensitivity_contrast(train, labels, reference)
    x = _transform(query, model["feature_stats"])
    branch = model["identity"]
    analytic = _gradients(branch, x)
    numeric = np.zeros_like(analytic)
    for column in range(x.shape[1]):
        delta = np.zeros_like(x)
        delta[:, column] = 1e-6
        upper = _kernel(x + delta, branch["anchors"], branch["diagonal"], branch["bandwidth"])
        lower = _kernel(x - delta, branch["anchors"], branch["diagonal"], branch["bandwidth"])
        numeric[:, column] = ((upper - lower) @ branch["dual"]) / 2e-6
    np.testing.assert_allclose(analytic, numeric, atol=2e-9)


def test_metric_floor_and_zero_strength():
    train, labels, reference, query = make_problem()
    model = fit_sensitivity_contrast(train, labels, reference)
    assert model["learned"]["diagonal"].min() >= 0.1 - 1e-12
    np.testing.assert_allclose(model["learned"]["diagonal"].sum(), 15)

    identity = fit_sensitivity_contrast(train, labels, reference, strength=0)
    np.testing.assert_array_equal(predict_sensitivity_contrast(identity, query), np.zeros(9))


def test_query_order_does_not_change_scores():
    train, labels, reference, query = make_problem()
    model = fit_sensitivity_contrast(train, labels, reference)
    scores = predict_sensitivity_contrast(model, query)
    reverse = np.arange(len(scores))[::-1]
    reordered = {key: value[reverse] for key, value in query.items()}
    np.testing.assert_allclose(predict_sensitivity_contrast(model, reordered)[reverse], scores)
