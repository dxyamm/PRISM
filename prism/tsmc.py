"""Task-sensitive metric contrast for frozen node representations."""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve

VIEWS = ('low', 'high', 'raw')


def _views(data):
    result = {}
    for name in VIEWS:
        value = np.asarray(data[name], dtype=np.float64)
        if value.ndim != 2 or value.shape[1] == 0 or not np.isfinite(value).all():
            raise ValueError('Each view must be a finite matrix with nonzero width')
        result[name] = value
    if len({len(x) for x in result.values()}) != 1:
        raise ValueError('All views must have the same number of rows')
    return result


def _transform(data, stats):
    arrays = _views(data)
    parts = []
    for name in VIEWS:
        x = arrays[name]
        if x.shape[1] != len(stats[name]['mean']):
            raise ValueError('Query and fitted feature widths differ')
        parts.append(np.tanh((x-stats[name]['mean'])/stats[name]['sd']) / np.sqrt(3*x.shape[1]))
    return np.concatenate(parts, axis=1)


def _kernel(x, anchors, diagonal, bandwidth):
    weighted = x*diagonal
    distances = (weighted*x).sum(1)[:, None] + (anchors*anchors*diagonal).sum(1)[None, :]
    distances -= 2*weighted@anchors.T
    return np.exp(-np.maximum(distances, 0)/bandwidth)


def _fit_predictor(x, y, reference, diagonal):
    weights = 0.5/np.bincount(y, minlength=2)[y]
    bandwidth = max(float(2*np.sum(x.var(axis=0)*diagonal)), 1e-6)
    gram = _kernel(x, x, diagonal, bandwidth)
    system = gram + np.diag(0.1/weights)
    dual = cho_solve(cho_factor(system, lower=True), 2*y-1)
    reference_score = _kernel(reference, x, diagonal, bandwidth)@dual
    return dict(anchors=x.copy(), diagonal=diagonal.copy(), bandwidth=bandwidth, dual=dual,
                center=float(reference_score.mean()),
                scale=max(float(np.sqrt(reference_score.var()+1e-6)), 0.05))


def _gradients(model, x):
    weighted = _kernel(x, model['anchors'], model['diagonal'], model['bandwidth'])*model['dual'][None, :]
    return (2/model['bandwidth'])*(weighted@model['anchors']-weighted.sum(1)[:, None]*x)*model['diagonal']


def _diagonal(importance, strength):
    total = float(importance.sum())
    if total <= 1e-20:
        return np.ones(len(importance))
    return (1-strength)+strength*len(importance)*importance/total


def fit_sensitivity_contrast(train, labels, reference, *, strength=0.9):
    """Learn a diagonal metric and its matched isotropic reference classifier.

    Inputs are low/high/raw feature dictionaries and binary training labels.
    Reference rows supply unlabeled normalization and score moments only.
    The metric uses balanced squared input gradients of an isotropic kernel
    classifier. ``strength`` is fixed; it is not selected from query labels.
    """
    train = _views(train)
    reference = _views(reference)
    n = len(train['low'])
    y = np.asarray(labels)
    if y.shape != (n,) or not np.array_equal(np.unique(y), [0, 1]):
        raise ValueError('Training labels must contain both binary classes and match rows')
    if not len(reference['low']):
        raise ValueError('Reference rows must be nonempty')
    if not np.isfinite(strength) or not 0 <= strength < 1:
        raise ValueError('Strength must lie in [0, 1)')
    stats = {}
    for name in VIEWS:
        if train[name].shape[1] != reference[name].shape[1]:
            raise ValueError('Training and reference feature widths differ')
        stats[name] = dict(mean=reference[name].mean(0), sd=np.sqrt(reference[name].var(0)+1e-6))
    x, ref = _transform(train, stats), _transform(reference, stats)
    y = y.astype(np.int64)
    base = _fit_predictor(x, y, ref, np.ones(x.shape[1]))
    gradients = _gradients(base, x)
    weights = 0.5/np.bincount(y, minlength=2)[y]
    importance = (weights[:, None]*gradients**2).sum(axis=0)
    diagonal = _diagonal(importance, strength)
    learned = _fit_predictor(x, y, ref, diagonal)
    return dict(format='apf-sensitivity-contrast-v1', feature_stats=stats,
                identity=base, learned=learned, strength=float(strength), importance=importance)


def predict_sensitivity_contrast(model, query):
    """Return learned-minus-isotropic standardized scores, in query order.

    Add ``0.1 * APF_scale * result`` to a frozen APF or full-method margin.
    Fit and prediction do not depend on either retained evidence module.
    """
    if model.get('format') != 'apf-sensitivity-contrast-v1':
        raise ValueError('Unknown model format')
    x = _transform(query, model['feature_stats'])
    values = []
    for key in ('identity', 'learned'):
        branch = model[key]
        score = _kernel(x, branch['anchors'], branch['diagonal'], branch['bandwidth'])@branch['dual']
        values.append((score-branch['center'])/branch['scale'])
    result = values[1]-values[0]
    if not np.isfinite(result).all():
        raise FloatingPointError('Nonfinite metric contrast')
    return result
