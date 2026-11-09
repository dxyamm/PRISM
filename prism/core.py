"""Core relational and neighborhood scoring branches for frozen APF features."""
from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import torch
from scipy import sparse

FEATURE_KEYS = ("low", "high", "raw")
APF_KEYS = (*FEATURE_KEYS, "teacher_gate", "teacher_logits")
NRD_ALPHA = 0.25
NHP_ALPHA = 0.1
RIDGE = 0.1
GP_NOISE = 0.1


def _tensor(value: object) -> torch.Tensor:
    return torch.as_tensor(value).detach().cpu().double()


def _feature_dict(data: Mapping[str, object]) -> dict[str, torch.Tensor]:
    missing = set(APF_KEYS).difference(data)
    if missing:
        raise ValueError(f"Missing APF fields: {sorted(missing)}")
    result = {key: _tensor(data[key]) for key in APF_KEYS}
    lengths = {value.shape[0] for value in result.values()}
    if len(lengths) != 1 or any(value.ndim != 2 for value in result.values()):
        raise ValueError("Each APF field must be a 2-D [nodes, features] tensor with equal rows")
    if result["teacher_logits"].shape[1] != 2:
        raise ValueError("teacher_logits must have two class columns")
    if result["teacher_gate"].shape[1] != result["low"].shape[1]:
        raise ValueError("teacher_gate and low/high embeddings must have equal width")
    if not all(torch.isfinite(value).all() for value in result.values()):
        raise ValueError("APF fields contain non-finite values")
    return result


def _moments(x: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "mean": x.mean(0, keepdim=True),
        "sd": (x.var(0, unbiased=False, keepdim=True) + 1e-6).sqrt(),
    }


def _normalize(x: torch.Tensor, stats: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return torch.tanh((x - stats["mean"]) / stats["sd"]) / math.sqrt(x.shape[-1])


def _margin(data: Mapping[str, torch.Tensor]) -> torch.Tensor:
    logits = data["teacher_logits"]
    return logits[:, 1] - logits[:, 0]


def _rbf(x: torch.Tensor, anchors: torch.Tensor, bandwidth: torch.Tensor) -> torch.Tensor:
    d2 = (x.square().sum(1, keepdim=True) + anchors.square().sum(1)[None, :] - 2 * x @ anchors.T)
    return torch.exp(-d2.clamp_min(0) / bandwidth.clamp_min(1e-6))


def _width(x: torch.Tensor) -> torch.Tensor:
    return (2 * (x.square().sum(1).mean() - x.mean(0).square().sum())).clamp_min(1e-6)


def _fit_classifier(x: torch.Tensor, y: torch.Tensor) -> dict[str, torch.Tensor]:
    if not ((y == 0).any() and (y == 1).any()):
        raise ValueError("Training labels must contain both classes")
    counts = torch.stack(((y == 0).sum(), (y == 1).sum())).double()
    weights = torch.where(y == 1, 0.5 / counts[1], 0.5 / counts[0])
    root = weights.sqrt()
    target = 2 * y.double() - 1
    bandwidth = _width(x)
    gram = _rbf(x, x, bandwidth)
    lhs = gram * root[:, None] * root[None, :] + RIDGE * torch.eye(len(y), dtype=x.dtype)
    dual = torch.linalg.solve(lhs, root * target) * root
    return {"anchors": x.clone(), "bandwidth": bandwidth, "dual": dual}


def _score_classifier(model: Mapping[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    return _rbf(x, model["anchors"], model["bandwidth"]) @ model["dual"]


def _basis(data: Mapping[str, torch.Tensor], stats: Mapping[str, Mapping[str, torch.Tensor]]) -> torch.Tensor:
    fused = data["teacher_gate"] * data["low"] + (1 - data["teacher_gate"]) * data["high"]
    z_fused = _normalize(fused, stats["fused"])
    z_margin = _normalize(_margin(data)[:, None], stats["margin"])
    return torch.cat((z_fused, z_margin, torch.ones_like(z_margin)), dim=1)


def _fit_teacher(train: Mapping[str, torch.Tensor], y: torch.Tensor,
                 reference: Mapping[str, torch.Tensor]) -> dict[str, object]:
    n = len(y)
    fused_ref = reference["teacher_gate"] * reference["low"] + (1 - reference["teacher_gate"]) * reference["high"]
    basis_stats = {"fused": _moments(fused_ref), "margin": _moments(_margin(reference)[:, None])}
    joined = {key: torch.cat((train[key], reference[key])) for key in APF_KEYS}
    basis = _basis(joined, basis_stats)
    b_ref = basis[n:]
    bdim = b_ref.shape[1]
    bgram = b_ref.T @ b_ref / len(reference["raw"])
    branches = []
    for key in FEATURE_KEYS:
        stats = _moments(reference[key])
        x = _normalize(joined[key], stats)
        ref_x = x[n:]
        lhs = bgram + RIDGE * torch.eye(bdim, dtype=x.dtype)
        projection = torch.linalg.solve(lhs, b_ref.T @ ref_x / len(ref_x))
        residual = x - basis @ projection
        model = _fit_classifier(residual[:n], y)
        ref_score = _score_classifier(model, residual[n:])
        calibration = _moments(ref_score[:, None])
        branches.append({
            "key": key,
            "stats": stats,
            "projection": projection,
            "classifier": model,
            "score_mean": calibration["mean"][0, 0],
            "score_sd": calibration["sd"][0, 0].clamp_min(0.05),
        })
    return {"basis_stats": basis_stats, "branches": branches}


def _predict_teacher(model: Mapping[str, object], data: Mapping[str, torch.Tensor]) -> torch.Tensor:
    basis = _basis(data, model["basis_stats"])
    scores = []
    for branch in model["branches"]:
        x = _normalize(data[branch["key"]], branch["stats"])
        residual = x - basis @ branch["projection"]
        raw_score = _score_classifier(branch["classifier"], residual)
        scores.append((raw_score - branch["score_mean"]) / branch["score_sd"])
    return torch.stack(scores, dim=1).mean(1)


def _conditioning(xs: list[torch.Tensor], target: int) -> torch.Tensor:
    return torch.cat([value for index, value in enumerate(xs) if index != target], dim=1) / math.sqrt(2)


def _contrast(x: torch.Tensor, anchors: torch.Tensor, y: torch.Tensor, bandwidth: torch.Tensor) -> torch.Tensor:
    distances = (x.square().sum(1, keepdim=True) + anchors.square().sum(1)[None, :] - 2 * x @ anchors.T).clamp_min(0)
    normal = y == 0
    anomaly = y == 1
    normal_logsim = -distances[:, normal] / bandwidth
    anomaly_logsim = -torch.log1p(distances[:, anomaly] / bandwidth)
    normal_score = torch.logsumexp(normal_logsim, dim=1) - math.log(int(normal.sum()))
    anomaly_score = torch.logsumexp(anomaly_logsim, dim=1) - math.log(int(anomaly.sum()))
    return anomaly_score - normal_score


def _fit_relation(train: Mapping[str, torch.Tensor], y: torch.Tensor,
                  reference: Mapping[str, torch.Tensor]) -> dict[str, object]:
    n = len(y)
    norms = {key: _moments(reference[key]) for key in FEATURE_KEYS}
    xs = [_normalize(torch.cat((train[key], reference[key])), norms[key]) for key in FEATURE_KEYS]
    normal_ids = (y == 0).nonzero().flatten()
    branches = []
    for index, x in enumerate(xs):
        q = _conditioning(xs, index)
        q_normal = q[normal_ids]
        x_normal = x[normal_ids]
        bandwidth = _width(q_normal)
        gram = _rbf(q_normal, q_normal, bandwidth)
        inverse = torch.linalg.inv(gram + GP_NOISE * torch.eye(len(normal_ids), dtype=x.dtype))
        dual = inverse @ x_normal
        residual = x - _rbf(q, q_normal, bandwidth) @ dual
        residual[normal_ids] = dual / inverse.diagonal()[:, None]

        residual_train = residual[:n]
        residual_ref = residual[n:]
        baseline = _fit_classifier(residual_train, y)
        baseline_ref = _score_classifier(baseline, residual_ref)
        baseline_cal = _moments(baseline_ref[:, None])
        support_bandwidth = _width(residual_train)
        asym_ref = _contrast(residual_ref, residual_train, y, support_bandwidth)
        asym_cal = _moments(asym_ref[:, None])
        branches.append({
            "stats": norms[FEATURE_KEYS[index]],
            "conditioning_index": index,
            "anchors": q_normal,
            "gp_bandwidth": bandwidth,
            "gp_dual": dual,
            "baseline": baseline,
            "baseline_mean": baseline_cal["mean"][0, 0],
            "baseline_sd": baseline_cal["sd"][0, 0].clamp_min(0.05),
            "support_anchors": residual_train,
            "support_labels": y.clone(),
            "support_bandwidth": support_bandwidth,
            "asym_mean": asym_cal["mean"][0, 0],
            "asym_sd": asym_cal["sd"][0, 0].clamp_min(0.05),
        })
    return {"norms": norms, "branches": branches}


def _predict_relation(model: Mapping[str, object], data: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    xs = [_normalize(data[key], model["norms"][key]) for key in FEATURE_KEYS]
    baseline_scores = []
    asym_scores = []
    for branch, x in zip(model["branches"], xs):
        q = _conditioning(xs, branch["conditioning_index"])
        residual = x - _rbf(q, branch["anchors"], branch["gp_bandwidth"]) @ branch["gp_dual"]
        baseline = _score_classifier(branch["baseline"], residual)
        baseline_scores.append((baseline - branch["baseline_mean"]) / branch["baseline_sd"])
        asym = _contrast(residual, branch["support_anchors"], branch["support_labels"], branch["support_bandwidth"])
        asym_scores.append((asym - branch["asym_mean"]) / branch["asym_sd"])
    return torch.stack(baseline_scores, dim=1).mean(1), torch.stack(asym_scores, dim=1).mean(1)


def edge_index_to_csr(edge_index: object, num_nodes: int) -> sparse.csr_matrix:
    """Build undirected CSR from edges; sparse input must already be canonical."""
    if sparse.issparse(edge_index):
        adjacency = edge_index.tocsr()
        if adjacency.shape != (num_nodes, num_nodes):
            raise ValueError("adjacency shape must match num_nodes")
        return adjacency
    edges = np.asarray(torch.as_tensor(edge_index).cpu(), dtype=np.int64)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, edges]")
    if edges.size and (edges.min() < 0 or edges.max() >= num_nodes):
        raise ValueError("edge_index contains an out-of-range node ID")
    src, dst = edges
    keep = src != dst
    rows = np.concatenate((src[keep], dst[keep]))
    cols = np.concatenate((dst[keep], src[keep]))
    adjacency = sparse.csr_matrix((np.ones(len(rows), dtype=np.uint8), (rows, cols)),
                                  shape=(num_nodes, num_nodes))
    adjacency.data[:] = 1
    return adjacency


def _dispersion_features(adjacency: sparse.csr_matrix, embeddings: Mapping[str, object],
                         node_ids: object) -> torch.Tensor:
    ids = np.asarray(torch.as_tensor(node_ids).cpu(), dtype=np.int64)
    if ids.ndim != 1 or (ids.size and (ids.min() < 0 or ids.max() >= adjacency.shape[0])):
        raise ValueError("node_ids must be a valid one-dimensional node index")
    low = _tensor(embeddings["low"])
    high = _tensor(embeddings["high"])
    if low.shape[0] != adjacency.shape[0] or high.shape != low.shape:
        raise ValueError("Full low/high embeddings must have shape [num_nodes, embedding_dim]")
    selected = adjacency[ids]
    row, col = selected.nonzero()
    row_t = torch.from_numpy(row.astype(np.int64, copy=False))
    counts = torch.from_numpy(np.diff(selected.indptr)).double().clamp_min(1)[:, None]
    outputs = []
    for emb in (low, high):
        total = torch.zeros((len(ids), emb.shape[1]), dtype=torch.float64)
        squares = torch.zeros_like(total)
        if len(row):
            values = emb[torch.from_numpy(col.astype(np.int64, copy=False))]
            total.index_add_(0, row_t, values)
            squares.index_add_(0, row_t, values.square())
        variance = (squares / counts - (total / counts).square()).clamp_min(0)
        sigma = variance.sqrt()
        outputs.extend((torch.zeros_like(sigma), sigma))
    return torch.cat(outputs, dim=1)


def _fit_dispersion(adjacency: sparse.csr_matrix, embeddings: Mapping[str, object],
                    train_ids: object, y: torch.Tensor, reference_ids: object) -> dict[str, object]:
    train_x = _dispersion_features(adjacency, embeddings, train_ids)
    ref_x = _dispersion_features(adjacency, embeddings, reference_ids)
    stats = _moments(ref_x)
    classifier = _fit_classifier(_normalize(train_x, stats), y)
    ref_score = _score_classifier(classifier, _normalize(ref_x, stats))
    score_stats = _moments(ref_score[:, None])
    return {
        "stats": stats,
        "classifier": classifier,
        "score_mean": score_stats["mean"][0, 0],
        "score_sd": score_stats["sd"][0, 0].clamp_min(0.05),
    }


def _predict_dispersion(model: Mapping[str, object], adjacency: sparse.csr_matrix,
                        embeddings: Mapping[str, object], query_ids: object) -> torch.Tensor:
    x = _dispersion_features(adjacency, embeddings, query_ids)
    score = _score_classifier(model["classifier"], _normalize(x, model["stats"]))
    return (score - model["score_mean"]) / model["score_sd"]


@torch.no_grad()
def fit(train: Mapping[str, object], labels: object, reference: Mapping[str, object],
        edge_index: object, embeddings: Mapping[str, object], train_ids: object,
        reference_ids: object, calibration_center: float, calibration_scale: float) -> dict[str, object]:
    """Fit the three post-processing branches from labeled train and unlabeled reference nodes.

    ``train`` and ``reference`` each contain low/high/raw/teacher_gate/teacher_logits.
    The full low/high embeddings and graph edges are needed for neighborhood dispersion.
    Calibration center/scale are supplied by the APF training pipeline.
    """
    train_data = _feature_dict(train)
    reference_data = _feature_dict(reference)
    y = torch.as_tensor(labels).detach().cpu().long().flatten()
    if len(y) != len(train_data["raw"]) or not torch.all((y == 0) | (y == 1)):
        raise ValueError("labels must be binary and match train rows")
    if len(y) < 2 or len(reference_data["raw"]) == 0:
        raise ValueError("At least two labeled rows and one unlabeled reference row are required")
    if not ((y == 0).any() and (y == 1).any()):
        raise ValueError("Training labels must contain both classes")
    if not math.isfinite(calibration_center) or not math.isfinite(calibration_scale) or calibration_scale <= 0:
        raise ValueError("APF calibration scale must be finite and positive")
    adjacency = edge_index_to_csr(edge_index, len(_tensor(embeddings["low"])))
    train_ids_t = torch.as_tensor(train_ids).long().flatten()
    reference_ids_t = torch.as_tensor(reference_ids).long().flatten()
    if len(train_ids_t) != len(y) or len(reference_ids_t) != len(reference_data["raw"]):
        raise ValueError("node IDs must match the train/reference feature row counts")
    if len(torch.unique(train_ids_t)) != len(train_ids_t) or len(torch.unique(reference_ids_t)) != len(reference_ids_t):
        raise ValueError("train_ids and reference_ids must not contain duplicates")
    if np.intersect1d(train_ids_t.numpy(), reference_ids_t.numpy()).size:
        raise ValueError("training and reference IDs must be disjoint")
    teacher = _fit_teacher(train_data, y, reference_data)
    relation = _fit_relation(train_data, y, reference_data)
    dispersion = _fit_dispersion(adjacency, embeddings, train_ids_t, y, reference_ids_t)
    return {
        "format": "prism-core-v1",
        "teacher": teacher,
        "relation": relation,
        "dispersion": dispersion,
        "calibration_center": float(calibration_center),
        "calibration_scale": float(calibration_scale),
        "train_ids": train_ids_t,
        "reference_ids": reference_ids_t,
    }


@torch.no_grad()
def predict(model: Mapping[str, object], query: Mapping[str, object], edge_index: object,
            embeddings: Mapping[str, object], query_ids: object) -> dict[str, torch.Tensor]:
    """Return APF, NRD, and NRD+NHP margins."""
    if model.get("format") != "prism-core-v1":
        raise ValueError("Unsupported model bundle")
    data = _feature_dict(query)
    ids = torch.as_tensor(query_ids).long().flatten()
    if len(ids) != len(data["raw"]):
        raise ValueError("query_ids must match query feature rows")
    raw = _margin(data)
    teacher_score = _predict_teacher(model["teacher"], data)
    normal_score, asym_score = _predict_relation(model["relation"], data)
    adjacency = edge_index_to_csr(edge_index, len(_tensor(embeddings["low"])))
    dispersion_score = _predict_dispersion(model["dispersion"], adjacency, embeddings, ids)
    center = model["calibration_center"]
    scale = model["calibration_scale"]
    standardized = (raw - center) / scale
    nrd = raw + NRD_ALPHA * scale * (
        teacher_score / 2 + normal_score / 4 + asym_score / 4 - standardized
    )
    nrd_nhp = nrd + NHP_ALPHA * scale * dispersion_score
    scores = {"apf": raw, "nrd": nrd, "nrd_nhp": nrd_nhp}
    if not all(torch.isfinite(score).all() for score in scores.values()):
        raise FloatingPointError("Non-finite score generated")
    return scores
