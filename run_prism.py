"""Fit and score PRISM from one frozen APF feature bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from prism import fit, fit_sensitivity_contrast, predict, predict_sensitivity_contrast

CONFIGURATIONS = ("000", "100", "010", "001", "110", "101", "011", "111")
FEATURE_KEYS = ("low", "high", "raw", "teacher_gate", "teacher_logits")


def _node_ids(value, size: int, name: str) -> torch.Tensor:
    raw = torch.as_tensor(value).detach().cpu()
    if raw.ndim != 1 or raw.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError(f"{name} must be a one-dimensional integer array")
    ids = raw.long()
    if not len(ids) or ids.min() < 0 or ids.max() >= size:
        raise ValueError(f"{name} must contain in-range node IDs")
    if len(torch.unique(ids)) != len(ids):
        raise ValueError(f"{name} must not contain duplicate IDs")
    return ids


def _take_rows(features: dict, ids: torch.Tensor) -> dict:
    return {key: torch.as_tensor(features[key])[ids] for key in FEATURE_KEYS}


def run(bundle_path: Path, output_path: Path) -> None:
    if output_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}")

    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    required = {
        "features", "embeddings", "edge_index", "train_ids", "train_labels",
        "reference_ids", "query_ids", "calibration_center", "calibration_scale",
    }
    missing = required.difference(bundle)
    if missing:
        raise ValueError(f"Feature bundle is missing keys: {sorted(missing)}")

    features = bundle["features"]
    row_counts = {len(torch.as_tensor(features[key])) for key in FEATURE_KEYS}
    if len(row_counts) != 1:
        raise ValueError("All feature fields must use the same global node order")
    num_nodes = row_counts.pop()
    train_ids = _node_ids(bundle["train_ids"], num_nodes, "train_ids")
    reference_ids = _node_ids(bundle["reference_ids"], num_nodes, "reference_ids")
    query_ids = _node_ids(bundle["query_ids"], num_nodes, "query_ids")
    groups = (train_ids, reference_ids, query_ids)
    if any(torch.isin(left, right).any() for index, left in enumerate(groups)
           for right in groups[index + 1:]):
        raise ValueError("Training, reference, and query IDs must be disjoint")
    train = _take_rows(features, train_ids)
    reference = _take_rows(features, reference_ids)
    query = _take_rows(features, query_ids)

    model = fit(
        train=train,
        labels=bundle["train_labels"],
        reference=reference,
        edge_index=bundle["edge_index"],
        embeddings=bundle["embeddings"],
        train_ids=train_ids,
        reference_ids=reference_ids,
        calibration_center=bundle["calibration_center"],
        calibration_scale=bundle["calibration_scale"],
    )
    core_scores = predict(
        model, query, bundle["edge_index"], bundle["embeddings"], query_ids
    )

    train_views = {key: train[key].numpy() for key in ("low", "high", "raw")}
    reference_views = {
        key: reference[key].numpy() for key in ("low", "high", "raw")
    }
    query_views = {key: query[key].numpy() for key in ("low", "high", "raw")}
    tsmc_model = fit_sensitivity_contrast(
        train_views,
        torch.as_tensor(bundle["train_labels"]).cpu().numpy().astype(np.int64, copy=False),
        reference_views,
    )
    tsmc = torch.from_numpy(predict_sensitivity_contrast(tsmc_model, query_views))

    apf = core_scores["apf"]
    nrd = core_scores["nrd"] - apf
    nhp = core_scores["nrd_nhp"] - core_scores["nrd"]
    corrections = {
        "1": nrd,
        "2": nhp,
        "3": 0.1 * float(bundle["calibration_scale"]) * tsmc,
    }
    scores = {"apf": apf}
    for configuration in CONFIGURATIONS:
        score = apf.clone()
        for index, flag in enumerate(configuration, start=1):
            if flag == "1":
                score += corrections[str(index)]
        scores[configuration] = score

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "prism-scores-v1",
            "query_ids": query_ids,
            "configurations": CONFIGURATIONS,
            "scores": scores,
            "model": {"core": model, "tsmc": tsmc_model},
        },
        output_path,
    )
    print(f"Saved {len(query_ids)} query scores for APF and eight PRISM configurations.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path, help="trusted PyTorch APF feature bundle")
    parser.add_argument("output", type=Path, help="output path for fitted models and scores")
    args = parser.parse_args()
    run(args.bundle, args.output)


if __name__ == "__main__":
    main()
