import numpy as np
import torch

from run_prism import CONFIGURATIONS, run


def test_runner_writes_baseline_and_all_factorial_scores(tmp_path):
    generator = torch.Generator().manual_seed(7)
    num_nodes, width = 100, 5
    labels = torch.arange(num_nodes) % 2
    low = torch.randn(num_nodes, width, generator=generator, dtype=torch.float64)
    high = torch.randn(num_nodes, width, generator=generator, dtype=torch.float64)
    raw = torch.randn(num_nodes, 4, generator=generator, dtype=torch.float64)
    gate = torch.sigmoid(torch.randn(num_nodes, width, generator=generator, dtype=torch.float64))
    logits = torch.randn(num_nodes, 2, generator=generator, dtype=torch.float64)
    train_ids = torch.arange(0, 40)
    reference_ids = torch.arange(40, 70)
    query_ids = torch.arange(70, 100)
    features = {"low": low, "high": high, "raw": raw,
                "teacher_gate": gate, "teacher_logits": logits}
    pairs = torch.randint(0, num_nodes, (2, 400), generator=generator)
    bundle = {
        "features": features,
        "embeddings": {"low": low, "high": high},
        "edge_index": pairs,
        "train_ids": train_ids,
        "train_labels": labels[train_ids],
        "reference_ids": reference_ids,
        "query_ids": query_ids,
        "calibration_center": float(logits[reference_ids, 1].mean()),
        "calibration_scale": float(logits[reference_ids, 1].std(unbiased=False).clamp_min(0.05)),
    }
    source, output = tmp_path / "bundle.pt", tmp_path / "scores.pt"
    torch.save(bundle, source)

    run(source, output)
    saved = torch.load(output, map_location="cpu", weights_only=False)
    scores = saved["scores"]

    assert saved["configurations"] == CONFIGURATIONS
    assert set(scores) == {"apf", *CONFIGURATIONS}
    assert all(value.shape == (len(query_ids),) for value in scores.values())
    assert all(torch.isfinite(value).all() for value in scores.values())
    np.testing.assert_allclose(scores["000"], scores["apf"])
    expected_gain = sum(scores[config] - scores["000"] for config in ("100", "010", "001"))
    np.testing.assert_allclose(scores["111"] - scores["000"], expected_gain)
