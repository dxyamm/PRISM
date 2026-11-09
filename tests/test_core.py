import numpy as np
import torch

from prism import edge_index_to_csr, fit, predict
from prism.core import _dispersion_features


def make_problem():
    generator = torch.Generator().manual_seed(41)
    num_nodes = 140
    labels = torch.arange(num_nodes) % 2
    low = torch.randn(num_nodes, 6, generator=generator, dtype=torch.float64)
    high = 0.6 * low + torch.randn(num_nodes, 6, generator=generator, dtype=torch.float64)
    raw = torch.randn(num_nodes, 5, generator=generator, dtype=torch.float64)
    gate = torch.sigmoid(torch.randn(num_nodes, 6, generator=generator, dtype=torch.float64))
    logits = torch.randn(num_nodes, 2, generator=generator, dtype=torch.float64)
    pairs = torch.randint(0, num_nodes, (2, 600), generator=generator)
    edges = torch.cat((pairs, pairs.flip(0), torch.arange(num_nodes).repeat(2, 1)), dim=1)

    def select(ids):
        return {
            "low": low[ids],
            "high": high[ids],
            "raw": raw[ids],
            "teacher_gate": gate[ids],
            "teacher_logits": logits[ids],
        }

    train_ids = torch.arange(0, 60)
    reference_ids = torch.arange(60, 100)
    query_ids = torch.arange(100, num_nodes)
    margin = logits[reference_ids, 1] - logits[reference_ids, 0]
    model = fit(
        train=select(train_ids),
        labels=labels[train_ids],
        reference=select(reference_ids),
        edge_index=edges,
        embeddings={"low": low, "high": high},
        train_ids=train_ids,
        reference_ids=reference_ids,
        calibration_center=float(margin.mean()),
        calibration_scale=float(margin.std(unbiased=False).clamp_min(0.05)),
    )
    return model, select(query_ids), edges, {"low": low, "high": high}, query_ids


def test_adjacency_is_undirected_deduplicated_and_loop_free():
    edges = torch.tensor([[0, 0, 1, 2], [1, 1, 0, 2]])
    adjacency = edge_index_to_csr(edges, 3)
    assert (adjacency != adjacency.T).nnz == 0
    assert adjacency.diagonal().sum() == 0
    assert np.all(adjacency.data == 1)


def test_core_scores_are_finite_additive_and_order_equivariant():
    model, query, edges, embeddings, query_ids = make_problem()
    scores = predict(model, query, edges, embeddings, query_ids)
    reverse = torch.arange(len(query_ids) - 1, -1, -1)
    reordered = predict(
        model,
        {key: value[reverse] for key, value in query.items()},
        edges,
        embeddings,
        query_ids[reverse],
    )

    assert set(scores) == {"apf", "nrd", "nrd_nhp"}
    for name, score in scores.items():
        assert score.shape == (len(query_ids),)
        assert torch.isfinite(score).all()
        torch.testing.assert_close(reordered[name][torch.argsort(reverse)], score)

    assert not torch.equal(scores["nrd_nhp"], scores["nrd"])


def test_neighborhood_descriptor_is_batch_invariant():
    _, _, edges, embeddings, query_ids = make_problem()
    adjacency = edge_index_to_csr(edges, len(embeddings["low"]))
    whole = _dispersion_features(adjacency, embeddings, query_ids)
    chunks = torch.cat([
        _dispersion_features(adjacency, embeddings, query_ids[start:start + 11])
        for start in range(0, len(query_ids), 11)
    ])
    torch.testing.assert_close(whole, chunks, atol=1e-12, rtol=1e-12)
