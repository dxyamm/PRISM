"""Run the complete scoring pipeline on generated data."""

from pathlib import Path
import sys
import tempfile

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_prism import run


def main() -> None:
    generator = torch.Generator().manual_seed(17)
    num_nodes, embed_dim, raw_dim = 240, 8, 6
    labels = torch.randint(0, 2, (num_nodes,), generator=generator)
    latent = torch.randn(num_nodes, embed_dim, generator=generator, dtype=torch.float64)
    low = latent + 0.25 * torch.randn(num_nodes, embed_dim, generator=generator, dtype=torch.float64)
    high = 0.7 * latent + 0.3 * torch.randn(num_nodes, embed_dim, generator=generator, dtype=torch.float64)
    raw = torch.randn(num_nodes, raw_dim, generator=generator, dtype=torch.float64)
    gate = torch.sigmoid(torch.randn(num_nodes, embed_dim, generator=generator, dtype=torch.float64))
    logits = torch.zeros(num_nodes, 2, dtype=torch.float64)
    logits[:, 1] = labels.double() * 0.4 + torch.randn(num_nodes, generator=generator, dtype=torch.float64)
    logits[:, 0] = -logits[:, 1]

    edges = torch.randint(0, num_nodes, (2, 1200), generator=generator)
    edges = torch.cat((edges, edges.flip(0), torch.arange(num_nodes).repeat(2, 1)), dim=1)
    train_ids = torch.arange(0, 80)
    reference_ids = torch.arange(80, 150)
    query_ids = torch.arange(150, num_nodes)
    take = lambda ids: {"low": low[ids], "high": high[ids], "raw": raw[ids],
                        "teacher_gate": gate[ids], "teacher_logits": logits[ids]}

    bundle = {
        "features": take(torch.arange(num_nodes)),
        "embeddings": {"low": low, "high": high},
        "edge_index": edges,
        "train_ids": train_ids,
        "train_labels": labels[train_ids],
        "reference_ids": reference_ids,
        "query_ids": query_ids,
        "calibration_center": float(logits[reference_ids, 1].mean()),
        "calibration_scale": float(logits[reference_ids, 1].std(unbiased=False).clamp_min(0.05)),
    }

    with tempfile.TemporaryDirectory() as folder:
        source = Path(folder) / "toy_bundle.pt"
        output = Path(folder) / "toy_scores.pt"
        torch.save(bundle, source)
        run(source, output)
        saved = torch.load(output, map_location="cpu", weights_only=False)

    assert saved["scores"]["111"].shape == (len(query_ids),)
    assert all(torch.isfinite(score).all() for score in saved["scores"].values())
    print("Toy smoke test passed. These scores are not experimental results.")


if __name__ == "__main__":
    main()
