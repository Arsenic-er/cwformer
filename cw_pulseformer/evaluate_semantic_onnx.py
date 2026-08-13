"""Run the complete semantic release gates against an exported ONNX model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .semantic_data import SyntheticSemanticDataset, collate_semantic
from .semantic_model import SEMANTIC_INPUT_NAMES
from .train_semantic import evaluate


class OnnxSemanticModel(nn.Module):
    def __init__(self, path: Path) -> None:
        super().__init__()
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise SystemExit("Install onnxruntime before evaluating ONNX exports") from error
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    def forward(self, *inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        feeds = {
            name: value.detach().cpu().numpy()
            for name, value in zip(SEMANTIC_INPUT_NAMES, inputs, strict=True)
        }
        return tuple(torch.from_numpy(value) for value in self.session.run(None, feeds))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=15_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = OnnxSemanticModel(args.model)
    dataset = SyntheticSemanticDataset(args.samples, seed=args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        collate_fn=collate_semantic,
    )
    report = {
        "model": str(args.model),
        "bytes": args.model.stat().st_size,
        "device": "cpu",
        "seed": args.seed,
        "metrics": evaluate(model, loader, torch.device("cpu")),
    }
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
