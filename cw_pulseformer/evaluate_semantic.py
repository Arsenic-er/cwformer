"""Evaluate a semantic checkpoint on a disjoint deterministic validation set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .semantic_data import SyntheticSemanticDataset, collate_semantic
from .semantic_model import QsoSemanticFormer
from .train_semantic import evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--promoted-checkpoint", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples <= 0:
        raise SystemExit("--samples must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = QsoSemanticFormer(checkpoint["config"])
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    dataset = SyntheticSemanticDataset(args.samples, seed=args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        collate_fn=collate_semantic,
    )
    metrics = evaluate(model, loader, device)
    report = {
        "modelVersion": checkpoint.get("modelVersion"),
        "semanticSchemaVersion": checkpoint.get("semanticSchemaVersion"),
        "checkpoint": str(args.checkpoint),
        "checkpointEpoch": checkpoint.get("epoch"),
        "device": str(device),
        "seed": args.seed,
        "metrics": metrics,
    }
    if args.promoted_checkpoint:
        if not metrics["releaseReady"]:
            raise SystemExit("refusing to promote a checkpoint that failed release gates")
        args.promoted_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint["metrics"] = metrics
        checkpoint["promotion"] = {
            "evaluationSeed": args.seed,
            "evaluationSamples": args.samples,
            "evaluationReport": str(args.output) if args.output else None,
        }
        torch.save(checkpoint, args.promoted_checkpoint)
        report["promotedCheckpoint"] = str(args.promoted_checkpoint)
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
