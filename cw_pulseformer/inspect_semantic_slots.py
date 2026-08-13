"""Print reproducible slot-boundary errors for a trained semantic checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .semantic_data import SLOT_TO_ID, SyntheticSemanticDataset, collate_semantic
from .semantic_model import QsoSemanticFormer
from .train_semantic import model_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--slot", default="POWER", choices=tuple(SLOT_TO_ID))
    parser.add_argument("--samples", type=int, default=15_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260805 + 90_000_000)
    parser.add_argument("--limit", type=int, default=40)
    return parser.parse_args()


def selected_text(text: str, labels: torch.Tensor, label_id: int) -> str:
    return "".join(
        character
        for index, character in enumerate(text)
        if int(labels[index + 1]) == label_id
    )


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = QsoSemanticFormer(checkpoint["config"])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    dataset = SyntheticSemanticDataset(args.samples, seed=args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_semantic)
    label_id = SLOT_TO_ID[args.slot]
    errors: list[dict[str, object]] = []
    eligible = exact = 0
    for batch in loader:
        outputs = model(*model_inputs(batch))
        prediction = outputs[2].argmax(dim=-1)
        truth = batch["slot_labels"]
        valid = batch["valid_mask"]
        for row, text in enumerate(batch["texts"]):
            truth_positions = truth[row] == label_id
            if not bool(truth_positions.any()):
                continue
            eligible += 1
            predicted_positions = (prediction[row] == label_id) & valid[row]
            if bool(((truth_positions == predicted_positions) | ~valid[row]).all()):
                exact += 1
                continue
            if len(errors) < args.limit:
                errors.append({
                    "text": text,
                    "truth": selected_text(text, truth[row], label_id),
                    "prediction": selected_text(text, prediction[row], label_id),
                    "pendingQuestionId": int(batch["context_ids"][row, 1]),
                    "safeToCommit": bool(batch["safe_to_commit"][row]),
                })
    print(json.dumps({
        "slot": args.slot,
        "checkpointEpoch": checkpoint.get("epoch"),
        "samples": args.samples,
        "eligible": eligible,
        "exact": exact,
        "accuracy": exact / max(1, eligible),
        "errors": errors,
    }, indent=2))


if __name__ == "__main__":
    main()
