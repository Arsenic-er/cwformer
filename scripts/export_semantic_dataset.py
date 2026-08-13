"""Export the deterministic QSO Semantics v4 splits as reproducible JSONL.GZ."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cw_pulseformer.semantic_data import SEMANTIC_SCHEMA_VERSION, SyntheticSemanticDataset


SPLITS = {
    "train": (100_000, 20_260_805),
    "validation": (15_000, 110_260_805),
    "confirm-90268741": (15_000, 90_268_741),
    "confirm-90268742": (15_000, 90_268_742),
    "confirm-90268743": (15_000, 90_268_743),
}


def record_for(split: str, index: int, dataset: SyntheticSemanticDataset) -> dict:
    spec = dataset[index]["spec"]
    return {
        "schemaVersion": SEMANTIC_SCHEMA_VERSION,
        "split": split,
        "index": index,
        "text": spec.text,
        "context": {
            "phase": spec.context.phase,
            "pendingQuestion": spec.context.pending_question,
            "selfCallsign": spec.context.self_callsign,
            "peerCallsign": spec.context.peer_callsign,
            "knownSlots": list(spec.context.known_slots),
        },
        "labels": {
            "acts": list(spec.acts),
            "topics": list(spec.topics),
            "procedure": spec.procedure,
            "spans": [
                {"start": start, "end": end, "slot": slot}
                for start, end, slot in spec.spans
            ],
            "safeToCommit": spec.safe_to_commit,
            "register": spec.register,
        },
    }


def export_split(output_dir: Path, split: str, samples: int, seed: int) -> dict:
    path = output_dir / f"{split}.jsonl.gz"
    dataset = SyntheticSemanticDataset(samples, seed=seed)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as handle:
                for index in range(samples):
                    json.dump(record_for(split, index, dataset), handle, ensure_ascii=True, separators=(",", ":"))
                    handle.write("\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"file": path.name, "samples": samples, "seed": seed, "bytes": path.stat().st_size, "sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Export all fixed train, validation, and confirmation splits")
    parser.add_argument("--split", choices=tuple(SPLITS), help="Export one fixed split")
    parser.add_argument("--output", type=Path, default=Path("datasets/qso-semantics-v4"))
    args = parser.parse_args()
    if not args.all and not args.split:
        parser.error("choose --all or --split")
    selected = tuple(SPLITS) if args.all else (args.split,)
    args.output.mkdir(parents=True, exist_ok=True)
    records = [export_split(args.output, split, *SPLITS[split]) for split in selected]
    manifest = {
        "schemaVersion": SEMANTIC_SCHEMA_VERSION,
        "generator": "cw_pulseformer.semantic_data.SyntheticSemanticDataset",
        "indexFormula": "seed + index * 1000003",
        "files": records,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
