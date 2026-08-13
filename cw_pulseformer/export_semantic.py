"""Export contextual QsoSemanticFormer to ONNX and validate CPU inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
from typing import Any

import numpy as np
import torch

from .semantic_data import (
    ACT_NAMES,
    CHAR_TO_ID,
    CLS_ID,
    MAX_CALLSIGN_LENGTH,
    MAX_SEMANTIC_LENGTH,
    PAD_ID,
    PENDING_QUESTION_NAMES,
    PENDING_QUESTION_TO_ID,
    PHASE_NAMES,
    PHASE_TO_ID,
    PROCEDURE_GRADES,
    REGISTER_NAMES,
    SEMANTIC_SCHEMA_VERSION,
    SLOT_NAMES,
    TOPIC_NAMES,
    TOPIC_TO_ID,
    SyntheticSemanticDataset,
    collate_semantic,
)
from .semantic_model import (
    SEMANTIC_INPUT_NAMES,
    SEMANTIC_MODEL_VERSION,
    SEMANTIC_OUTPUT_NAMES,
    QsoSemanticFormer,
    semantic_parameter_count,
)


RUNTIME_CONTRACT_VERSION = "qso-semantic-runtime-4"
OPSET_VERSION = 17
OUTPUT_NAMES = SEMANTIC_OUTPUT_NAMES


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_dummy_inputs(batch_size: int = 1) -> tuple[torch.Tensor, ...]:
    input_ids = torch.zeros((batch_size, MAX_SEMANTIC_LENGTH), dtype=torch.long)
    valid_mask = torch.zeros((batch_size, MAX_SEMANTIC_LENGTH), dtype=torch.bool)
    input_ids[:, 0] = CLS_ID
    valid_mask[:, 0] = True
    return (
        input_ids,
        valid_mask,
        torch.zeros((batch_size, 2), dtype=torch.long),
        torch.zeros((batch_size, MAX_CALLSIGN_LENGTH), dtype=torch.long),
        torch.zeros((batch_size, MAX_CALLSIGN_LENGTH), dtype=torch.long),
        torch.zeros((batch_size, len(TOPIC_NAMES)), dtype=torch.float32),
    )


def model_inputs_from_batch(batch: dict[str, Any]) -> tuple[torch.Tensor, ...]:
    return tuple(batch[name] for name in SEMANTIC_INPUT_NAMES)


def feeds_from_batch(batch: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        name: batch[name].detach().cpu().numpy()
        for name in SEMANTIC_INPUT_NAMES
    }


def export_semantic_model(model: QsoSemanticFormer, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    dynamic_axes = {name: {0: "batch"} for name in (*SEMANTIC_INPUT_NAMES, *SEMANTIC_OUTPUT_NAMES)}
    torch.onnx.export(
        model,
        make_dummy_inputs(),
        output,
        input_names=SEMANTIC_INPUT_NAMES,
        output_names=SEMANTIC_OUTPUT_NAMES,
        dynamic_axes=dynamic_axes,
        opset_version=OPSET_VERSION,
        do_constant_folding=True,
        dynamo=False,
    )


def _new_parity_accumulator() -> dict[str, dict[str, float | int]]:
    return {
        name: {"maximum": 0.0, "absoluteSum": 0.0, "values": 0}
        for name in SEMANTIC_OUTPUT_NAMES
    }


def _update_parity(
    accumulator: dict[str, dict[str, float | int]],
    reference_outputs: tuple[np.ndarray, ...],
    exported_outputs: list[np.ndarray],
) -> None:
    for name, reference, exported in zip(
        SEMANTIC_OUTPUT_NAMES, reference_outputs, exported_outputs, strict=True,
    ):
        if reference.shape != exported.shape:
            raise RuntimeError(
                f"ONNX output {name} shape {exported.shape} does not match PyTorch {reference.shape}"
            )
        difference = np.abs(reference.astype(np.float64) - exported.astype(np.float64))
        stats = accumulator[name]
        stats["maximum"] = max(float(stats["maximum"]), float(difference.max(initial=0.0)))
        stats["absoluteSum"] = float(stats["absoluteSum"]) + float(difference.sum())
        stats["values"] = int(stats["values"]) + int(difference.size)


def _finish_parity(
    accumulator: dict[str, dict[str, float | int]],
) -> dict[str, dict[str, float]]:
    return {
        name: {
            "maxAbsError": float(stats["maximum"]),
            "meanAbsError": float(stats["absoluteSum"]) / max(1, int(stats["values"])),
        }
        for name, stats in accumulator.items()
    }


def build_runtime_contract(config: dict[str, Any], reports: list[dict[str, Any]]) -> dict[str, Any]:
    schema = {
        "contractVersion": RUNTIME_CONTRACT_VERSION,
        "schemaVersion": SEMANTIC_SCHEMA_VERSION,
        "modelVersion": SEMANTIC_MODEL_VERSION,
        "executionMode": "post-transcription-offline-contextual",
        "maxSemanticLength": MAX_SEMANTIC_LENGTH,
        "maxCallsignLength": MAX_CALLSIGN_LENGTH,
        "inputs": {
            "input_ids": {"dtype": "int64", "shape": ["batch", MAX_SEMANTIC_LENGTH]},
            "valid_mask": {"dtype": "bool", "shape": ["batch", MAX_SEMANTIC_LENGTH]},
            "context_ids": {
                "dtype": "int64", "shape": ["batch", 2],
                "columns": ["phase", "pendingQuestion"],
            },
            "self_callsign_ids": {"dtype": "int64", "shape": ["batch", MAX_CALLSIGN_LENGTH]},
            "peer_callsign_ids": {"dtype": "int64", "shape": ["batch", MAX_CALLSIGN_LENGTH]},
            "known_slots": {"dtype": "float32", "shape": ["batch", len(TOPIC_NAMES)]},
        },
        "contextSchema": {
            "phaseNames": PHASE_NAMES,
            "phaseToId": PHASE_TO_ID,
            "pendingQuestionNames": PENDING_QUESTION_NAMES,
            "pendingQuestionToId": PENDING_QUESTION_TO_ID,
            "knownSlotNames": TOPIC_NAMES,
            "knownSlotToIndex": TOPIC_TO_ID,
            "bareAnswerRule": (
                "A value-only reply is eligible for slot commit only in EXCHANGE when "
                "pendingQuestion identifies that slot."
            ),
        },
        "tokenizer": {
            "padId": PAD_ID,
            "clsId": CLS_ID,
            "characterToId": CHAR_TO_ID,
            "normalization": "uppercase supported ASCII; remove unsupported characters; collapse spaces",
        },
        "outputs": {
            "act_logits": {"names": ACT_NAMES, "activation": "sigmoid"},
            "topic_logits": {"names": TOPIC_NAMES, "activation": "sigmoid"},
            "slot_logits": {"names": SLOT_NAMES, "activation": "argmax per character"},
            "register_logits": {"names": REGISTER_NAMES, "activation": "argmax"},
            "procedure_logits": {"names": PROCEDURE_GRADES, "activation": "argmax"},
            "safe_to_commit_logits": {
                "activation": "sigmoid", "threshold": 0.5,
                "meaning": "false forbids QSO state mutation; true remains subject to deterministic validation",
            },
        },
        "config": config,
        "opset": OPSET_VERSION,
    }
    canonical = json.dumps(
        schema, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return {
        **schema,
        "schemaSha256": hashlib.sha256(canonical).hexdigest(),
        "artifacts": [
            {
                "file": Path(report["path"]).name,
                "bytes": report["bytes"],
                "sha256": report["sha256"],
            }
            for report in reports
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quantized-output", type=Path)
    parser.add_argument("--benchmark-samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260805)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.benchmark_samples <= 0:
        raise SystemExit("--benchmark-samples must be positive")
    try:
        import onnx
        import onnxruntime as ort
    except ImportError as error:
        raise SystemExit("Install the export dependencies: onnx and onnxruntime") from error

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = QsoSemanticFormer(checkpoint["config"])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    export_semantic_model(model, args.output)
    onnx.checker.check_model(onnx.load(args.output))

    exported_paths = [args.output]
    if args.quantized_output:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        args.quantized_output.parent.mkdir(parents=True, exist_ok=True)
        quantize_dynamic(
            str(args.output),
            str(args.quantized_output),
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["MatMul", "Gemm"],
        )
        onnx.checker.check_model(onnx.load(args.quantized_output))
        exported_paths.append(args.quantized_output)

    dataset = SyntheticSemanticDataset(args.benchmark_samples, seed=args.seed + 90_000_000)
    batches = [collate_semantic([dataset[index]]) for index in range(args.benchmark_samples)]
    reference_outputs: list[tuple[np.ndarray, ...]] = []
    with torch.inference_mode():
        for batch in batches:
            reference_outputs.append(tuple(
                output.detach().cpu().numpy()
                for output in model(*model_inputs_from_batch(batch))
            ))

    reports = []
    for path in exported_paths:
        session_started = time.perf_counter()
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        session_creation_ms = (time.perf_counter() - session_started) * 1000.0
        first_feeds = feeds_from_batch(batches[0])
        cold_started = time.perf_counter()
        session.run(None, first_feeds)
        cold_run_ms = (time.perf_counter() - cold_started) * 1000.0
        for _ in range(3):
            session.run(None, first_feeds)

        latencies: list[float] = []
        parity = _new_parity_accumulator()
        for batch, reference in zip(batches, reference_outputs, strict=True):
            feeds = feeds_from_batch(batch)
            started = time.perf_counter()
            outputs = session.run(None, feeds)
            latencies.append((time.perf_counter() - started) * 1000.0)
            _update_parity(parity, reference, outputs)
        reports.append({
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "batchSize": 1,
            "samples": args.benchmark_samples,
            "outputParityWithPytorch": _finish_parity(parity),
            "latencyMs": {
                "sessionCreation": session_creation_ms,
                "coldRun": cold_run_ms,
                "coldStartTotal": session_creation_ms + cold_run_ms,
                "hotMedian": statistics.median(latencies),
                "hotP95": float(np.percentile(latencies, 95)),
            },
        })

    contract = build_runtime_contract(checkpoint["config"], reports)
    contract_path = args.output.with_suffix(".runtime-contract.json")
    contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    manifest = {
        "modelVersion": checkpoint.get("modelVersion", SEMANTIC_MODEL_VERSION),
        "checkpoint": str(args.checkpoint),
        "checkpointEpoch": checkpoint.get("epoch"),
        "checkpointMetrics": checkpoint.get("metrics"),
        "config": checkpoint["config"],
        "parameters": semantic_parameter_count(model),
        "opset": OPSET_VERSION,
        "inputs": SEMANTIC_INPUT_NAMES,
        "outputs": SEMANTIC_OUTPUT_NAMES,
        "exports": reports,
        "runtimeContract": str(contract_path),
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
