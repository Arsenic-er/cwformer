"""Train the local context-aware QSO text-understanding network."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .semantic_data import (
    ACT_NAMES, ACT_TO_ID, PROCEDURE_GRADES, PROCEDURE_TO_ID, REGISTER_NAMES,
    REGISTER_TO_ID, SEMANTIC_SCHEMA_VERSION, SLOT_NAMES, SLOT_TO_ID,
    TOPIC_NAMES, TOPIC_TO_ID,
    SyntheticSemanticDataset, collate_semantic,
)
from .semantic_model import QsoSemanticFormer, SEMANTIC_MODEL_VERSION, SemanticModelConfig, semantic_parameter_count
from .semantic_slot_decoder import (
    SLOT_PATTERNS, decode_slot_values, labeled_slot_values,
)


def micro_f1(truth: torch.Tensor, prediction: torch.Tensor) -> float:
    truth = truth.bool()
    prediction = prediction.bool()
    tp = int((truth & prediction).sum())
    fp = int((~truth & prediction).sum())
    fn = int((truth & ~prediction).sum())
    return 2 * tp / max(1, 2 * tp + fp + fn)


def macro_f1(truth: torch.Tensor, prediction: torch.Tensor) -> float:
    """Average the binary F1 of every label so rare QSO intents stay visible."""
    truth = truth.bool()
    prediction = prediction.bool()
    scores: list[float] = []
    for label in range(truth.shape[-1]):
        label_truth = truth[..., label]
        label_prediction = prediction[..., label]
        tp = int((label_truth & label_prediction).sum())
        fp = int((~label_truth & label_prediction).sum())
        fn = int((label_truth & ~label_prediction).sum())
        scores.append(2 * tp / max(1, 2 * tp + fp + fn))
    return sum(scores) / max(1, len(scores))


def f1_by_label(
    truth: torch.Tensor,
    prediction: torch.Tensor,
    names: tuple[str, ...],
) -> dict[str, float]:
    truth = truth.bool()
    prediction = prediction.bool()
    return {
        name: micro_f1(truth[..., index], prediction[..., index])
        for index, name in enumerate(names)
    }


def multiclass_f1_by_label(
    truth: torch.Tensor,
    prediction: torch.Tensor,
    names: tuple[str, ...],
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for index, name in enumerate(names):
        label_truth = truth == index
        label_prediction = prediction == index
        tp = int((label_truth & label_prediction).sum())
        fp = int((~label_truth & label_prediction).sum())
        fn = int((label_truth & ~label_prediction).sum())
        scores[name] = 2 * tp / max(1, 2 * tp + fp + fn)
    return scores


def weighted_binary_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    positive_weight: float = 1.0,
    negative_weight: float = 1.0,
) -> torch.Tensor:
    raw = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    weights = torch.where(
        targets >= 0.5,
        torch.as_tensor(positive_weight, dtype=raw.dtype, device=raw.device),
        torch.as_tensor(negative_weight, dtype=raw.dtype, device=raw.device),
    )
    return (raw * weights).mean()


def model_inputs(batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
    return (
        batch["input_ids"], batch["valid_mask"], batch["context_ids"],
        batch["self_callsign_ids"], batch["peer_callsign_ids"], batch["known_slots"],
    )


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, object]:
    model.eval()
    act_truth, act_predicted, topic_truth, topic_predicted = [], [], [], []
    register_truth, register_predicted = [], []
    procedure_truth, procedure_predicted = [], []
    slot_exact_correct = {name: 0 for name in SLOT_NAMES if name != "O"}
    slot_exact_total = {name: 0 for name in SLOT_NAMES if name != "O"}
    slot_correct = slot_total = register_correct = procedure_correct = commit_correct = examples = 0
    decoded_slot_correct = {name: 0 for name in SLOT_PATTERNS}
    decoded_slot_total = {name: 0 for name in SLOT_PATTERNS}
    ambiguous_total = ambiguous_false_accept = 0
    hard_negative_total = hard_negative_false_slot_fill = 0
    ask_provide_total = ask_provide_confusion = 0
    elliptical_total = elliptical_correct = 0
    for batch in loader:
        tensors = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
        acts, topics, slots, register, procedure, safe_to_commit = model(*model_inputs(tensors))
        act_prediction = torch.sigmoid(acts) >= 0.5
        topic_scores = torch.sigmoid(topics)
        topic_prediction = topic_scores >= 0.5
        act_truth.append(tensors["acts"].cpu())
        act_predicted.append(act_prediction.cpu())
        topic_truth.append(tensors["topics"].cpu())
        topic_predicted.append(topic_prediction.cpu())
        mask = tensors["slot_labels"] != -100
        valid = tensors["valid_mask"]
        slot_prediction = slots.argmax(-1)
        slot_correct += int(((slot_prediction == tensors["slot_labels"]) & mask).sum())
        slot_total += int(mask.sum())
        for name in slot_exact_total:
            label_id = SLOT_TO_ID[name]
            truth_positions = tensors["slot_labels"] == label_id
            prediction_positions = (slot_prediction == label_id) & valid
            has_truth = truth_positions.any(dim=1)
            exact = ((truth_positions == prediction_positions) | ~valid).all(dim=1)
            slot_exact_correct[name] += int((exact & has_truth).sum())
            slot_exact_total[name] += int(has_truth.sum())
        register_correct += int((register.argmax(-1) == tensors["register"]).sum())
        procedure_prediction = procedure.argmax(-1)
        register_truth.append(tensors["register"].cpu())
        register_predicted.append(register.argmax(-1).cpu())
        procedure_truth.append(tensors["procedure"].cpu())
        procedure_predicted.append(procedure_prediction.cpu())
        procedure_correct += int((procedure_prediction == tensors["procedure"]).sum())
        commit_prediction = torch.sigmoid(safe_to_commit) >= 0.5
        commit_truth = tensors["safe_to_commit"].bool()
        commit_correct += int((commit_prediction == commit_truth).sum())
        truth_has_slot = ((tensors["slot_labels"] != SLOT_TO_ID["O"]) & mask).any(dim=1)
        for row, text in enumerate(batch["texts"]):
            score_by_topic = {
                name: float(topic_scores[row, TOPIC_TO_ID[name]])
                for name in SLOT_PATTERNS
            }
            decoded = decode_slot_values(
                text, score_by_topic, safe_to_commit=bool(commit_prediction[row]),
            )
            labels = tensors["slot_labels"][row]
            for name in decoded_slot_total:
                label_id = SLOT_TO_ID[name]
                if not bool((labels == label_id).any()):
                    continue
                decoded_slot_total[name] += 1
                truth_values = labeled_slot_values(text, labels.tolist(), label_id)
                if decoded.get(name, ()) == truth_values:
                    decoded_slot_correct[name] += 1
        prediction_has_slot = ((slot_prediction != SLOT_TO_ID["O"]) & valid).any(dim=1)
        hard_negative = ~truth_has_slot & ~commit_truth
        hard_negative_total += int(hard_negative.sum())
        hard_negative_false_slot_fill += int((hard_negative & prediction_has_slot).sum())
        truth_ask = tensors["acts"][:, ACT_TO_ID["ASK"]].bool()
        truth_provide = tensors["acts"][:, ACT_TO_ID["PROVIDE"]].bool()
        prediction_ask = act_prediction[:, ACT_TO_ID["ASK"]]
        prediction_provide = act_prediction[:, ACT_TO_ID["PROVIDE"]]
        exclusive_question_or_answer = truth_ask ^ truth_provide
        confused = (
            (truth_ask & ~truth_provide & prediction_provide)
            | (truth_provide & ~truth_ask & prediction_ask)
        )
        ask_provide_total += int(exclusive_question_or_answer.sum())
        ask_provide_confusion += int((confused & exclusive_question_or_answer).sum())
        elliptical = (
            (tensors["acts"].sum(dim=1) == 1)
            & truth_provide
            & (tensors["procedure"] == PROCEDURE_TO_ID["INTELLIGIBLE_NONCANONICAL"])
            & (tensors["register"] == REGISTER_TO_ID["CW_ABBREVIATED"])
            & (tensors["context_ids"][:, 1] != 0)
        )
        elliptical_success = (
            prediction_provide & commit_prediction
            & (topic_prediction == tensors["topics"].bool()).all(dim=1)
        )
        elliptical_total += int(elliptical.sum())
        elliptical_correct += int((elliptical & elliptical_success).sum())
        ambiguous = tensors["procedure"] == PROCEDURE_TO_ID["AMBIGUOUS"]
        ambiguous_total += int(ambiguous.sum())
        ambiguous_false_accept += int((commit_prediction & ambiguous).sum())
        examples += tensors["input_ids"].shape[0]
    act_truth_tensor = torch.cat(act_truth)
    act_prediction_tensor = torch.cat(act_predicted)
    topic_truth_tensor = torch.cat(topic_truth)
    topic_prediction_tensor = torch.cat(topic_predicted)
    register_f1 = multiclass_f1_by_label(
        torch.cat(register_truth), torch.cat(register_predicted), REGISTER_NAMES,
    )
    procedure_f1 = multiclass_f1_by_label(
        torch.cat(procedure_truth), torch.cat(procedure_predicted), PROCEDURE_GRADES,
    )
    slot_exact = {
        name: slot_exact_correct[name] / max(1, slot_exact_total[name])
        for name in slot_exact_total
    }
    critical_slot_names = ("CALLSIGN", "RST", "POWER")
    other_slot_names = tuple(name for name in slot_exact if name not in critical_slot_names)
    hard_negative_false_fill_rate = hard_negative_false_slot_fill / max(1, hard_negative_total)
    ask_provide_confusion_rate = ask_provide_confusion / max(1, ask_provide_total)
    elliptical_accuracy = elliptical_correct / max(1, elliptical_total)
    decoded_slot_exact = {
        name: decoded_slot_correct[name] / max(1, decoded_slot_total[name])
        for name in SLOT_PATTERNS
    }
    ambiguous_false_accept_rate = ambiguous_false_accept / max(1, ambiguous_total)
    release_gates = {
        "actMacroF1": macro_f1(act_truth_tensor, act_prediction_tensor) >= 0.97,
        "topicMacroF1": macro_f1(topic_truth_tensor, topic_prediction_tensor) >= 0.97,
        "criticalSlotExact": all(decoded_slot_exact[name] >= 0.995 for name in critical_slot_names),
        "otherSlotExact": all(decoded_slot_exact[name] >= 0.97 for name in other_slot_names),
        "askProvideConfusion": ask_provide_confusion_rate <= 0.005,
        "registerMacroF1": sum(register_f1.values()) / max(1, len(register_f1)) >= 0.97,
        "ellipticalAnswerAccuracy": elliptical_accuracy >= 0.98,
        "hardNegativeFalseSlotFill": hard_negative_false_fill_rate <= 0.005,
        "ambiguousFalseAccept": ambiguous_false_accept_rate <= 0.005,
    }
    return {
        "actMicroF1": micro_f1(act_truth_tensor, act_prediction_tensor),
        "actMacroF1": macro_f1(act_truth_tensor, act_prediction_tensor),
        "actF1ByLabel": f1_by_label(act_truth_tensor, act_prediction_tensor, ACT_NAMES),
        "topicMicroF1": micro_f1(topic_truth_tensor, topic_prediction_tensor),
        "topicMacroF1": macro_f1(topic_truth_tensor, topic_prediction_tensor),
        "topicF1ByLabel": f1_by_label(topic_truth_tensor, topic_prediction_tensor, TOPIC_NAMES),
        "slotAccuracy": slot_correct / max(1, slot_total),
        "slotExactByLabel": slot_exact,
        "rawCriticalSlotExactMinimum": min(slot_exact[name] for name in critical_slot_names),
        "rawOtherSlotExactMinimum": min(slot_exact[name] for name in other_slot_names),
        "decodedSlotExactByLabel": decoded_slot_exact,
        "criticalSlotExactMinimum": min(decoded_slot_exact[name] for name in critical_slot_names),
        "otherSlotExactMinimum": min(decoded_slot_exact[name] for name in other_slot_names),
        "hardNegativeFalseSlotFill": hard_negative_false_fill_rate,
        "askProvideConfusion": ask_provide_confusion_rate,
        "ellipticalAnswerAccuracy": elliptical_accuracy,
        "registerAccuracy": register_correct / max(1, examples),
        "registerMacroF1": sum(register_f1.values()) / max(1, len(register_f1)),
        "registerF1ByLabel": register_f1,
        "procedureAccuracy": procedure_correct / max(1, examples),
        "procedureMacroF1": sum(procedure_f1.values()) / max(1, len(procedure_f1)),
        "procedureF1ByLabel": procedure_f1,
        "safeCommitAccuracy": commit_correct / max(1, examples),
        "ambiguousFalseAccept": ambiguous_false_accept_rate,
        "releaseGates": release_gates,
        "releaseReady": all(release_gates.values()),
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--train-samples", type=int, default=50_000)
    parser.add_argument("--val-samples", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--act-positive-weight", type=float, default=1.5)
    parser.add_argument("--topic-positive-weight", type=float, default=2.0)
    parser.add_argument("--safe-negative-weight", type=float, default=3.0)
    parser.add_argument("--slot-outside-weight", type=float, default=0.35)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = SemanticModelConfig()
    model = QsoSemanticFormer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    class_ce = nn.CrossEntropyLoss(ignore_index=-100)
    slot_weights = torch.ones(len(SLOT_NAMES), device=device)
    slot_weights[SLOT_TO_ID["O"]] = args.slot_outside_weight
    slot_ce = nn.CrossEntropyLoss(ignore_index=-100, weight=slot_weights)
    train_data = SyntheticSemanticDataset(args.train_samples, seed=args.seed)
    validation_data = SyntheticSemanticDataset(args.val_samples, seed=args.seed + 90_000_000)
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size, num_workers=args.workers // 2, collate_fn=collate_semantic)
    best = -1.0
    print(json.dumps({"event": "start", "modelVersion": SEMANTIC_MODEL_VERSION, "parameters": semantic_parameter_count(model), "device": str(device)}), flush=True)
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate_semantic)
        model.train()
        total_loss = seen = 0.0
        for batch in loader:
            batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                acts, topics, slots, register, procedure, safe_to_commit = model(*model_inputs(batch))
                loss = (
                    weighted_binary_cross_entropy(
                        acts.float(), batch["acts"], positive_weight=args.act_positive_weight,
                    )
                    + weighted_binary_cross_entropy(
                        topics.float(), batch["topics"], positive_weight=args.topic_positive_weight,
                    )
                    + slot_ce(
                        slots.float().reshape(-1, slots.shape[-1]), batch["slot_labels"].reshape(-1),
                    )
                    + 0.35 * class_ce(register.float(), batch["register"])
                    + 0.5 * class_ce(procedure.float(), batch["procedure"])
                    + 0.5 * weighted_binary_cross_entropy(
                        safe_to_commit.float(), batch["safe_to_commit"],
                        negative_weight=args.safe_negative_weight,
                    )
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach()) * batch["input_ids"].shape[0]
            seen += batch["input_ids"].shape[0]
        scheduler.step()
        metrics = evaluate(model, validation_loader, device)
        score = (
            metrics["actMacroF1"] + metrics["topicMacroF1"]
            + metrics["criticalSlotExactMinimum"] + metrics["otherSlotExactMinimum"]
            + metrics["registerMacroF1"] + metrics["procedureMacroF1"]
            + metrics["safeCommitAccuracy"] + metrics["ellipticalAnswerAccuracy"]
            - metrics["ambiguousFalseAccept"] - metrics["hardNegativeFalseSlotFill"]
            - metrics["askProvideConfusion"]
        )
        record = {"event": "epoch", "epoch": epoch, "seconds": round(time.perf_counter() - started, 2), "loss": total_loss / seen, **metrics}
        print(json.dumps(record), flush=True)
        checkpoint = {
            "modelVersion": SEMANTIC_MODEL_VERSION, "semanticSchemaVersion": SEMANTIC_SCHEMA_VERSION,
            "config": config.to_dict(), "epoch": epoch, "metrics": metrics, "model": model.state_dict(),
            "training": {
                "actPositiveWeight": args.act_positive_weight,
                "topicPositiveWeight": args.topic_positive_weight,
                "safeNegativeWeight": args.safe_negative_weight,
                "slotOutsideWeight": args.slot_outside_weight,
                "seed": args.seed,
            },
        }
        torch.save(checkpoint, args.output / "last.pt")
        if score > best:
            best = score
            torch.save(checkpoint, args.output / "best.pt")
            (args.output / "best-metrics.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
