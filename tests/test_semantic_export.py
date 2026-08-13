import pytest

from cw_pulseformer.export_semantic import build_runtime_contract, export_semantic_model
from cw_pulseformer.semantic_data import REGISTER_NAMES, TOPIC_NAMES
from cw_pulseformer.semantic_model import (
    SEMANTIC_INPUT_NAMES,
    SEMANTIC_OUTPUT_NAMES,
    QsoSemanticFormer,
    SemanticModelConfig,
)


def test_semantic_runtime_contract_records_context_and_safe_commit() -> None:
    config = SemanticModelConfig(d_model=16, layers=1, heads=4, ff_dim=32).to_dict()
    contract = build_runtime_contract(config, [])
    assert tuple(contract["inputs"]) == SEMANTIC_INPUT_NAMES
    assert contract["inputs"]["context_ids"]["columns"] == ["phase", "pendingQuestion"]
    assert contract["inputs"]["known_slots"]["shape"] == ["batch", len(TOPIC_NAMES)]
    assert contract["contextSchema"]["pendingQuestionToId"]["NONE"] == 0
    assert contract["contextSchema"]["pendingQuestionToId"]["NAME"] > 0
    assert contract["contextSchema"]["phaseToId"]["EXCHANGE"] >= 0
    safe_commit = contract["outputs"]["safe_to_commit_logits"]
    assert safe_commit["threshold"] == 0.5
    assert "forbids QSO state mutation" in safe_commit["meaning"]
    assert contract["outputs"]["register_logits"]["names"] == REGISTER_NAMES
    assert contract["contractVersion"] == "qso-semantic-runtime-4"


def test_semantic_onnx_export_smoke_has_six_inputs_and_six_outputs(tmp_path) -> None:
    onnx = pytest.importorskip("onnx")
    model = QsoSemanticFormer(
        SemanticModelConfig(d_model=16, layers=1, heads=4, ff_dim=32, dropout=0.0),
    ).eval()
    output = tmp_path / "semantic-smoke.onnx"
    export_semantic_model(model, output)
    graph = onnx.load(output).graph
    assert tuple(value.name for value in graph.input) == SEMANTIC_INPUT_NAMES
    assert tuple(value.name for value in graph.output) == SEMANTIC_OUTPUT_NAMES
