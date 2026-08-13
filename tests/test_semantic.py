import numpy as np
import pytest
import torch

from cw_pulseformer.semantic_data import (
    ACT_TO_ID,
    MAX_CALLSIGN_LENGTH,
    PROCEDURE_TO_ID,
    REGISTER_NAMES,
    REGISTER_TO_ID,
    SLOT_TO_ID,
    TOPIC_NAMES,
    TOPIC_TO_ID,
    build_omitted_answer_spec,
    collate_semantic,
    encode_semantic_spec,
    generate_semantic_spec,
)
from cw_pulseformer.semantic_model import (
    SEMANTIC_INPUT_NAMES,
    SEMANTIC_OUTPUT_NAMES,
    QsoSemanticFormer,
    SemanticModelConfig,
    semantic_parameter_count,
)
from cw_pulseformer.train_semantic import macro_f1


def _model_inputs(batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
    return tuple(batch[name] for name in SEMANTIC_INPUT_NAMES)


def test_macro_f1_keeps_rare_labels_visible() -> None:
    truth = torch.tensor([
        [1, 0, 1],
        [1, 0, 0],
    ])
    prediction = torch.tensor([
        [1, 0, 0],
        [1, 0, 0],
    ])
    assert macro_f1(truth, prediction) == pytest.approx(1 / 3)


def test_optional_question_and_provide_topics_are_multilabel() -> None:
    observed_question = observed_provide = False
    for seed in range(1_000):
        sample = encode_semantic_spec(generate_semantic_spec(np.random.default_rng(seed)))
        if sample["acts"][ACT_TO_ID["ASK"]] and sample["topics"].sum() >= 1:
            observed_question = True
        if sample["acts"][ACT_TO_ID["PROVIDE"]] and sample["topics"].sum() >= 1:
            observed_provide = True
        if observed_question and observed_provide:
            break
    assert observed_question and observed_provide
    assert ACT_TO_ID["ASK"] != ACT_TO_ID["PROVIDE"]


def test_name_age_location_weather_power_topics_exist() -> None:
    required = {"NAME", "AGE", "LOCATION", "WEATHER", "POWER"}
    assert required.issubset(TOPIC_TO_ID)


def test_v04_control_acts_and_cq_facets_are_generated() -> None:
    required_acts = {
        "IDENTIFY", "ACK", "CONFIRM", "REPEAT_REQUEST", "SPEED_REQUEST",
        "GREETING", "HANDOVER",
    }
    required_topics = {"REGION", "CQ_SCOPE"}
    assert required_acts.issubset(ACT_TO_ID)
    assert required_topics.issubset(TOPIC_TO_ID)
    observed_acts: set[str] = set()
    observed_topics: set[str] = set()
    observed_registers: set[str] = set()
    for seed in range(10_000):
        spec = generate_semantic_spec(np.random.default_rng(seed))
        observed_acts.update(spec.acts)
        observed_topics.update(spec.topics)
        observed_registers.add(spec.register)
    assert required_acts.issubset(observed_acts)
    assert required_topics.issubset(observed_topics)
    assert set(REGISTER_NAMES).issubset(observed_registers)


def test_noncanonical_and_incomplete_question_provide_templates_are_observable() -> None:
    observed: set[tuple[str, str]] = set()
    for seed in range(5_000):
        spec = generate_semantic_spec(np.random.default_rng(seed))
        family = (
            "PROVIDE" if "PROVIDE" in spec.acts and "REPORT" not in spec.acts
            else "QUESTION" if "ASK" in spec.acts and "REPEAT_REQUEST" not in spec.acts
            else None
        )
        if family is None:
            continue
        if spec.procedure == "INTELLIGIBLE_NONCANONICAL":
            observed.add((family, spec.procedure))
            if family == "QUESTION":
                assert any(marker in spec.text for marker in ("WHAT ", "HOW ", "WHERE ", "PSE "))
            elif not spec.text.endswith(" K"):
                assert any(marker in spec.text for marker in (" IS ", "MY ", "I AM ", "I USE ", "THE ", "YOUR "))
        elif spec.procedure == "INCOMPLETE":
            observed.add((family, spec.procedure))
            if family == "QUESTION":
                assert spec.text.startswith("UR ")
                assert "?" not in spec.text
            else:
                assert spec.text.endswith("IS")
                assert not spec.spans
                assert not spec.safe_to_commit
        if len(observed) == 4:
            break
    assert observed == {
        ("QUESTION", "INTELLIGIBLE_NONCANONICAL"),
        ("PROVIDE", "INTELLIGIBLE_NONCANONICAL"),
        ("QUESTION", "INCOMPLETE"),
        ("PROVIDE", "INCOMPLETE"),
    }


def test_same_text_and_same_relevant_context_never_conflict() -> None:
    labels_by_observation: dict[tuple[str, str, str], tuple[str, str, tuple[str, ...], tuple[str, ...], bool]] = {}
    for seed in range(30_000):
        spec = generate_semantic_spec(np.random.default_rng(seed))
        key = (spec.text, spec.context.phase, spec.context.pending_question)
        label = (spec.procedure, spec.register, spec.acts, spec.topics, spec.safe_to_commit)
        previous = labels_by_observation.setdefault(key, label)
        assert previous == label, f"{key!r} was generated with both {previous!r} and {label!r}"


def test_pending_name_turn_disambiguates_bare_ken_k() -> None:
    contextual = build_omitted_answer_spec("NAME", "KEN", pending_question="NAME")
    context_free = build_omitted_answer_spec("NAME", "KEN", pending_question="NONE")
    assert contextual.text == context_free.text == "KEN K"
    contextual_sample = encode_semantic_spec(contextual)
    context_free_sample = encode_semantic_spec(context_free)
    assert contextual_sample["acts"][ACT_TO_ID["PROVIDE"]] == 1
    assert contextual_sample["topics"][TOPIC_TO_ID["NAME"]] == 1
    assert contextual_sample["procedure"].item() == PROCEDURE_TO_ID["INTELLIGIBLE_NONCANONICAL"]
    assert contextual_sample["safe_to_commit"].item() == 1
    assert (contextual_sample["slot_labels"] == SLOT_TO_ID["NAME"]).sum().item() == 3
    assert context_free_sample["acts"].sum().item() == 0
    assert context_free_sample["topics"].sum().item() == 0
    assert context_free_sample["procedure"].item() == PROCEDURE_TO_ID["AMBIGUOUS"]
    assert context_free_sample["safe_to_commit"].item() == 0
    valid_context_free_slots = context_free_sample["slot_labels"][context_free_sample["valid_mask"]]
    assert torch.all(valid_context_free_slots == SLOT_TO_ID["O"])


def test_pending_qth_accepts_bare_location_but_wrong_context_rejects() -> None:
    contextual = build_omitted_answer_spec("LOCATION", "AOBA", pending_question="LOCATION")
    wrong_topic = build_omitted_answer_spec("LOCATION", "AOBA", pending_question="WEATHER")
    wrong_phase = build_omitted_answer_spec(
        "LOCATION", "AOBA", pending_question="LOCATION", phase="CLOSING",
    )
    assert contextual.text == wrong_topic.text == wrong_phase.text == "AOBA K"
    assert contextual.acts == ("PROVIDE",)
    assert contextual.topics == ("LOCATION",)
    assert contextual.safe_to_commit
    for rejected in (wrong_topic, wrong_phase):
        assert rejected.procedure == "AMBIGUOUS"
        assert rejected.acts == rejected.topics == ()
        assert not rejected.safe_to_commit


def test_context_tensor_contract_is_fixed_and_collates() -> None:
    samples = [
        encode_semantic_spec(build_omitted_answer_spec("NAME", "KEN", pending_question="NAME")),
        encode_semantic_spec(build_omitted_answer_spec("LOCATION", "AOBA", pending_question="NONE")),
    ]
    batch = collate_semantic(samples)
    assert batch["context_ids"].shape == (2, 2)
    assert batch["self_callsign_ids"].shape == (2, MAX_CALLSIGN_LENGTH)
    assert batch["peer_callsign_ids"].shape == (2, MAX_CALLSIGN_LENGTH)
    assert batch["known_slots"].shape == (2, len(TOPIC_NAMES))
    assert batch["context_ids"].dtype == torch.long
    assert batch["known_slots"].dtype == torch.float32
    assert batch["register"].shape == (2,)
    assert batch["register"].dtype == torch.long


def test_semantic_model_shapes_budget_and_exportable_signature() -> None:
    config = SemanticModelConfig()
    model = QsoSemanticFormer(config).eval()
    samples = [
        encode_semantic_spec(build_omitted_answer_spec("NAME", "KEN", pending_question="NAME")),
        encode_semantic_spec(build_omitted_answer_spec("NAME", "KEN", pending_question="NONE")),
    ]
    batch = collate_semantic(samples)
    with torch.inference_mode():
        outputs = model(*_model_inputs(batch))
    acts, topics, slots, register, procedure, safe_to_commit = outputs
    assert SEMANTIC_OUTPUT_NAMES == (
        "act_logits", "topic_logits", "slot_logits", "register_logits",
        "procedure_logits", "safe_to_commit_logits",
    )
    assert acts.shape == (2, len(ACT_TO_ID))
    assert topics.shape == (2, len(TOPIC_TO_ID))
    assert slots.shape[:2] == (2, config.max_length)
    assert register.shape == (2, len(REGISTER_NAMES))
    assert procedure.shape == (2, len(PROCEDURE_TO_ID))
    assert safe_to_commit.shape == (2,)
    assert len(REGISTER_TO_ID) == len(REGISTER_NAMES)
    assert 200_000 <= semantic_parameter_count(model) <= 1_500_000
