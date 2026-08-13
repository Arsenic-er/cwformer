# CW traffic semantics v0.4

The runtime is split into three auditable layers:

1. **PulseFormer** converts key events to N-best text and timing/style features.
2. **QsoSemanticFormer** identifies speech acts, topics, register, procedure grade, safe-to-commit state and token-level value spans.
3. **Context + NPC policy** checks propagation, contest and equipment facts, then lets the selected fictional operator decide whether to answer, ask, query copy or close.

This separation prevents an impatient NPC from turning a semantically correct transmission into an objectively wrong one. NPC personality never changes the text model's meaning or the objective signal observation.

## v0.4 schema

- Register: `CW_PROCEDURAL`, `CW_ABBREVIATED`, `NATURAL_TEXT`, `MIXED`, `NOISE_UNKNOWN`.
- Multi-label acts: `CQ`, `DIRECTED`, `REPLY`, `IDENTIFY`, `ASK`, `PROVIDE`, `REPORT`, `ACK`, `CONFIRM`, `CORRECTION`, `REPEAT_REQUEST`, `SPEED_REQUEST`, `GREETING`, `THANKS`, `HANDOVER`, `SIGNOFF`.
- Topics: callsign, name, age, location, weather, power, rig, antenna, RST, target region and CQ scope.
- Procedure grades: canonical, intelligible noncanonical, incomplete, ambiguous and irrelevant.
- Six ONNX outputs: act, topic, token slot, register, procedure and safe-to-commit logits.

The model receives the current QSO phase, pending question, both known callsigns and known slots. This resolves elliptical answers such as `KEN K` only when the NPC has just asked for the operator's name.

## Architecture and local runtime

`QsoSemanticFormer` is a six-layer character Transformer with late dialogue-context fusion in the token slot head. It has 942,898 parameters. The release artifacts use ONNX opset 17:

- FP32: 3,830,708 bytes;
- dynamic INT8: 1,164,897 bytes;
- six fixed inputs and six outputs under runtime contract `qso-semantic-runtime-4`.

On the server CPU benchmark at batch size 1, FP32 hot median latency was 3.98 ms and INT8 hot median latency was 3.79 ms. The game does not need a GPU and must not contact the training server during play.

## Hybrid value decoding

The neural model decides the intent, topic and whether a turn is safe to commit. A deterministic decoder then constrains values that have a game grammar or a finite save-specific catalog:

- callsign, RST, power and age use format/context constraints;
- name, location, weather, rig and antenna use replaceable catalogs;
- region and CQ scope use their configured game values.

Catalogs are parameters, not baked-in world state. Procedurally generated NPC names and locations can therefore supply the current save's catalog. Raw token accuracy remains in every report as a diagnostic; release gates use the values the game actually receives after semantic safety and constrained decoding.

## Training and confirmation

The release checkpoint was trained from scratch on 100,000 deterministic synthetic examples for 16 epochs with training seed `20260805`. The training validation set used 15,000 separate examples. Promotion required three additional 15,000-example confirmation sets with seeds `90268741`, `90268742` and `90268743`.

Across the three confirmation sets:

- act macro-F1: 0.999993 to 1.000000;
- topic macro-F1: 0.999191 to 0.999326;
- decoded callsign/RST/power exact minimum: 1.000000;
- decoded other-slot exact minimum: 0.985437 to 0.988764;
- ASK/PROVIDE confusion: 0;
- elliptical-answer accuracy: 1.000000;
- hard-negative false slot fill: 0 to 0.000387;
- ambiguous false accept: 0 to 0.000975;
- register macro-F1: 1.000000.

The FP32 ONNX matched the PyTorch result on the promotion set. The INT8 ONNX passed the complete release suite on all three confirmation seeds.

## Release gates

- act macro-F1 >= 0.97;
- topic macro-F1 >= 0.97;
- decoded callsign/RST/power exact value >= 0.995;
- decoded other-slot exact value >= 0.97;
- ASK/PROVIDE confusion <= 0.5%;
- register macro-F1 >= 0.97;
- pending-question elliptical answer accuracy >= 0.98;
- hard-negative false slot fill <= 0.5%;
- ambiguous false accept <= 0.5%;
- NPC privacy-policy violations = 0.

## Scope and limitations

These figures measure the deterministic synthetic protocol domain, not arbitrary human conversation and not raw Morse decoding under real RF noise. Pulse recognition quality remains a separate upstream gate. Before shipping gameplay integration, the public runtime must still pass handwritten multilingual UI-independent behavior cases, corrupted N-best text cases and end-to-end NPC policy tests.

On-air language remains Morse-friendly ASCII English abbreviations and romanized text. UI localization remains a separate seven-language concern.
