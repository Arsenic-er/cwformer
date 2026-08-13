# QsoSemanticFormer v0.4 model card

## Intended use

QsoSemanticFormer classifies the meaning and procedural state of short,
already-decoded CW QSO text inside Game Morse Adventurer. The game combines its
semantic result with deterministic signal-quality observations and a fictional
NPC's skill, personality, and tolerance policy.

It does not decode raw audio or Morse pulses, produce unrestricted dialogue,
or replace final game scoring logic.

## Architecture

- character-level Transformer;
- 6 encoder layers, 128-dimensional hidden state, 4 attention heads;
- late dialogue-context fusion for token-slot decisions;
- 942,898 trainable parameters;
- six outputs: acts, topics, token slots, register, procedure grade, and
  safe-to-commit;
- ONNX opset 17, CPU-only runtime supported.

## Released artifacts

The promoted release is in `releases/semantic-v0.4/`:

- dynamic INT8 ONNX: 1,164,897 bytes, used by the desktop game;
- FP32 ONNX: 3,830,708 bytes, parity/reference artifact;
- PyTorch checkpoint: 3,810,346 bytes, reproducible export source;
- runtime contract, manifest, and evaluation reports.

## Data and evaluation

The model was trained from scratch on 100,000 deterministic synthetic examples
for 16 epochs. Checkpoint selection used 15,000 separate validation examples.
Promotion required three disjoint 15,000-example confirmation suites. Exact
splits and regeneration instructions are documented in `DATASET.md`.

## Limitations

Reported metrics cover the deterministic synthetic CW-protocol domain. They do
not establish accuracy on arbitrary human conversation, raw RF audio, QRM,
QSB, or unreviewed third-party text. On-air text is Morse-friendly ASCII,
English abbreviations, and romanized values; the game's seven-language UI is a
separate layer. Values with game grammars or finite catalogs are constrained by
a deterministic decoder before gameplay state is committed.

The project is not affiliated with or endorsed by amateur-radio equipment
manufacturers or trademark owners named in fictional examples.
