<div align="center">

# Game-Morse-Adventurer-AI

### Local semantic understanding for fictional CW QSOs

[**English**](./README.md) · [**简体中文**](./README.zh-CN.md) · [**繁體中文**](./README.zh-TW.md) · [**日本語**](./README.ja.md) · [**Español**](./README.es.md) · [**Deutsch**](./README.de.md) · [**Русский**](./README.ru.md)

**Game repository:** [Game-Morse-Adventurer](https://github.com/Arsenic-er/Game-Morse-Adventurer)

</div>

## About

This repository contains the deterministic training dataset, generator, model code, evaluation tools, and published artifacts for the local QSO semantic-understanding system used by Game-Morse-Adventurer.

It is not an open-domain conversation model. It classifies decoded, Morse-friendly text so the game can interpret QSO intent, topics, values, procedure quality, and whether a transmission is safe to commit.

## Data policy

All training, validation, and confirmation data is generated deterministically from synthetic QSO templates. Compressed review copies and SHA-256 hashes are included under [`datasets/qso-semantics-v4`](./datasets/qso-semantics-v4); [`DATASET.md`](./DATASET.md) documents every fixed split.

- No player data, telemetry, save files, messages, or recordings.
- No scraped QSO logs or private communications.
- No dependency on an online service during gameplay.
- Synthetic callsigns and identities are fictional; any resemblance is coincidental.
- The included generator reproduces the labeled data from documented seeds.

The released checkpoint used 100,000 synthetic training examples for 16 epochs, a separate 15,000-example validation set, and three disjoint 15,000-example confirmation sets with seeds `90268741`, `90268742`, and `90268743`.

## QsoSemanticFormer v0.4

`QsoSemanticFormer` is a six-layer character Transformer with 942,898 parameters. It receives decoded text plus QSO context and produces six output groups:

- speech acts;
- topics;
- token-level value slots;
- register;
- procedure grade;
- safe-to-commit state.

Published files include:

- `qso-semanticformer-v0.4.release.pt` — promoted PyTorch checkpoint;
- `qso-semanticformer-v0.4.release.fp32.onnx` — FP32 reference;
- `qso-semanticformer-v0.4.release.int8.onnx` — CPU desktop runtime;
- runtime contract, manifest, parity, benchmark, and release-gate reports.

The upstream PulseFormer key-event decoder is not included here. It remains independently evaluated and has not been approved as a shipped game feature.

## Reproduce

Python 3.11 or later is required.

```bash
python -m pip install -e ".[export,test]"
python -m pytest
python scripts/export_semantic_dataset.py --all

python -m cw_pulseformer.train_semantic \
  --epochs 16 \
  --train-samples 100000 \
  --val-samples 15000 \
  --seed 20260805 \
  --output runs/semantic-v0.4
```

Evaluation and export commands, schemas, and release gates are documented in [`MODEL_ARCHITECTURE.md`](./MODEL_ARCHITECTURE.md) and [`MODEL_CARD.md`](./MODEL_CARD.md).

## Limitations

Reported results measure a deterministic synthetic protocol domain. They do not measure arbitrary conversation, raw Morse decoding, RF noise, propagation, or operator timing. Those concerns belong to separate game and signal-processing layers.

On-air text is limited to Morse-friendly ASCII English abbreviations and romanized values. Interface localization is handled by the game.

## Author and copyright

**Author:** [Arsenic-er](https://github.com/Arsenic-er)

Copyright © 2026 Arsenic-er. All rights reserved. No additional license is granted.
