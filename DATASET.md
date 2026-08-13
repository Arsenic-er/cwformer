# QSO Semantics v4 dataset

The dataset is synthetic, deterministic, and privacy-safe. It contains no
player keying, real QSO logs, scraped conversations, or real-person profiles.
Fictional callsigns, names, locations, equipment, and radio-procedure variants
are generated from versioned code.

## Fixed splits

| Split | Samples | Seed | Purpose |
| --- | ---: | ---: | --- |
| `train` | 100,000 | `20260805` | Model fitting for 16 epochs |
| `validation` | 15,000 | `110260805` | Checkpoint selection |
| `confirm-90268741` | 15,000 | `90268741` | Independent release confirmation |
| `confirm-90268742` | 15,000 | `90268742` | Independent release confirmation |
| `confirm-90268743` | 15,000 | `90268743` | Independent release confirmation |

The generator derives each row from `seed + index * 1,000,003`, so the same
revision, seed, and index reproduce the same example. The compressed JSONL
exports in `datasets/qso-semantics-v4/` are review copies of these exact splits;
the source of truth remains `cw_pulseformer/semantic_data.py`.

Each record contains normalized text, QSO phase, pending question, both known
callsigns, known fields, semantic acts, topics, procedure grade, register,
character spans, and the safe-to-commit label.

Regenerate all review files and their SHA-256 manifest with:

```bash
python scripts/export_semantic_dataset.py --all
```

The confirmation splits are evaluation-only and are not used for gradient
updates. See `MODEL_CARD.md` for model scope and release limitations.
