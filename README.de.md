<div align="center">

# Game-Morse-Adventurer-AI

### Lokales semantisches Verständnis für fiktive CW-QSOs

[**English**](./README.md) · [**简体中文**](./README.zh-CN.md) · [**繁體中文**](./README.zh-TW.md) · [**日本語**](./README.ja.md) · [**Español**](./README.es.md) · [**Deutsch**](./README.de.md) · [**Русский**](./README.ru.md)

**Spiel-Repository:** [Game-Morse-Adventurer](https://github.com/Arsenic-er/Game-Morse-Adventurer)

</div>

## Über das Projekt

Dieses Repository enthält den deterministischen Trainingsdatensatz, den Generator, den Modellcode, die Evaluierungswerkzeuge und die veröffentlichten Artefakte des lokalen QSO-Semantiksystems von Game-Morse-Adventurer.

Es ist kein allgemeines Dialogmodell. Es klassifiziert dekodierten, Morse-tauglichen Text, damit das Spiel QSO-Absicht, Themen, Werte, Verfahrensqualität und die sichere Fortsetzung beurteilen kann.

## Datenrichtlinie

Alle Daten werden deterministisch aus synthetischen QSO-Vorlagen erzeugt. Komprimierte Prüfdaten und SHA-256-Werte liegen unter [`datasets/qso-semantics-v4`](./datasets/qso-semantics-v4); [`DATASET.md`](./DATASET.md) beschreibt jede feste Aufteilung.

- Keine Spielerdaten, Telemetrie, Spielstände, Nachrichten oder Aufnahmen.
- Keine gesammelten QSO-Logs oder privaten Mitteilungen.
- Kein Onlinedienst während des Spielens erforderlich.
- Synthetische Rufzeichen und Identitäten sind fiktiv; Ähnlichkeiten sind zufällig.
- Der Generator reproduziert die beschrifteten Daten mit dokumentierten Seeds.

Der Checkpoint wurde 16 Epochen lang mit 100.000 Beispielen trainiert. Hinzu kamen 15.000 getrennte Validierungsbeispiele und drei unabhängige Bestätigungssätze mit je 15.000 Beispielen und den Seeds `90268741`, `90268742` und `90268743`.

## QsoSemanticFormer v0.4

`QsoSemanticFormer` ist ein sechsschichtiger Zeichen-Transformer mit 942.898 Parametern. Er erzeugt Sprechakte, Themen, zeichenbasierte Wertefelder, Register, Verfahrensbewertung und den Zustand zur sicheren Fortsetzung.

Veröffentlicht werden der PyTorch-Checkpoint, die FP32-ONNX-Referenz, das INT8-ONNX-CPU-Modell sowie Laufzeitvertrag, Manifest und Paritäts-, Leistungs- und Freigabeberichte.

Der vorgeschaltete PulseFormer-Decoder ist nicht enthalten; er wird getrennt evaluiert und ist nicht als Spielfunktion freigegeben.

## Reproduktion

Python 3.11 oder neuer ist erforderlich.

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

Evaluierungs- und Exportbefehle, Schemata und Freigabekriterien stehen in [`MODEL_ARCHITECTURE.md`](./MODEL_ARCHITECTURE.md) und [`MODEL_CARD.md`](./MODEL_CARD.md).

## Einschränkungen

Die Ergebnisse messen eine deterministische synthetische Funkprotokolldomäne, nicht beliebige Gespräche, rohe Morse-Dekodierung, HF-Störungen, Ausbreitung oder Operator-Timing. Dafür sind getrennte Schichten zuständig.

Der Funktext ist auf Morse-taugliche englische ASCII-Abkürzungen und romanisierte Werte begrenzt. Die UI-Lokalisierung erfolgt im Spiel.

## Autor und Urheberrecht

**Autor:** [Arsenic-er](https://github.com/Arsenic-er)

Copyright © 2026 Arsenic-er. Alle Rechte vorbehalten. Es wird keine zusätzliche Lizenz erteilt.
