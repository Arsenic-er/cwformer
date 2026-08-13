<div align="center">

# cwformer

### Comprensión semántica local para QSO ficticios en CW

[**English**](./README.md) · [**简体中文**](./README.zh-CN.md) · [**繁體中文**](./README.zh-TW.md) · [**日本語**](./README.ja.md) · [**Español**](./README.es.md) · [**Deutsch**](./README.de.md) · [**Русский**](./README.ru.md)

**Repositorio del juego:** [Game-Morse-Adventurer](https://github.com/Arsenic-er/Game-Morse-Adventurer)

</div>

## Acerca del proyecto

Este repositorio contiene el conjunto de datos determinista, su generador, el código del modelo, las herramientas de evaluación y los artefactos publicados del sistema local de comprensión semántica de QSO usado por Game-Morse-Adventurer.

No es un modelo de conversación general. Clasifica texto ya decodificado y adaptado al código Morse para que el juego interprete la intención, los temas, los valores, la calidad del procedimiento y si una transmisión permite continuar el QSO de forma segura.

## Política de datos

Todos los datos se generan de forma determinista a partir de plantillas QSO sintéticas. Las copias comprimidas y sus SHA-256 están en [`datasets/qso-semantics-v4`](./datasets/qso-semantics-v4), y [`DATASET.md`](./DATASET.md) documenta cada partición fija.

- No se utilizan datos, telemetría, partidas, mensajes ni grabaciones de jugadores.
- No se recopilan registros QSO ni comunicaciones privadas.
- El juego no necesita servicios en línea.
- Los indicativos y las identidades sintéticas son ficticios; cualquier parecido es coincidencia.
- El generador reproduce los datos etiquetados mediante semillas documentadas.

El punto de control se entrenó durante 16 épocas con 100.000 ejemplos, se validó con 15.000 ejemplos independientes y se confirmó con tres conjuntos separados de 15.000 ejemplos, con semillas `90268741`, `90268742` y `90268743`.

## QsoSemanticFormer v0.4

`QsoSemanticFormer` es un Transformer de caracteres de seis capas con 942.898 parámetros. Produce actos comunicativos, temas, campos de valor por carácter, registro, calidad del procedimiento y estado seguro para continuar.

Se publican el punto de control PyTorch, la referencia FP32 ONNX, el modelo INT8 ONNX para CPU y los informes de contrato, manifiesto, paridad, rendimiento y criterios de publicación.

El decodificador de eventos PulseFormer no se incluye aquí; continúa bajo evaluación independiente y no está aprobado como función publicada del juego.

## Reproducción

Se requiere Python 3.11 o posterior.

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

Los comandos de evaluación y exportación, los esquemas y los criterios se describen en [`MODEL_ARCHITECTURE.md`](./MODEL_ARCHITECTURE.md) y [`MODEL_CARD.md`](./MODEL_CARD.md).

## Limitaciones

Los resultados miden un dominio sintético y determinista de procedimientos de radio. No miden conversación arbitraria, decodificación Morse sin procesar, ruido de RF, propagación ni temporización del operador. Estas funciones pertenecen a capas independientes.

El texto en el aire se limita a abreviaturas inglesas ASCII aptas para Morse y valores romanizados. La localización de la interfaz se gestiona en el juego.

## Autor y derechos de autor

**Autor:** [Arsenic-er](https://github.com/Arsenic-er)

Copyright © 2026 Arsenic-er. Todos los derechos reservados. No se concede ninguna licencia adicional.
