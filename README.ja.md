<div align="center">

# Game-Morse-Adventurer-AI

### 架空 CW QSO のためのローカル意味理解システム

[**English**](./README.md) · [**简体中文**](./README.zh-CN.md) · [**繁體中文**](./README.zh-TW.md) · [**日本語**](./README.ja.md) · [**Español**](./README.es.md) · [**Deutsch**](./README.de.md) · [**Русский**](./README.ru.md)

**ゲームリポジトリ：** [Game-Morse-Adventurer](https://github.com/Arsenic-er/Game-Morse-Adventurer)

</div>

## 概要

このリポジトリには、Game-Morse-Adventurer で使用するローカル QSO 意味理解システムの決定論的訓練データ、生成器、モデルコード、評価ツール、公開済み成果物が含まれます。

汎用会話モデルではありません。復号済みのモールス向けテキストを分類し、QSO の意図、話題、値、運用手順の品質、および通信を安全に進められるかをゲームが判断できるようにします。

## データポリシー

訓練、検証、確認データは、すべて合成 QSO テンプレートから決定論的に生成されます。圧縮済みの確認用データと SHA-256 は [`datasets/qso-semantics-v4`](./datasets/qso-semantics-v4)、固定分割の説明は [`DATASET.md`](./DATASET.md) にあります。

- プレイヤーデータ、テレメトリー、セーブデータ、メッセージ、録音は使用しません。
- QSO ログや私的通信を収集しません。
- ゲームプレイ中にオンラインサービスを必要としません。
- 合成されたコールサインと人物は架空であり、類似は偶然です。
- 収録された生成器と公開シードでラベル付きデータを再現できます。

公開チェックポイントは、100,000 件の合成訓練例を 16 エポック学習し、独立した 15,000 件の検証セットと、各 15,000 件の三つの確認セットで評価されました。確認シードは `90268741`、`90268742`、`90268743` です。

## QsoSemanticFormer v0.4

`QsoSemanticFormer` は、942,898 パラメータを持つ六層の文字レベル Transformer です。復号テキストと QSO コンテキストを受け取り、発話行為、話題、文字単位の値スロット、レジスター、運用手順の評価、安全な継続状態を出力します。

公開物には PyTorch チェックポイント、FP32 ONNX 参照モデル、デスクトップ CPU 用 INT8 ONNX モデル、およびランタイム契約、マニフェスト、等価性、性能、リリースゲート報告が含まれます。

上流の PulseFormer キーイベント復号器はこのリポジトリに含まれず、独立評価中で、ゲームの正式機能として承認されていません。

## 再現方法

Python 3.11 以降が必要です。

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

評価・エクスポート手順、スキーマ、リリース条件は [`MODEL_ARCHITECTURE.md`](./MODEL_ARCHITECTURE.md) と [`MODEL_CARD.md`](./MODEL_CARD.md) を参照してください。

## 制限事項

報告値は決定論的な合成通信手順領域を測定したものです。任意の会話、未処理のモールス復号、RF ノイズ、伝搬状態、送信タイミングの性能を示すものではありません。これらは別のゲーム層および信号処理層が担当します。

通信文はモールスに適した ASCII 英語略語とローマ字表記の値に限定され、UI の多言語化はゲーム側で処理されます。

## 作者と著作権

**作者：** [Arsenic-er](https://github.com/Arsenic-er)

Copyright © 2026 Arsenic-er. All rights reserved. 追加のライセンスは付与されません。
