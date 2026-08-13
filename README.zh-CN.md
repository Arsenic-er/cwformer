<div align="center">

# cwformer

### 面向虚构 CW 通联的本地语义理解系统

[**English**](./README.md) · [**简体中文**](./README.zh-CN.md) · [**繁體中文**](./README.zh-TW.md) · [**日本語**](./README.ja.md) · [**Español**](./README.es.md) · [**Deutsch**](./README.de.md) · [**Русский**](./README.ru.md)

**游戏仓库：** [Game-Morse-Adventurer](https://github.com/Arsenic-er/Game-Morse-Adventurer)

</div>

## 项目介绍

本仓库包含 Game-Morse-Adventurer 本地 QSO 语义理解系统的确定性训练数据、生成器、模型代码、评估工具和发布产物。

它不是开放领域聊天模型，而是对已经解码的、适合摩尔斯电码的文本进行分类，让游戏理解通联意图、话题、字段值、规程质量以及当前发射是否可以安全推进通联。

## 数据政策

全部训练、验证和确认数据均由合成 QSO 模板确定性生成。压缩审查副本及 SHA-256 哈希位于 [`datasets/qso-semantics-v4`](./datasets/qso-semantics-v4)，所有固定切分记录在 [`DATASET.md`](./DATASET.md)。

- 不使用玩家数据、遥测、存档、消息或录音。
- 不抓取 QSO 日志或私人通信。
- 游戏运行时不依赖在线服务。
- 合成呼号和身份均为虚构，如有雷同纯属巧合。
- 仓库内的生成器可通过公开的随机种子复现带标签数据。

发布检查点使用 100,000 条合成训练样本训练 16 轮，另使用 15,000 条独立验证样本，以及三组互不重叠、每组 15,000 条的确认样本；确认种子为 `90268741`、`90268742` 和 `90268743`。

## QsoSemanticFormer v0.4

`QsoSemanticFormer` 是一个六层字符级 Transformer，共有 942,898 个参数。它接收解码文本和 QSO 上下文，并输出六组结果：

- 语句行为；
- 通联话题；
- 字符级字段位置；
- 表达类型；
- 规程等级；
- 是否可以安全推进通联。

发布文件包括 PyTorch 检查点、FP32 ONNX 参考模型、桌面 CPU 使用的 INT8 ONNX 模型，以及运行时合同、清单、等价性、性能和发布门槛报告。

上游 PulseFormer 电键事件解码器不包含在本仓库中。它仍在独立评估，尚未获准作为游戏正式功能发布。

## 复现

需要 Python 3.11 或更高版本。

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

评估、导出命令、数据结构及发布门槛见 [`MODEL_ARCHITECTURE.md`](./MODEL_ARCHITECTURE.md) 和 [`MODEL_CARD.md`](./MODEL_CARD.md)。

## 限制

报告结果只衡量确定性的合成通联规程领域，不代表任意对话、原始摩尔斯解码、射频噪声、传播条件或操作员发报时序的性能。这些内容由独立的游戏和信号处理层负责。

空中通联文本限于适合摩尔斯电码的 ASCII 英语缩写和罗马字字段。界面本地化由游戏项目处理。

## 作者与版权

**作者：** [Arsenic-er](https://github.com/Arsenic-er)

Copyright © 2026 Arsenic-er。保留所有权利，未授予任何额外许可。
