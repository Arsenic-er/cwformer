<div align="center">

# cwformer

### 面向虛構 CW 通聯的本地語意理解系統

[**English**](./README.md) · [**简体中文**](./README.zh-CN.md) · [**繁體中文**](./README.zh-TW.md) · [**日本語**](./README.ja.md) · [**Español**](./README.es.md) · [**Deutsch**](./README.de.md) · [**Русский**](./README.ru.md)

**遊戲儲存庫：** [Game-Morse-Adventurer](https://github.com/Arsenic-er/Game-Morse-Adventurer)

</div>

## 專案介紹

本儲存庫包含 Game-Morse-Adventurer 本地 QSO 語意理解系統的確定性訓練資料、產生器、模型程式碼、評估工具及發布產物。

它不是開放領域聊天模型，而是對已解碼且適合摩爾斯電碼的文字進行分類，讓遊戲理解通聯意圖、話題、欄位值、規程品質，以及目前發射是否能安全推進通聯。

## 資料政策

所有訓練、驗證與確認資料均由合成 QSO 範本確定性產生。壓縮審查副本及 SHA-256 雜湊位於 [`datasets/qso-semantics-v4`](./datasets/qso-semantics-v4)，所有固定切分記錄於 [`DATASET.md`](./DATASET.md)。

- 不使用玩家資料、遙測、存檔、訊息或錄音。
- 不擷取 QSO 日誌或私人通訊。
- 遊戲執行時不依賴線上服務。
- 合成呼號與身分均為虛構，如有雷同純屬巧合。
- 儲存庫內的產生器可透過公開的隨機種子重現帶標籤資料。

發布檢查點使用 100,000 筆合成訓練樣本訓練 16 輪，另使用 15,000 筆獨立驗證樣本，以及三組互不重疊、每組 15,000 筆的確認樣本；確認種子為 `90268741`、`90268742` 與 `90268743`。

## QsoSemanticFormer v0.4

`QsoSemanticFormer` 是六層字元級 Transformer，共有 942,898 個參數。它接收解碼文字與 QSO 上下文，並輸出六組結果：語句行為、通聯話題、字元級欄位位置、表達類型、規程等級，以及是否可以安全推進通聯。

發布檔案包括 PyTorch 檢查點、FP32 ONNX 參考模型、桌面 CPU 使用的 INT8 ONNX 模型，以及執行期合約、清單、等價性、效能和發布門檻報告。

上游 PulseFormer 電鍵事件解碼器不包含於本儲存庫。它仍在獨立評估，尚未獲准作為遊戲正式功能發布。

## 重現

需要 Python 3.11 或更新版本。

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

評估、匯出命令、資料結構與發布門檻請見 [`MODEL_ARCHITECTURE.md`](./MODEL_ARCHITECTURE.md) 和 [`MODEL_CARD.md`](./MODEL_CARD.md)。

## 限制

報告結果僅衡量確定性的合成通聯規程領域，不代表任意對話、原始摩爾斯解碼、射頻雜訊、傳播條件或操作員發報時序的效能。這些內容由獨立的遊戲及訊號處理層負責。

空中通聯文字限於適合摩爾斯電碼的 ASCII 英語縮寫與羅馬字欄位。介面本地化由遊戲專案處理。

## 作者與版權

**作者：** [Arsenic-er](https://github.com/Arsenic-er)

Copyright © 2026 Arsenic-er。保留所有權利，未授予任何額外授權。
