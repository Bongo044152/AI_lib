# 🧠 AI Test

本專案旨在透過基礎 API 邏輯構建一個簡易的 AI 應用平台，初期將 AI 模型應用於遊戲環境中（例如《目擊者之夜》），未來可能擴展至《狼人殺》等更具複雜邏輯與推理需求的遊戲場景。

核心目標在於研究**提示詞工程（Prompt Engineering）**對 AI 行為與輸出結果的影響，尤其是在**不更動模型本身參數**的情況下，探索：

1. 多個 AI 模型之間的互動是否能共同解決問題（初步應用為遊戲對戰與推理）
2. AI 是否能對自身輸出進行檢查與反饋，從而提高正確率
3. 在格式要求（如 JSON 輸出）明確的情況下，AI 的正確性與一致性如何？有無改善策略？
4. 更廣泛地探討 AI 開發中「行為預測」、「格式容錯」、「提示詞規劃」等實踐問題

> ⚠️ 本專案目前仍處於早期階段，功能與架構將持續調整與擴充。

---

## 🛠️ Environment Setup

> **Prerequisite:**
> Python **3.12 或以上**版本為必要條件。

---

### 1. 建立虛擬環境

```bash
python -m venv .venv
```

---

### 2. 啟用虛擬環境

#### ▸ Windows（PowerShell）

```powershell
.\.venv\Scripts\Activate.ps1
```

#### ▸ Windows（CMD）

```cmd
.\.venv\Scripts\activate.bat
```

#### ▸ Linux/macOS

```bash
source .venv/bin/activate
```

---

### ✅ 建議操作

啟用環境後安裝依賴：

```bash
pip install -r requirements.txt
```

如需保存目前環境的套件版本清單：

```bash
pip freeze > requirements.txt
```

---

## 🔐 API 金鑰設定

請在專案根目錄下建立 `config/.env` 檔案，並填入對應的 API 金鑰：

```env
OPENAI_API_KEY=sk-proj-OW...
ANTHROPIC_API_KEY=sk-ant-api03-W...
NVIDIA_DEEPSEEK_API_KEY=nvapi-9G...
```

---

## ✅ 測試程式碼

使用 [`pytest`](https://docs.pytest.org/) 進行單元測試。

在專案根目錄執行以下指令：

```bash
pytest test/
```