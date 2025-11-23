# 🤖 AI Lib

一個輕量級的 AI 模型封裝庫，提供統一的介面來使用不同的 AI 模型服務。

## 📦 專案簡介

本專案透過抽象基類（Abstract Base Class）設計模式，為多個主流 AI 模型提供統一的呼叫介面，讓開發者能夠：

- 🔄 **統一介面**：使用相同的 API 結構呼叫不同的 AI 模型
- 🎯 **簡化整合**：快速切換不同的 AI 服務供應商，無需改動業務邏輯
- 🛠️ **易於擴展**：基於 `BaseModel` 抽象類別，輕鬆新增其他 AI 模型支援
- 💡 **標準化輸入輸出**：統一的 `ModelIn` 和 `ModelOut` 格式，簡化資料處理

### 支援的 AI 模型

- ✅ **GPT** (OpenAI) - 支援 GPT-4o、o3 等模型
- ✅ **Claude** (Anthropic) - 支援 Claude Opus 4、Sonnet 4 等模型
- ✅ **HuggingFace** - 支援 HuggingFace Inference API
- ⚠️ **DeepSeek** - 暫不支援

### 核心功能

- ✅ 統一的聊天介面（`chat` 方法）
- ✅ 支援多輪對話
- ✅ 支援 System Prompt
- ✅ 支援 Thinking 模式（Extended Thinking）
- ✅ 支援串流輸出（Stream Output）
- ✅ 可自訂模型參數（temperature、max_tokens 等）

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
HUGGINGFACE_API_KEY=hf_...
```

---

## ✅ 測試程式碼

使用 [`pytest`](https://docs.pytest.org/) 進行單元測試。

在專案根目錄執行以下指令：

```bash
pytest test/
```