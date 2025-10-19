# hf_story_aliens.py
# 完整範例：自訂 prompt 與 generation_config，生成長篇創作內容。

from model import HuggingFaceModel, ModelIn

# 載入 BLOOM-7B 模型（預設使用 4-bit 量化，節省記憶體）
model = HuggingFaceModel(model="bigscience/bloom-7b1")

# 撰寫提示詞（Prompt）
prompt = """
你是一位擅長寫科幻與懸疑短篇小說的作家。
請以 繁體中文 撰寫一個關於 外星人 的神秘短篇故事。
故事要求如下：
- 至少 300 字。
- 分為 三段，每段約 6～8 句。
- 氛圍為「深夜、孤獨、未知、緊張」。
- 在 最後一段 製造一個意想不到的 反轉 結局。
- 不得以「好主意」「讓我想想」「好的」等寒暄開場，要直接從敘事開始。
""".strip()

# 自訂生成參數（會傳入 pipeline）
generation_config = {
    "max_new_tokens": 500,       # 最大生成長度
    "min_new_tokens": 300,       # 最小生成長度（部分模型不支援可省略）
    "do_sample": True,           # 啟用隨機取樣模式
    "temperature": 0.9,          # 溫度控制創意程度
    "top_p": 0.95,               # Nucleus sampling 機制
    "repetition_penalty": 1.1,   # 重複懲罰
    "no_repeat_ngram_size": 4,   # 避免重複片段
    "return_full_text": False,   # 不包含 prompt，只回傳生成文字
}

# 呼叫 chat() 生成文字
res = model.chat(
    message=ModelIn(content=prompt),
    **generation_config
)

# 輸出故事內容
print(res["output"])
