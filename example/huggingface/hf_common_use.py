# hf_common_use.py
# 最簡範例：展示如何初始化 Hugging Face 模型、顯示基本資訊並執行單句輸入生成。

from model import HuggingFaceModel, ModelIn

# 初始化 Hugging Face 模型（此處使用 7B 參數的 BLOOM）
hf = HuggingFaceModel(model="bigscience/bloom-7b1")

# debug：顯示模型基本資訊（會呼叫 __repr__）
print(repr(hf))

# 給模型一個最簡單的輸入
res = hf.chat(ModelIn(content="請用繁體中文打招呼"))

# 只輸出模型生成的文字內容
print(res["output"])
