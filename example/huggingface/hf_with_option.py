# hf_with_option.py
# 示範如何使用 HuggingFaceOption 來設定溫度、最大生成長度與其他生成參數。

from model import HuggingFaceModel, HuggingFaceOption, ModelIn

# 建立一個自訂的選項物件，用來控制生成行為
hf_opt = HuggingFaceOption(
    temperature=0.9,  # 溫度控制隨機性，越高越有創意
    max_output_tokens=512,  # 最大生成長度（token 數）
    generation_config={  # 傳遞給 transformers.pipeline() 的參數
        "do_sample": True,  # 啟用隨機取樣（避免機械化回答）
        "top_p": 0.95,  # Nucleus sampling 機制
        "repetition_penalty": 1.1,  # 懲罰重複內容
        "no_repeat_ngram_size": 4,  # 限制重複的 4-gram 片段
        "return_full_text": False,  # 僅回傳生成內容，不含原 prompt
    },
)

# 使用自訂選項初始化模型
hf = HuggingFaceModel(model="bigscience/bloom-7b1", opt=hf_opt)

# debug：印出模型資訊
print(repr(hf))

# 可以比較與 hf_common_use.py 的差異
res = hf.chat(ModelIn(content="請用繁體中文打招呼"))

# 顯示生成結果
print(res["output"])
