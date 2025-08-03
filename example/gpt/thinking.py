from model import GPTModel, GPTOption, ModelIn

gpt_option = GPTOption(model="o3-mini", stream=True)
gpt = GPTModel(opt=gpt_option)

prompt = """
你會打世紀帝國嗎？
你認為哪一個世紀帝國的版本是最棒的？
"""

res = gpt.chat(ModelIn(content=prompt, thinking=True))
# print(res)