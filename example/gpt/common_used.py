from model import GPTModel, ModelIn

gpt = GPTModel()
print(repr(gpt))  # debug: see model info

res = gpt.chat(ModelIn(content="hello"))
print(res)
