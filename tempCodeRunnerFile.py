from model.DeepSeekModel import DeepSeekModel,DeepSeekOption,ModelIn
deepseek = DeepSeekModel(opt=DeepSeekOption(stream=True))
resp = deepseek.chat(ModelIn(content="Hello"))
print()
print(resp)