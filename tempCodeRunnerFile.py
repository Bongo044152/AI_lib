from model.DeepSeekModel import DeepSeekModel,DeepSeekOption,ModelIn
option = DeepSeekOption(stream=True)
deepseek = DeepSeekModel(opt=option)
resp = deepseek.chat(ModelIn(content="Hello"))
print(resp)