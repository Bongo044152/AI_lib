from model import ClaudeModel, ModelIn

cld = ClaudeModel()
print(repr(cld)) # debug: see model info

res = cld.chat(ModelIn(content="hello"))
print(res)