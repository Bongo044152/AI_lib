from model import ClaudeModel, ClaudeOption, ModelIn

cld_option = ClaudeOption(stream=True, max_tokens=1024)
cld = ClaudeModel(opt=cld_option)
print("reasoning model avliable list: ", ClaudeOption.REASONING_MODELS)

res = cld.chat(ModelIn(content="tell me a story about a dragon", system_prompt="你是一個專業的說書人,總是喜歡用繁體中文講一些很感人的故事。"))
print(res["output"])