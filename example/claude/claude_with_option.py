from model import ClaudeModel, ClaudeOption, ModelIn

cld_option = ClaudeOption(
    model="claude-opus-4-20250514", temperature=0.8, max_tokens=1024
)
cld = ClaudeModel(opt=cld_option)
print("reasoning model avliable list: ", ClaudeOption.REASONING_MODELS)

res = cld.chat(ModelIn(content="tell me a story about a dragon"))
print(res["output"])
