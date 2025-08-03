from model import GPTModel, GPTOption, ModelIn

gpt_option = GPTOption(model="o3-mini", temperature=1.6, max_output_tokens=1024)
gpt = GPTModel(opt=gpt_option)

print("common model avliable list: ", GPTOption.COMMON_MODELS)
print("reasoning model avliable list: ", GPTOption.REASONING_MODELS)

res = gpt.chat(ModelIn(content="tell me a story about a dragon"))
print(res["output"])