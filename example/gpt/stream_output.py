from model import GPTModel, GPTOption, ModelIn

gpt_option = GPTOption(stream=True, max_output_tokens=1024)
gpt = GPTModel(opt=gpt_option)

res = gpt.chat(ModelIn(content="tell me a story about a dragon"))
print(res)
