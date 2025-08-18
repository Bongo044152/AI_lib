from model import ClaudeModel, ClaudeOption, ModelIn

cld_option = ClaudeOption(stream=True, max_tokens=5000)
cld = ClaudeModel(opt=cld_option)

prompt = """
你會打世紀帝國嗎？
你認為哪一個世紀帝國的版本是最棒的？
"""

res = cld.chat(ModelIn(
    content=prompt,
    system_prompt="你是一個專業的說書人,總是喜歡用繁體中文講一些很感人的故事。",
    thinking=True
))
print(res)