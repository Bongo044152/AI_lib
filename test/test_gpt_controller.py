# file: test/test_gpt_controller.py
# using pytest to run the test

import json
from model.GPT_controller import GPTController, GPTOption

def _check_field(res):
    assert res.get("id")
    assert res.get("content")
    assert res.get("role")
    assert res.get("status")
    assert res.get("type")

def test_link():
    print()
    gpt = GPTController()
    res = gpt.chat("隨意輸出一些訊息，長度不為0，以測試連線")

    _check_field(res)
    
    print(json.dumps(res, indent=2))

def test_stream():
    print()
    gpt_options = GPTOption(stream=True)
    gpt = GPTController(gpt_options)
    res = gpt.chat("隨意輸出一些訊息，長度不為0，以測試連線")

    _check_field(res)

    print(json.dumps(res, indent=2))

def test_history():
    print()
    option = GPTOption(model="gpt-4o", stream=True)
    controller = GPTController(opt=option)

    output = controller.chat("Hello, how are you?")

    _check_field(output)

    print(output)

    messages = [
        {"role": "system", "content": "你是一個反應快速又有趣的 AI 助理，說話風格輕鬆活潑。"},
        {"role": "user", "content": "嗨～你在嗎？"},
        {"role": "assistant", "content": "我一直都在呀！有什麼想聊的？😄"},
        {"role": "user", "content": "今天天氣好像不錯耶～"},
        {"role": "assistant", "content": "對啊，陽光普照、心情加倍☀️ 要不要一起來點腦力活動？"},
        {"role": "user", "content": "嗯... 給我一個腦筋急轉彎！"},
        {"role": "assistant", "content": "好嘞！問題來囉～「小明的媽媽有三個小孩，分別叫做一月、二月，請問第三個叫什麼？」🤔"},
        {"role": "user", "content": "咦…是三月嗎？"},
        {"role": "assistant", "content": "答錯啦～是小明啦！題目不是說『小明的媽媽』嗎？😆"},
        {"role": "user", "content": "哈哈我上當了 XDD"},
        {"role": "user", "content": "那你可以告訴我我們剛剛聊了什麼嗎？"}
    ]

    history = controller.chat(messages)

    _check_field(output)

    print(history)

def test_repr():
    print()
    model_list = GPTOption.get_model_option()
    gpt_options = GPTOption(stream=True, model=model_list[-1])
    gpt = GPTController(gpt_options)

    assert gpt.opt.stream is True
    assert gpt.opt.model == model_list[-1]

    assert isinstance(repr(gpt), str)
    print(gpt)

    opt = gpt.get_option()

    assert isinstance(repr(opt), str)
    print(opt)
    print(opt.to_dict())