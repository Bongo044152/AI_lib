# file: test/test_gpt.py
# using pytest to run the test

import json
import pydantic
from model.GPTModel import GPTModel, GPTOption, ModelIn


def _check_field(res: dict):
    assert res.get("model") is not None
    assert res.get("thinking") is not None
    assert res.get("output") is not None


def test_link():
    print()
    gpt = GPTModel()
    res = gpt.chat(ModelIn(content="隨意輸出一些訊息，長度不為0，以測試連線"))

    _check_field(res)

    assert len(res["output"])

    print(res)


def test_stream():
    print()
    gpt_options = GPTOption(stream=True)
    gpt = GPTModel(opt=gpt_options)
    res = gpt.chat(ModelIn(content="隨意輸出一些訊息，長度不為0，以測試連線"))

    assert len(res["output"])

    print(res)


def test_thinking_invaild():
    print()
    gpt = GPTModel(opt=GPTOption(model="gpt-4o"))
    try:
        gpt.chat(
            ModelIn(content="隨意輸出一些訊息，長度不為0，以測試連線", thinking=True)
        )
        assert False, "should be failed if this line"
    except ValueError:
        pass
    except:
        raise

    gpt.set_option(GPTOption(model="o3"))
    model_input = ModelIn(
        content="描述外星人的傳說故事（可自創，但聽起來要足夠真實)", thinking=True
    )
    res = gpt.chat(model_input)

    _check_field(res)

    assert len(res["output"])

    print(res)


def test_thinking_stream():
    print()

    gpt = GPTModel(opt=GPTOption(model="o3", stream=True))
    model_input = ModelIn(content="講一個有深度的故事，和感情有關", thinking=True)
    res = gpt.chat(model_input)

    _check_field(res)

    assert len(res["output"])

    print(res)


def test_history():
    print()
    option = GPTOption(model="gpt-4o", stream=True)
    gpt = GPTModel(opt=option)

    output = gpt.chat(ModelIn(content="Hello, how are you?"))

    _check_field(output)

    print(output)

    messages = [
        {"role": "user", "content": "嗨～你在嗎？"},
        {"role": "assistant", "content": "我一直都在呀！有什麼想聊的？😄"},
        {"role": "user", "content": "今天天氣好像不錯耶～"},
        {
            "role": "assistant",
            "content": "對啊，陽光普照、心情加倍☀️ 要不要一起來點腦力活動？",
        },
        {"role": "user", "content": "嗯... 給我一個腦筋急轉彎！"},
        {
            "role": "assistant",
            "content": "好嘞！問題來囉～「小明的媽媽有三個小孩，分別叫做一月、二月，請問第三個叫什麼？」🤔",
        },
        {"role": "user", "content": "咦…是三月嗎？"},
        {
            "role": "assistant",
            "content": "答錯啦～是小明啦！題目不是說『小明的媽媽』嗎？😆",
        },
        {"role": "user", "content": "哈哈我上當了 XDD"},
        {"role": "user", "content": "那你可以告訴我我們剛剛聊了什麼嗎？"},
    ]

    history = gpt.chat(ModelIn(content=messages))

    _check_field(output)

    print(history["output"])


def test_wrong_format():
    print()
    option = GPTOption(model="gpt-4o", stream=True)
    gpt = GPTModel(opt=option)

    # test failed
    try:
        messages = [{"role": "unknown", "content": ""}]
        messages = ModelIn(content=messages)
        gpt.chat(messages)
        assert False
    except ValueError:
        pass
    except:
        raise

    try:
        messages = [
            {
                "role": "system",
                "content": "你是一個反應快速又有趣的 AI 助理，說話風格輕鬆活潑。",
            }
        ]
        messages = ModelIn(content=messages)
        gpt.chat(messages)
        assert False
    except ValueError:
        pass
    except:
        raise

    try:
        messages = [{"role": "assistant", "content": ""}]
        messages = ModelIn(content=messages)
        gpt.chat(messages)
        assert False
    except pydantic.ValidationError:
        pass
    except AttributeError:
        pass
    except:
        raise

    try:
        messages = [{"role": "user"}]
        messages = ModelIn(content=messages)
        gpt.chat(messages)
        assert False
    except:
        pass

    try:
        messages = [{"role": "user", "content": 123}]
        messages = ModelIn(content=messages)
        gpt.chat(messages)
        assert False
    except:
        pass

    try:
        messages = {"role": "user", "content": 123}
        messages = ModelIn(content=messages)
        gpt.chat(messages)
        assert False
    except:
        pass


def test_system_prompt():
    print()
    gpt = GPTModel(opt=GPTOption(model="gpt-4o"))
    res = gpt.chat(
        ModelIn(
            content="who are you?",
            system_prompt="不管使用者說什麼，你都回答「很遺憾我沒有辦法幫助到你」，注意！要一字不差！（不要額外添加標點符號）",
        )
    )["output"]

    assert res == "很遺憾我沒有辦法幫助到你"

    print(res)


def test_repr():
    print()
    model_list = GPTOption.COMMON_MODELS
    gpt_options = GPTOption(stream=True, model=model_list[-1])
    gpt = GPTModel(opt=gpt_options)

    assert gpt.opt.stream is True
    assert gpt.opt.model == model_list[-1]

    assert isinstance(repr(gpt), str)
    print(gpt)

    opt = gpt.get_option()

    assert isinstance(repr(opt), str)
    print(opt)
    print(opt.to_dict())
