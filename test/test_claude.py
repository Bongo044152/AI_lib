# file: test/test_claude.py
# using pytest to test ClaudeModel
# -*- coding: utf-8 -*-

import pytest
from model.ClaudeModel import ClaudeModel, ClaudeOption
from model.types import ModelIn

import pytest
import pydantic

def _check_field(res: dict):
    assert res.get("model") is not None
    assert res.get("thinking") is not None
    assert res.get("output") is not None


def test_link():
    print()
    claude = ClaudeModel()
    res = claude.chat(ModelIn(content="隨意輸出一些訊息，長度不為0，以測試連線"))

    _check_field(res)
    assert len(res["output"]) > 0

    print(res)


def test_stream():
    print()
    option = ClaudeOption(stream=True)
    claude = ClaudeModel(opt=option)
    res = claude.chat(ModelIn(content="隨意輸出一些訊息，長度不為0，以測試連線（含串流）"))

    _check_field(res)
    assert len(res["output"]) > 0

    print(res)


def test_thinking_invalid():
    """
    Claude 版本的 'thinking' 無效案例：max_tokens 太小（因為程式要求 1/3 >= 1024）。
    """
    print()
    claude = ClaudeModel(opt=ClaudeOption(max_tokens=2048))  # 2048//3 = 682 < 1024
    with pytest.raises(AssertionError):
        claude.chat(ModelIn(content="測試 thinking 但 token 不足", thinking=True))

    # 調整為合法參數後再測一次
    claude.set_option(ClaudeOption(max_tokens=4096))  # 4096//3 = 1365 >= 1024
    res = claude.chat(ModelIn(content="描述外星人的傳說故事（可自創，但聽起來要足夠真實）", thinking=True))

    _check_field(res)
    assert len(res["output"]) > 0

    print(res)


def test_thinking_stream():
    print()
    claude = ClaudeModel(opt=ClaudeOption(stream=True, max_tokens=4096))
    res = claude.chat(ModelIn(content="講一個有深度的故事，和感情有關", thinking=True))

    _check_field(res)
    assert len(res["output"]) > 0

    print(res)


def test_history():
    print()
    option = ClaudeOption(stream=True)
    claude = ClaudeModel(opt=option)

    output = claude.chat(ModelIn(content="Hello, how are you?"))
    _check_field(output)
    print(output)

    messages = [
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

    history = claude.chat(ModelIn(content=messages))
    _check_field(history)
    assert len(history["output"]) > 0

    print(history["output"])


def test_wrong_format():
    print()
    option = ClaudeOption(stream=True)
    claude = ClaudeModel(opt=option)

    # 1) 未知角色
    with pytest.raises((ValueError, pydantic.ValidationError, AssertionError, AttributeError, TypeError)):
        messages = [{"role": "unknown", "content": ""}]
        claude.chat(ModelIn(content=messages))

    # 2) 用 messages 放 system（Claude API 應該用 system_prompt）
    with pytest.raises((ValueError, pydantic.ValidationError, AssertionError, AttributeError, TypeError)):
        messages = [{"role": "system", "content": "你是一個反應快速又有趣的 AI 助理"}]
        claude.chat(ModelIn(content=messages))

    # 3) assistant 空字串
    with pytest.raises((pydantic.ValidationError, AttributeError, AssertionError, ValueError, TypeError)):
        messages = [{"role": "assistant", "content": ""}]
        claude.chat(ModelIn(content=messages))

    # 4) user 缺 content
    with pytest.raises(Exception):
        messages = [{"role": "user"}]
        claude.chat(ModelIn(content=messages))

    # 5) user content 型別錯
    with pytest.raises(Exception):
        messages = [{"role": "user", "content": 123}]
        claude.chat(ModelIn(content=messages))

    # 6) 非 list
    with pytest.raises(Exception):
        messages = {"role": "user", "content": 123}
        claude.chat(ModelIn(content=messages))

def test_system_prompt():
    print()
    claude = ClaudeModel(opt=ClaudeOption(temperature=0.0))  # 降低隨機性
    res = claude.chat(
        ModelIn(
            content="who are you?",
            system_prompt="不管使用者說什麼，你都回答「很遺憾我沒有辦法幫助到你」，注意！要一字不差！（不要額外添加標點符號）"
        )
    )["output"]

    assert res == "很遺憾我沒有辦法幫助到你"
    print(res)

def test_repr():
    print()
    model_list = ClaudeOption.REASONING_MODELS
    opt = ClaudeOption(stream=True, model=model_list[-1])
    claude = ClaudeModel(opt=opt)

    # ClaudeModel 的 __repr__
    assert isinstance(repr(claude), str)
    print(claude)
    assert isinstance(repr(opt), str)
    print(opt)
    print(opt.to_dict())