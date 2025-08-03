# file: test/test_deepseek_controller.py
# using pytest to test DeepSeekController

import pytest
from model.DeepSeekModel import DeepSeekController, DeepSeekOption


def test_link():
    print()
    # Non-streaming mode
    option = DeepSeekOption(stream=False)
    ds = DeepSeekController(opt=option)

    # Minimal valid message list
    messages = [
        {"role": "user", "content": "環境檢查：請回傳一段測試文字:\"Hello DeepSeek\" (回傳的每個字都要完全相同,作為行為測試的準則之一；禁止添加任何其他訊息)"},
    ]
    res = ds.chat(messages)

    assert isinstance(res, str)
    assert len(res) > 0
    assert res != "Hello DeepSeek"

    print(res)

def test_stream():
    print()
    # Streaming mode
    option = DeepSeekOption(stream=True)
    ds = DeepSeekController(opt=option)

    messages = [
        {"role": "user", "content": "環境檢查：請依序回傳測試文字:\"Hello DeepSeek\"(回傳的每個字都要完全相同,作為行為測試的準則之一；禁止添加任何其他訊息)"},
    ]
    res = ds.chat(messages)

    assert isinstance(res, str)
    assert len(res) > 0

    print(res)

def test_check_safe1():
    try:
        ds = DeepSeekController()
        messages = [
            {"role": "?", "content": ""},
        ]
        ds.chat(messages)
    except:
        print("check_safe1 passed !")
        return
    
    assert False

def test_check_safe2():
    try:
        ds = DeepSeekController()
        messages = "hello?"
        ds.chat(messages)
    except:
        print("check_safe2 passed !")
        return
    
    assert False

def test_repr():
    print()
    models = DeepSeekOption.get_model_option()
    option = DeepSeekOption(stream=True, model=models[-1])
    ds = DeepSeekController(opt=option)

    assert ds.opt.stream is True
    assert ds.opt.model == models[-1]

    assert isinstance(repr(ds), str)
    print(ds)

    opt = ds.get_option()

    assert isinstance(repr(opt), str)
    print(opt)
    print(opt.to_dict())