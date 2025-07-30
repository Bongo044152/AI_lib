# file: test/test_claude_controller.py
# using pytest to test ClaudeController

import pytest
from model.Claude_controller import ClaudeController, ClaudeOption


def test_link():
    print()
    # Non-streaming mode
    option = ClaudeOption(stream=False)
    controller = ClaudeController(opt=option)

    # Minimal valid message list
    messages = [
        {"role": "user", "content": "環境檢查：請回傳一段測試文字:\"Hello Claude\" (回傳的每個字都要完全相同,作為行為測試的準則之一；禁止添加任何其他訊息)"}
    ]
    res = controller.chat(messages)

    assert isinstance(res, str)
    assert len(res) > 0
    print(res)

    assert res == "Response: Hello Claude"


def test_stream():
    print()
    # Streaming mode
    option = ClaudeOption(stream=True)
    controller = ClaudeController(opt=option)

    messages = [
        {"role": "user", "content": "環境檢查：請依序回傳測試文字:\"Hello Claude\"(回傳的每個字都要完全相同,作為行為測試的準則之一；禁止添加任何其他訊息)"}
    ]
    res = controller.chat(messages)

    assert isinstance(res, str)
    assert len(res) > 0
    assert res == "Hello Claude"


def test_think():
    print()
    # Streaming mode
    try:
        option = ClaudeOption(thinking=True)
        assert False, "will cause AssertionError if not provied thinking_budget_tokens"
    except AssertionError:
        pass
    thinking_token = 2**10
    option = ClaudeOption(
        max_tokens=thinking_token*2,
        thinking=True,
        thinking_budget_tokens=thinking_token
    )
    controller = ClaudeController(opt=option)
    res = controller.chat(message = {
        "role": "user",
        "content": (
            "使用教師的口吻，說明為什麼 1+1 = 2"
        )
    }, system="reply in Chinses")
    print(res)


def test_think_stream():
    print()
    # Streaming mode
    try:
        option = ClaudeOption(thinking=True, stream=True)
        assert False, "will cause AssertionError if not provied thinking_budget_tokens"
    except AssertionError:
        pass
    thinking_token = 2**10
    option = ClaudeOption(
        max_tokens=thinking_token*2,
        thinking=True,
        thinking_budget_tokens=thinking_token,
        stream=True
    )
    controller = ClaudeController(opt=option)
    res = controller.chat(message = {
        "role": "user",
        "content": (
            "使用教師的口吻，說明為什麼 1+1 = 2"
        )
    }, system="reply in Chinses")
    print(res)


def test_check_safe1():
    try:
        controller = ClaudeController()
        messages = [
            {"role": "invalid", "content": ""}
        ]
        controller.chat(messages)
    except:
        print("check_safe1 passed !")
        return

    assert False


def test_check_safe2():
    try:
        controller = ClaudeController()
        messages = "hello?"
        controller.chat(messages)
    except:
        print("check_safe2 passed !")
        return

    assert False


def test_repr():
    print()
    models = ClaudeOption.get_model_option()
    option = ClaudeOption(stream=True, model=models[-1])
    controller = ClaudeController(opt=option)

    assert controller.opt.stream is True
    assert controller.opt.model == models[-1]

    assert isinstance(repr(controller), str)
    print(controller)

    opt = controller.get_option()

    assert isinstance(repr(opt), str)
    print(opt)
    print(opt.to_dict())