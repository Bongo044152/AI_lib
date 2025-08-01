# file: module/__init__.py
# This package contains various AI controllers that define unified interfaces
# for interacting with different AI models.

# A "controller" is represented as a class that exposes a standard interface,
# allowing users to interact with different AI services in a consistent way.

# Available controllers:
# - Base_controller.py: Defines an abstract base class for unifying controller interfaces.
# - GPT_controller.py:      Controller implementation for GPT models, including: 
#       + gpt-4o: https://platform.openai.com/docs/models/gpt-4o
#       + o3: https://platform.openai.com/docs/models/o3
# - DeepSeek_controller.py: Controller implementation for DeepSeek models, including: 
#       + deepseek-r1: https://docs.api.nvidia.com/nim/reference/deepseek-ai-deepseek-r1
# - Claude_controller.py:   Controller implementation for Claude models, including: 
#       + Claude Opus 4: https://www.anthropic.com/claude/opus
#       + Claude Sonnet 4: https://www.anthropic.com/claude/sonnet

# since leak understanding on AI, this project will not traning module but using api serverse instead

from .BaseModel import BaseModel
from .GPTModel import GPTModel
# from .DeepSeekModel import DeepSeekModel
# from .ClaudeModel import ClaudeModel

# usage of __all__:
#   https://docs.python.org/zh-tw/3.13/tutorial/modules.html#importing-from-a-package
__all__ = [
    "BaseModel",
    "GPTModel",
    # "DeepSeekController",
    # "ClaudeController",
]



# model/
# ├── base.py                ← 你自己的 BaseModel
# ├── types/                 ← 共用型別定義
# │   ├── __init__.py
# │   ├── core.py            ← Message, Role, TokenUsage 等
# │   ├── gpt.py             ← ModelName for GPT
# │   ├── gemini.py          ← Gemini 專屬型別（如 model name、角色）
# │   └── deepseek.py        ← DeepSeek 型別
# ├── gpt/                   ← GPT 模型 schema
# │   ├── input.py
# │   ├── output.py
# │   └── __init__.py
# ├── gemini/                ← Gemini 模型 schema
# │   ├── input.py
# │   ├── output.py
# │   └── __init__.py
# ├── deepseek/              ← DeepSeek 模型 schema
# │   ├── input.py
# │   ├── output.py
# │   └── __init__.py