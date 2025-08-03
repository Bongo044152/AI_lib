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

from .types import ModelIn, ModelOut
from .BaseModel import BaseModel, BaseOption
from .GPTModel import GPTModel, GPTOption
# from .DeepSeekModel import DeepSeekModel
# from .ClaudeModel import ClaudeModel

# usage of __all__:
#   https://docs.python.org/zh-tw/3.13/tutorial/modules.html#importing-from-a-package
__all__ = [
    "ModelIn",
    "ModelOut",
    "BaseModel",
    "BaseOption",
    "GPTModel",
    "GPTOption"
    # "DeepSeekController",
    # "ClaudeController",
]
