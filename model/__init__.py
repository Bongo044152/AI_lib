# file: module/__init__.py
# This package contains various AI Model that define unified interfaces
# for interacting with different AI models.

# A "Model" is represented as a class that exposes a standard interface,
# allowing users to interact with different AI services in a consistent way.

# Available Models:
# - Base_Model.py: Defines an abstract base class for unifying Model interfaces.
# - GPT_Model.py:      Model implementation for GPT models, including:
#       + gpt-4o: https://platform.openai.com/docs/models/gpt-4o
#       + o3: https://platform.openai.com/docs/models/o3
# - DeepSeek_Model.py: Model implementation for DeepSeek models, including:
#       + deepseek-r1: https://docs.api.nvidia.com/nim/reference/deepseek-ai-deepseek-r1
# - Claude_Model.py:   Model implementation for Claude models, including:
#       + Claude Opus 4: https://www.anthropic.com/claude/opus
#       + Claude Sonnet 4: https://www.anthropic.com/claude/sonnet

# since leak understanding on AI, this project will not traning module but using api serverse instead

from .types import ModelIn, ModelOut
from .BaseModel import BaseModel, BaseOption
from .GPTModel import GPTModel, GPTOption
from .HuggingFaceModel import HuggingFaceModel, HuggingFaceOption

# from .DeepSeekModel import DeepSeekModel
from .ClaudeModel import ClaudeModel, ClaudeOption

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
    "ClaudeModel",
    "ClaudeOption",
    "HuggingFaceModel",
    "HuggingFaceOption",
]
