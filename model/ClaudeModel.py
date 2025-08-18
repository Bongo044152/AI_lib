"""
file: model/Claude_controller.py

Defines a controller for Anthropic Claude AI inference, providing a unified interface
for sending chat messages, handling streaming or non-streaming responses, and managing
controller options.

Example:
    ```python
    from model.Claude_controller import ClaudeController, ClaudeOption

    # Initialize with streaming enabled
    option = ClaudeOption(stream=True)
    controller = ClaudeController(opt=option)
    messages = [
        {"role": "user",      "content": "Hello, how are you?"},
    ]
    response_text = controller.chat(messages)
    print(response_text)
    ```

References:
- Anthropic Claude API (Messages API): https://docs.anthropic.com/en/api/messages
- Claude models overview: https://docs.anthropic.com/en/docs/about-claude/models/overview
- Streaming with Claude: https://docs.anthropic.com/en/docs/build-with-claude/streaming
"""

import anthropic
from anthropic.types.message import Message
from anthropic.types.raw_message_stream_event import RawMessageStreamEvent

import logging, os
from .BaseModel import BaseOption, BaseModel
from typing import *
import pydantic

from dotenv import load_dotenv
from .config import env_path

from .types.ModelLists import Claude_model_types
from .types import ModelIn, ModelOut

# Load API key from .env
load_dotenv(env_path)
API_KEY = os.getenv('ANTHROPIC_API_KEY')
if not API_KEY:
    raise RuntimeError("ANTHROPIC_API_KEY is required in environment")

# Initialize global client for Anthropic API
client = anthropic.Anthropic(api_key=API_KEY)


class ClaudeOption(BaseOption):
    """
    The option used to config Claude model.
    """

    # class attribute
    REASONING_MODELS: ClassVar[tuple[str, ...]] = get_args(Claude_model_types)

    model: Claude_model_types = REASONING_MODELS[0]
    temperature: float = 0.6
    max_tokens: int = 3072
    stream: bool = False

    @pydantic.field_validator("model")
    @classmethod
    def _check_model(cls, model):
        model_list = []
        for i in cls.REASONING_MODELS:
            model_list.append(i)
        
        assert model in model_list, "model not supported"
        return model

    @pydantic.field_validator("temperature")
    def _check_temperature(temperature):
        assert 0 <= temperature <= 1, "temperature must be between 0 and 1"
        return temperature
    
    @pydantic.field_validator("max_tokens")
    def _check_max_tokens(max_tokens):
        assert max_tokens >= 1024, "max_tokens too small; must be at least 1024"
        return max_tokens

    def to_dict(
        self
    ) -> Dict[str, Any]:
        """
        Serialize options to API payload.
        """
        res = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": self.stream
        }
        return res

    def __repr__(
        self
    ) -> str:
        return (
            f"<ClaudeOption(model={self.model}, temperature={self.temperature}, "
            f"max_tokens={self.max_tokens}, stream={self.stream})>"
        )


class ClaudeModel(BaseModel):
    # class attribute
    REASONING_MODELS: ClassVar[tuple[str, ...]] = get_args(Claude_model_types)

    # __init__
    opt: Optional[ClaudeOption] = ClaudeOption()
    timeout: Union[tuple[int, int], int] = (5, 60)
    thinking_param: dict | None = None

    @pydantic.validate_call
    def chat(
        self, 
        message: ModelIn
    ) -> ModelOut:
        """
        Send a chat request and return the AI's response text.

        Args:
            message (List[Dict[str, Any]]): List of messages, each dict must include:
                - "role": either 'user' or 'assistant'
                - "content": the message text
                (see https://docs.anthropic.com/en/api/messages)
            system (Optional[str]): Optional system prompt override.

        Notes:
          - Claude’s Messages API does not support a dedicated 'system' role.
          - If the last message role is 'assistant', the model continues from that content.
          - Streaming is enabled if self.opt.stream is True.

        Returns:
            str: The generated response text.
        """
        if isinstance(message.content, str):
            message.content = [{"role": "user", "content": message.content}]

        if not message.system_prompt:
            message.system_prompt = ""

        if message.thinking:
            assert self.opt.max_tokens // 3 >= 1024, (
                "When thinking is enabled, at least 1024 tokens must be reserved "
                f"(currently only {self.opt.max_tokens // 3}).\n",
                "enable thinking only if max_tokens is large enough (≥ 3072, since 1/3 must be ≥ 1024)."
            )

        if message.thinking:
            self.thinking_param = {
                "budget_tokens": self.opt.max_tokens // 3,
                "type": "enabled"
            }
        else:
            self.thinking_param = {
                "type": "disabled"
            }

        if self.opt.stream:
            return self._post_stream(message)
        else:
            return self._post_message(message)

    @pydantic.validate_call
    def _post_message(
        self,
        message: ModelIn
    ) -> ModelOut:
        """
        Internal: send request to Claude messages.create endpoint.
        """

        try:
            # system parameter passed as keyword
            response = client.messages.create(
                model=self.opt.model,
                max_tokens=self.opt.max_tokens,
                temperature=1 if message.thinking else self.opt.temperature,  # always set to 1 if thinking mod is on
                system=message.system_prompt,
                messages=message.content,
                stream=False,
                thinking=self.thinking_param
            )

            response_dict: dict = response.to_dict()

            # Check for API-level error
            if isinstance(response, Message):
                if response_dict.get("type") == "error":
                    # logging here if needed
                    raise RuntimeError("Error occurred in Claude API response")

            result_output: ModelOut = {
                "model": self.opt.model,
                "output": "",
                "thinking": ""
            }

            for item in response_dict["content"]:
                if item["type"] == "thinking":
                    result_output["thinking"] += item["thinking"]
                elif item["type"] == "text":
                    result_output["output"] += item["text"]
            return result_output
        except Exception as e:
            # logging here if needed
            raise

    def _post_stream(
        self,
        message: ModelIn
    ) -> ModelOut:
        """
        Internal: parse streaming response from Claude.
        """
        result_output: ModelOut = {
            "model": self.opt.model,
            "output": "",
            "thinking": ""
        }

        with client.messages.stream(
            model=self.opt.model,
            max_tokens=self.opt.max_tokens,
            temperature=1 if message.thinking else self.opt.temperature,  # always set to 1 if thinking mod is on
            system=message.system_prompt,
            messages=message.content,
            thinking=self.thinking_param
        ) as stream:
            thinking_started = False

            for event in stream:
                if event.type == "content_block_delta":
                    if event.delta.type == "thinking_delta":
                        if not thinking_started:
                            print("<thinking>", flush=True)
                            thinking_started = True
                        print(event.delta.thinking, end="", flush=True)
                        result_output["thinking"] += event.delta.thinking
                    elif event.delta.type == "text_delta":
                        if thinking_started:
                            print("\n</thinking>", flush=True)
                            thinking_started = False
                        print(event.delta.text, end="", flush=True)
                        result_output["output"] += event.delta.text

        return result_output

    def get_option(self) -> ClaudeOption:
        """
        Get the current ClaudeOption.
        """
        return self.opt
    
    def set_option(self, opt: Optional[ClaudeOption] = None) -> None:
        """
        Set a new ClaudeOption. If None, resets to defaults.

        Args:
            opt (Optional[ClaudeOption]): new option for ClaudeOption
        """
        if opt:
            assert isinstance(opt, ClaudeOption)
        self.opt = opt if opt else ClaudeOption()

    def __repr__(self) -> str:
        return f"<ClaudeModel(model={self.opt.model}, timeout={self.timeout})>"
