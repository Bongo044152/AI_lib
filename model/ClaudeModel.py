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
import os
import logging
from dotenv import load_dotenv
from .config import env_path
from .BaseModel import BaseController, BaseOption
from typing import Optional, List, Dict, Any, Union

# Load API key from .env
load_dotenv(env_path)
API_KEY = os.getenv('ANTHROPIC_API_KEY')
if not API_KEY:
    raise RuntimeError("ANTHROPIC_API_KEY is required in environment")

# Initialize global client for Anthropic API
client = anthropic.Anthropic(api_key=API_KEY)


class ClaudeOption(BaseOption):
    """
    Options for ClaudeController, defining model parameters and streaming behavior.

    Attributes:
        model (str): Claude model identifier.
        temperature (float): Sampling temperature between 0 and 1.
        max_tokens (int): Maximum tokens allowed in the response.
        stream (bool): Whether to enable streaming via SSE.
        thinking (bool): Whether to use Claude’s internal “thinking” mode.
        thinking_budget_tokens (int): Token budget for “thinking” if enabled.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        temperature: float = 0.6,
        max_tokens: int = 4096,
        stream: bool = False,
        thinking: bool = False,
        thinking_budget_tokens: int = 0
    ):
        # Validate model selection
        if model not in ClaudeOption.get_model_option():
            raise ValueError(f"Unsupported model: {model}")
        # Validate temperature range
        if not (0 <= temperature <= 1):
            raise ValueError("temperature must be between 0 and 1")
        # Validate types
        assert isinstance(stream, bool), "stream must be a bool"
        assert isinstance(max_tokens, int), "max_tokens must be an int"
        # Ensure minimum token limits
        if max_tokens < 1024:
            raise ValueError("max_tokens too small; must be at least 1024")
        assert isinstance(thinking, bool), "thinking must be a bool"
        if thinking:
            # When thinking mode is on, enforce token budget constraints
            assert thinking_budget_tokens >= 1024, (
                "when thinking mode is enabled, thinking_budget_tokens must be ≥ 1024"
            )
            assert thinking_budget_tokens < max_tokens, (
                f"thinking_budget_tokens ({thinking_budget_tokens}) must be < max_tokens ({max_tokens})"
            )

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stream = stream
        self.thinking = thinking
        self.thinking_budget_tokens = thinking_budget_tokens

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
            "stream": self.stream,
            "thinking": "enabled" if self.thinking else "disabled"
        }
        # Include thinking budget only if thinking mode is enabled
        if self.thinking:
            res.update({"thinking_budget_tokens": self.thinking_budget_tokens})
        return res

    @staticmethod
    def get_model_option() -> List[str]:
        """
        Returns supported model names.
        """
        return ["claude-sonnet-4-20250514", "claude-opus-4-20250514"]
    
    def __repr__(
        self
    ) -> str:
        return (
            f"<ClaudeOption(model={self.model}, temperature={self.temperature}, "
            f"max_tokens={self.max_tokens}, stream={self.stream}, thinking={self.thinking})>"
        )


class ClaudeController(BaseController):
    """
    Controller for Anthropic Claude AI inference.

    This class encapsulates the logic to send chat messages to Claude, handle
    streaming vs. non-streaming responses, and manage request timeouts.

    Typical usage:
        opt = ClaudeOption(stream=True)
        controller = ClaudeController(opt=opt)
        messages = [{"role": "user", "content": "Hello!"}]
        response = controller.chat(messages)
    """

    def __init__(
        self,
        timeout: tuple[int, int] = (5, 60),
        opt: Optional[ClaudeOption] = None
    ) -> None:
        # Validate and set options
        assert isinstance(opt, (type(None), ClaudeOption)), "opt must be a ClaudeOption or None"
        assert isinstance()
        self.opt = opt or ClaudeOption()
        # Tuple: (connect_timeout, read_timeout)
        self.timeout = timeout

    def chat(
        self, 
        message: Union[List[Dict[str, Any]], str],
        system: Optional[str] = None
    ) -> str:
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
        # Normalize single dict to list
        if isinstance(message, dict):
            message = [message]
        assert isinstance(message, list), "message must be a list of dicts"

        # Validate each message entry
        for d in message:
            role = d.get("role")
            content = d.get("content")

            if not role:
                raise ValueError("field 'role' is required on each message")
            if role not in ["user", "assistant"]:
                raise ValueError(f"unsupported role: {role}")
            
            if content is None:
                raise ValueError("field 'content' is required on each message")
            if not isinstance(content, (str, dict)):
                raise TypeError("content must be a string or a dict")

        # Perform API call
        response = self._post_message(message, system)

        # Handle streaming vs. non-streaming
        if self.opt.stream:
            return self._parse_stream(response)
        else:
            out = ""
            for block in response.content:
                if block.type == "thinking":
                    out += f"Thinking summary: {block.thinking}\n"
                elif block.type == "text":
                    out += f"Response: {block.text}"
            return out
        
    def _post_message(
        self,
        message: List[Dict[str, Any]],
        system: Optional[str]
    ):
        """
        Internal: send request to Claude messages.create endpoint.
        """
        try:
            # system parameter passed as keyword
            if self.opt.thinking:
                res = client.messages.create(
                    model=self.opt.model,
                    max_tokens=self.opt.max_tokens,
                    temperature=1,  # always set to 1 if thinking mod is on
                    system=system or "",
                    messages=message,
                    stream=self.opt.stream,
                    thinking={
                        "type": "enabled",
                        "budget_tokens": self.opt.thinking_budget_tokens
                    },
                )
            else:
                res = client.messages.create(
                    model=self.opt.model,
                    max_tokens=self.opt.max_tokens,
                    temperature=self.opt.temperature,
                    system=system or "",
                    messages=message,
                    stream=self.opt.stream,
                )

            # Check for API-level error
            if isinstance(res, Message):
                if res.to_dict().get("type") == "error":
                    # logging here if needed
                    raise RuntimeError("Error occurred in Claude API response")

            return res
        except Exception as e:
            # logging here if needed
            raise

    def _parse_stream(
        self,
        stream
    ) -> str:
        """
        Internal: parse streaming response from Claude.

        Only handles:
        - thinking deltas  → print under “Thinking:”
        - content deltas   → print under “Text:”

        References:
        - https://docs.anthropic.com/en/docs/build-with-claude/streaming
        - https://github.com/anthropics/anthropic-sdk-python/blob/main/examples/messages_stream.py
        - https://github.com/anthropics/anthropic-sdk-python/blob/main/helpers.md
        - https://github.com/anthropics/anthropic-sdk-python/blob/main/examples/thinking_stream.py
        - https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/types/raw_message_stream_event.py
        """
        out_text = ""
        thinking_started = False
        text_started = False

        # https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/types/raw_message_stream_event.py
        for event in stream:
            if event.type == "content_block_delta":
                delta = event.delta
                # 有 thinking 情況
                if getattr(delta, "thinking", None) is not None:
                    if not thinking_started:
                        print("\nThinking:\n---------")
                        thinking_started = True
                    # 假設 delta.thinking 裡面是片段字串
                    print(delta.thinking, end="", flush=True)

                # 有文字情況
                if getattr(delta, "text", None):
                    if not text_started:
                        print("\n\nText:\n-----")
                        text_started = True
                    print(delta.text, end="", flush=True)
                    out_text += delta.text
            elif event.type == "message_stop":
                break
            else:
                continue

        return out_text

    def get_option(self) -> ClaudeOption:
        """
        Get the current ClaudeOption.
        """
        return self.opt
    
    def set_option(self, opt: Optional[ClaudeOption] = None) -> None:
        """
        Set a new ClaudeOption. If None, resets to defaults.

        Args:
            opt (Optional[ClaudeOption]): new option instance or None
        """
        self.opt = opt if opt else ClaudeOption()

    def __repr__(self) -> str:
        return f"<ClaudeController(model={self.opt.model}, timeout={self.timeout})>"
