"""
file: model/DeepSeek_controller.py

Define a class named DeepSeekController used to control NVIDIA DeepSeek AI behavior.

Example:
    ```python
    from model.DeepSeek_controller import DeepSeekController, DeepSeekOption

    # Initialize with streaming enabled
    option = DeepSeekOption(stream=True)
    controller = DeepSeekController(opt=option)
    messages = [
        {"role": "system",    "content": "You are a helpful AI assistant."},
        {"role": "user",      "content": "Hello, how are you?"}
    ]
    response_text = controller.chat(messages)
    print(response_text)
    ```

As mentioned, we don’t really know how to train AI. The operating principles rely on simple APIs to interact with AI.
Using python's `requests` module to post network requests and finally extract the key information.

We use Server-Sent Events (SSE) to avoid unknown waiting, just like chat-gpt does in their web applications.
- See more about SSE:
    https://blackbing.medium.com/%E6%B7%BA%E8%AB%87-server-sent-events-9c81ef21ca8e5eb654

To enable SSE, simply include `"stream": True` in the HTTP POST body.

In common usage, a chat is not a single request, and requests happen repeatedly. To enhance performance, we use `requests.Session`.
- See requests.Session:
    https://requests.readthedocs.io/en/latest/user/advanced/#session-objects

We also configure timeouts to avoid indefinite waiting: 
    https://requests.readthedocs.io/en/latest/user/advanced/#timeouts

NOTE: NVIDIA's free-tier API cannot handle extremely heavy usage.

References:
- NVIDIA DeepSeek API documentation:
    https://docs.api.nvidia.com/nim/reference/deepseek-ai-deepseek-r1-infer
"""

import os, json
import logging
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

from .BaseModel import BaseController
from .config import env_path

# Load API key from .env
load_dotenv(env_path)
API_KEY = os.getenv('NVIDIA_DEEPSEEK_API_KEY')
if not API_KEY:
    raise RuntimeError("NVIDIA_DEEPSEEK_API_KEY is required in environment")


class DeepSeekOption:
    """
    Options for DeepSeekController.
    """

    def __init__(
        self,
        model: str = "deepseek-ai/deepseek-r1",
        temperature: float = 0.6,
        max_tokens: int = 4096,
        stream: bool = False,
    ) -> None:
        if model not in DeepSeekOption.get_model_option():
            raise ValueError(f"Unsupported model: {model}")
        if not (0 <= temperature <= 1):
            raise ValueError("temperature must be between 0 and 1")
        assert isinstance(stream, bool), "stream must be a bool"
        assert isinstance(max_tokens, int), "max_tokens must be an int"

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stream = stream

    def to_dict(self) -> Dict[str, Any]:
        """Serialize options to API payload."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": self.stream,
        }

    @staticmethod
    def get_model_option() -> List[str]:
        """
        Returns supported model names.
        """
        # see https://docs.api.nvidia.com/nim/reference/deepseek-ai-deepseek-r1-infer
        return ["deepseek-ai/deepseek-r1"]

    def __repr__(self) -> str:
        return (
            f"<DeepSeekOption(model={self.model}, temperature={self.temperature}, "
            f"max_tokens={self.max_tokens}, stream={self.stream})> "
        )


class DeepSeekController(BaseController):
    """
    Controller for NVIDIA DeepSeek AI inference.
    """

    BASE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

    def __init__(
        self,
        timeout: tuple[int, int] = (5, 60),
        opt: Optional[DeepSeekOption] = None
    ) -> None:
        assert isinstance(opt, Optional[DeepSeekOption])
        self.opt = opt or DeepSeekOption()
        self.session = requests.Session()
        self.timeout = timeout

    def chat(
        self,
        message: List[Dict]
    ) -> str:
        """
        Send a chat request and return the AI's response text.

        Args:
            message (List[Dict]): An array of conversation messages. Each dict must include:
                - "role": one of 'system', 'assistant', or 'user'
                - "content": the message text

            Roles:
                * system: system-level instructions
                * assistant: previous AI responses
                * user: user inputs (must be the last message)

        Example:
            messages = [
                {"role": "system",    "content": "You are a helpful assistant."},
                {"role": "user",      "content": "Hi, how are you?"},
                {"role": "assistant", "content": "I'm doing well—how can I help?"},
                {"role": "user",      "content": "Tell me a joke."}
            ]

        NOTE: The final message's role must be 'user'.
        
        learn more: https://docs.api.nvidia.com/nim/reference/deepseek-ai-deepseek-r1-infer

        Returns:
            str: The AI-generated response content.
        """
        # Validate message
        if not isinstance(message, list):
            raise TypeError("message must be a list of dicts, see NVIDIA API docs")
        if len(message) == 0:
            raise RuntimeError("message list must not be empty")
        if message[-1].get("role") != "user":
            raise RuntimeError("Last message role must be 'user', see NVIDIA API docs: \n\t\
                               https://docs.api.nvidia.com/nim/reference/deepseek-ai-deepseek-r1-infer")

        payload = {
            **self.opt.to_dict(),
            "top_p": 0.7,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "messages": message
        }

        try:
            resp = self._post_message(payload)
            if self.opt.stream:
                return self._parse_stream(resp)
            return resp.json()["choices"][0]["message"]["content"]

        except requests.ReadTimeout:
            # logging here if needed
            raise
        except requests.RequestException as e:
            # logging here if needed
            raise

    def _post_message(
        self,
        payload: Dict[str, Any]
    ) -> requests.Response:
        """
        Internal: Perform HTTP POST with headers, using custom timeouts.
        """
        headers = {
            "Accept": "text/event-stream" if self.opt.stream else "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        }
        resp = self.session.post(
            DeepSeekController.BASE_URL,
            headers=headers,
            json=payload,
            timeout=self.timeout,
            stream=self.opt.stream
        )

        # if the URL is invalid or returns a 4xx/5xx status code, it raises an HTTPError.
        # see more: https://www.geeksforgeeks.org/python/response-raise_for_status-python-requests/
        resp.raise_for_status()
        return resp

    def _parse_stream(self, response: requests.Response) -> str:
        """
        Parse Server-Sent Events (SSE) from a streaming response.
        """
        out_text: List[str] = []
        for raw_line in response.iter_lines(decode_unicode=True):

            assert isinstance(raw_line, str)

            if isinstance(raw_line, (bytes, bytearray)):
                try:
                    line = raw_line.decode('utf-8').strip()
                except UnicodeDecodeError:
                    # logging here if needed
                    raise # or pass
            else:
                line = str(raw_line).strip()


            if not line:
                continue
            elif line == "data: [DONE]":
                break
            elif not line.startswith("data: "):
                raise RuntimeError("no prefix: \"data:\"")

            try:
                chunk = json.loads(line[len("data: "):])
                delta = chunk["choices"][0]["delta"]
                if content := delta.get("content"):
                    out_text.append(content)
                    print(content, end="")
            except json.JSONDecodeError:
                # logging here if needed
                raise # or pass

        return "".join(out_text)
    
    def get_option(self) -> DeepSeekOption:
        """
        Get the current DeepSeekOption.
        """
        return self.opt
    
    def set_option(self, opt: Optional[DeepSeekOption] = None) -> None:
        """
        Set a new DeepSeekOption. If None, resets to defaults.

        Args:
            opt (Optional[DeepSeekOption]): new option instance or None
        """
        self.opt = opt if opt else DeepSeekOption()

    def __repr__(self) -> str:
        return f"<DeepSeekController(model={self.opt.model}, timeout={self.timeout})>"