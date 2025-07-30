"""
file: module/GPT_controller.py

Controller for OpenAI Responses API (gpt-4o, o3, etc.).

Example:
    ```python
    from model.GPT_controller import GPTController, GPTOption

    # Initialize with default options
    option = GPTOption(model="gpt-4o", stream=True)
    controller = GPTController(opt=option)

    # Single string input
    output = controller.chat("Hello, how are you?")
    print(output)

    # Structured messages
    messages = [
        {"role": "system",    "content": "You are a helpful assistant."},
        {"role": "user",      "content": "Tell me a joke."}
    ]
    history = controller.chat(messages)
    print(history)
    ```

This module exposes GPTOption for configuration and GPTController to send
requests via the OpenAI Responses API. Supports both streaming and non-streaming
modes. Use streaming to receive incremental deltas and record the final message
for history tracking.

See:
    https://platform.openai.com/docs/api-reference/responses
"""

# https://github.com/openai/openai-python?tab=readme-ov-file
from openai import OpenAI, APITimeoutError
from openai.types.responses import Response as openai_Response

import logging, os
from .Base_controller import BaseController
from typing import Dict, Any, List, Union, Optional

# using .env file to store api key
from dotenv import load_dotenv
from .config import env_path
load_dotenv(env_path)

API_KEY = os.environ.get("OPENAI_API_KEY")

if not API_KEY:
    # logging if needed
    raise RuntimeError("OPENAI_API_KEY is required in environment")

client = OpenAI(
    api_key=API_KEY
)

class GPTOption:
    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: Union[int, float] = 1,
        max_output_tokens: int = 4096,
        stream: bool = False,
    ) -> None:
        if model not in GPTOption.get_model_option():
            raise ValueError(f"Unsupported model: {model}")
        if not (0 <= temperature <= 2):
            raise ValueError("temperature must be between 0 and 2")
        assert isinstance(stream, bool)
        assert isinstance(max_output_tokens, int)

        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.stream = stream

    def to_dict(self) -> Dict[str, Any]:
        """Serialize options to the API payload."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "stream": self.stream,
        }

    @staticmethod
    def get_model_option() -> List[str]:
        """
        Return supported model names.

        Returns:
            list[str]: available model identifiers, currently: ["gpt-4o", "o3"]
        """
        # see https://platform.openai.com/docs/models
        return ["gpt-4o", "o3"]

    def __repr__(self) -> str:
        return (
            f"<GPTOption("
            f"model={self.model}, "
            f"temperature={self.temperature}, "
            f"max_output_tokens={self.max_output_tokens}, "
            f"stream={self.stream})>"
        )

class GPTController(BaseController):
    """
    Controller for OpenAI Responses API.
    """

    def __init__(
        self, 
        opt: Optional[GPTOption] = None,
        timeout: int = 60
        ):
        assert isinstance(opt, Optional[GPTOption])
        assert isinstance(timeout, int)
        self.opt = opt or GPTOption()
        self.timeout = timeout

    def _post_message(
        self,
        message: List[Dict],
        timeout: int
    ) -> openai_Response:
        """
        Internal: make the HTTP POST.
        the argument message is provided from `chat` function
        """
        try:
            response: openai_Response = client.with_options(timeout=timeout).responses.create(
                model=self.opt.model,
                input=message,
                temperature=self.opt.temperature,
                stream=self.opt.stream,
            )
        except APITimeoutError as e:
            # https://github.com/openai/openai-python?tab=readme-ov-file#timeouts
            # logging if needed
            raise
        except Exception as e:
            # logging if needed
            raise

        # note: not real openai_Response if stream is true
        # see: https://github.com/openai/openai-python/blob/main/src/openai/resources/responses/responses.py
        return response
    
    def _prase_stream(
        self,
        response
    ) -> Dict[str, Any]:
        """
        Internal: output the recorded message for history track 
        and also print the message AI generate in the stream

        Reference:
            - https://platform.openai.com/docs/guides/streaming-responses?api-mode=responses
        """
        recorded = {
            "id": None,
            "content": [],
            "role": "assistant",
            "status": "in_progress",
            "type": "message"
        }
        text = ""

        for event in response:
            # https://github.com/openai/openai-python/blob/main/src/openai/types/responses/response_created_event.py
            event = event.to_dict()

            t = event["type"]
            d = event.get("data") or event.get("item")

            if t == "response.output_item.added":
                recorded["id"] = d["id"]

            elif t == "response.output_text.delta":
                delta = event.get("delta")
                print(delta, end="")
                text += delta

            elif t == "response.output_text.done":
                recorded["content"].append({
                    "type": "output_text",
                    "text": text,
                    "annotations": []
                })

            elif t == "response.output_item.done":
                recorded["status"] = "completed"
                break

            elif t == "error":
                # logging here if needed
                pass
        
        print()
        return recorded

    def chat(self, message: Union[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Send a chat request to the model and return the assistant's response text.

        Args:
            message (Union[str, List[Dict[str, Any]]]): The input to generate a response from.
                According to the OpenAI Responses API documentation:
                https://platform.openai.com/docs/api-reference/responses/create

                This parameter supports multiple formats:
                
                - str: A plain text input, treated as a user message by default.
                - list: A list of structured input items (dicts), each representing part of the model's context.

                Each item in the list may include:
                    - Input messages with roles such as 'user', 'assistant', 'system', or 'developer'.
                      Instructions from 'developer' or 'system' roles override those from 'user'.
                      Messages with the 'assistant' role are typically responses from previous turns.
                      ( means the output from ai )
                    - Other content types like images, audio, or tool call outputs, depending on context.
                    ( auto mode in defult )

        Returns:
            The recorded message object used for history tracking.

            Example:
                {
                    "id": "msg_68.....3fa",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "...",
                            "annotations": [],
                            "logprobs": []
                        }
                    ],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message"
                }
        """

        if not isinstance(message, (str, list)):
            raise TypeError("message must be str or list of dicts")

        # list[dict]
        if isinstance(message, list):
            if not all(isinstance(item, dict) for item in message):
                raise TypeError("each item in message must be a dict")

        response = self._post_message(message, self.timeout)
        
        if self.opt.stream:
            return self._prase_stream(response)
        else:
            return response.to_dict()["output"][0]
        
    def get_option(self) -> GPTOption:
        """
        Get the current GPTOption.
        """
        return self.opt
    
    def set_option(self, opt: Optional[GPTOption] = None) -> None:
        """
        Set a new GPTOption. If None, resets to defaults.

        Args:
            opt (Optional[GPTOption]): new option instance or None
        """
        self.opt = opt if opt else GPTOption()
        
    def __repr__(self) -> str:
        return f"<GPTController(model={self.opt.model}, timeout={self.timeout})>"
