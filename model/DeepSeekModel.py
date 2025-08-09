"""
file: model/DeepSeekModel.py

Model for DeepSeek Responses API


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
import requests
import os, json
import logging
from typing import *
import requests
from dotenv import load_dotenv
import pydantic
from .BaseModel import BaseOption
from .BaseModel import BaseModel,BaseOption
from .config import env_path
from .types import ModelIn, ModelOut

# Load API key from .env
load_dotenv(env_path)
API_KEY = os.getenv('NVIDIA_DEEPSEEK_API_KEY')
if not API_KEY:
    #logging if needed
    raise RuntimeError("NVIDIA_DEEPSEEK_API_KEY is required in environment")


class DeepSeekOption(BaseOption):
    """
    Options for DeepSeekController.
    """

    #__init__
    model: str = "deepseek-ai/deepseek-r1"
    temperature: float = 0.6
    max_tokens: int = 4096
    stream: bool = False

    @pydantic.field_validator("model")
    def _check_model(model):
        assert model != "deepseek-ai/deepseek-r1", "model not supported"
        return model
    
    @pydantic.field_validator("temperature")
    def _check_temperature(temperature):
        assert 0 <= temperature <= 1, "temperature must be between 0 and 1"
        return temperature
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize options to API payload."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": self.stream,
        }
    
    def __repr__(self) -> str:
        return (
            f"<DeepSeekOption("
            f"model={self.model}, "
            f"temperature={self.temperature}, "
            f"max_output_tokens={self.max_tokens}, "
            f"stream={self.stream})>"
        )


class DeepSeekModel(BaseModel):
    """
    Controller for NVIDIA DeepSeek AI inference.
    """
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
    BASE_URL:ClassVar[str] = "https://integrate.api.nvidia.com/v1/chat/completions"
    opt: Optional[DeepSeekOption] = DeepSeekOption()
    timeout: Union[tuple[int, int], int] = (5, 60)
    session: requests.Session = requests.Session()

    def chat(
        self,
        message: ModelIn
    ) -> ModelOut:
        """
        Send a chat request to the model and return the assistant's response text.

        Args:
            message (ModelIn): Unified model input object format.

        Returns:
            ModelOut:
            The message object with three fields:
                + model: the model which reply your message.
                + thinking: the process of model thinking, in text.
                + output: the model output, text only.

            Example:
                {
                    "model": model_name,
                    "thinking": "ai thinking, if it did",
                    "output": "ai output"
                }

        NOTE: The final message's role must be 'user'.
        
        learn more: https://docs.api.nvidia.com/nim/reference/deepseek-ai-deepseek-r1-infer

        Returns:
            str: The AI-generated response content.
        """
        if isinstance(message.content, str):
            message.content = [{
                "role": "user",
                "content": message.content
            }]
        payload = {
            "model": self.opt.model,
            "temperature": self.opt.temperature,
            "max_tokens": self.opt.max_tokens,
            "stream": self.opt.stream,
            "messages": message.content
        }
        ###post message
        response = self._post_message(payload)
        
        if self.opt.stream:
            return self._parse_stream(response)
        else:
            return_data = response.json()
            res = {
                "model": self.opt.model,
                "thinking": "",
                "output": ""
            }
        
            for item in return_data["choices"]:
                res["output"] = item["message"]["content"]

            s = res["output"]
            START, END = "<think>", "</think>"
            tail = s.index(END, len(START))
            res["thinking"] = s[len(START) : tail].strip().strip('\n')
            res["output"]   = s[tail + len(END) :].strip().strip('\n')

            return res
        
    @pydantic.validate_call
    def _post_message(
        self,
        payload: Dict[str,Any]
    ) -> requests.Response:
        """
        Internal: Perform HTTP POST with headers, using custom timeouts.
        """
        headers = {
            "accept": "text/event-stream" if self.opt.stream else "application/json",
            "content-Type": "application/json",
            "authorization": f"Bearer {API_KEY}"
        }
        resp = self.session.post(
            DeepSeekModel.BASE_URL,
            headers=headers,
            json=payload,
            timeout=self.timeout,
            stream=self.opt.stream
        )

        # if the URL is invalid or returns a 4xx/5xx status code, it raises an HTTPError.
        # see more: https://www.geeksforgeeks.org/python/response-raise_for_status-python-requests/
        resp.raise_for_status()
        return resp
    
    @pydantic.validate_call
    def _parse_stream(
        self, 
        response
        ) -> ModelOut:
        """
        Parse Server-Sent Events (SSE) from a streaming response.
        """
        res: ModelOut = {
            "model": self.opt.model,
            "thinking": "",
            "output": ""
        }

        is_think_end = False
        for raw_line in response.iter_lines(decode_unicode=True):
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
                    if not is_think_end:
                        if content == "</think>":
                            is_think_end = True
                        elif content != "<think>":
                            res["thinking"] += content
                    else:
                        res["output"] += content
                    print(content, end="", flush=True)
            except json.JSONDecodeError:
                # logging here if needed
                raise # or pass
        
        res["thinking"] = res["thinking"].strip().strip('\n')
        return res
    
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