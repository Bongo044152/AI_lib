"""
file: module/GPTModel.py

Model for OpenAI Responses API (gpt-4o, o3, etc.).

This module exposes GPTOption for configuration and GPTModel to send
requests via the OpenAI Responses API. Supports both streaming and non-streaming
modes. Use streaming to receive incremental deltas, and increase user experience.

Example:
    see example/gpt

See:
    https://platform.openai.com/docs/api-reference/responses
"""

# https://github.com/openai/openai-python?tab=readme-ov-file
from openai import OpenAI, APITimeoutError, BadRequestError
from openai.types.responses import Response as openai_Response

import logging, os
from .BaseModel import BaseModel, BaseOption
from typing import *
from pydantic import Field
import pydantic

from .types.ModelLists import Gpt_common_model_types, Gpt_reasoning_model_types
from .types import ModelIn, ModelOut

# using .env file to store api key
from dotenv import load_dotenv
from .config import env_path


class GPTOption(BaseOption):
    """
    The option used to config GPT model.
    """

    # class attribute
    COMMON_MODELS: ClassVar[tuple[str, ...]] = get_args(Gpt_common_model_types)
    REASONING_MODELS: ClassVar[tuple[str, ...]] = get_args(Gpt_reasoning_model_types)

    # __init__
    model: Union[Gpt_common_model_types, Gpt_reasoning_model_types] = Field(
        default=COMMON_MODELS[0]
    )
    temperature: Union[int, float] = 0.8
    max_output_tokens: int = 2048
    stream: bool = False

    @pydantic.field_validator("model")
    @classmethod
    def _check_model(cls, model):
        model_list = []
        for i in cls.COMMON_MODELS:
            model_list.append(i)
        for i in cls.REASONING_MODELS:
            model_list.append(i)

        assert model in model_list, "model not supported"
        return model

    @pydantic.field_validator("temperature")
    def _check_temperature(temperature):
        assert 0 <= temperature <= 2, "temperature must be between 0 and 2"
        return temperature

    def __repr__(self) -> str:
        return (
            f"<GPTOption("
            f"model={self.model}, "
            f"temperature={self.temperature}, "
            f"max_output_tokens={self.max_output_tokens}, "
            f"stream={self.stream})>"
        )


class GPTModel(BaseModel):
    # class attribute
    COMMON_MODELS: ClassVar[tuple[str, ...]] = get_args(Gpt_common_model_types)
    REASONING_MODELS: ClassVar[tuple[str, ...]] = get_args(Gpt_reasoning_model_types)
    CLIENT: ClassVar[OpenAI] = None

    # __init__
    opt: Optional[GPTOption] = Field(default_factory=lambda: GPTOption())
    timeout: Union[tuple[int, int], int] = (5, 60)

    @pydantic.model_validator(mode="after")
    @classmethod
    def _init_client(cls, self) -> Self:
        if cls.CLIENT:
            return self
        load_dotenv(env_path)
        API_KEY = os.environ.get("OPENAI_API_KEY")
        if not API_KEY:
            # logging if needed
            raise RuntimeError("OPENAI_API_KEY is required in environment")
        cls.CLIENT = OpenAI(api_key=API_KEY)
        return self

    @pydantic.validate_call
    def chat(self, message: ModelIn) -> ModelOut:
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
        """

        ##################################
        # format to the correct field
        ##################################

        if isinstance(message.content, list):
            for item in message.content:
                role = item["role"]
                text = item["content"]

                if role == "user":
                    item["content"] = [{"type": "input_text", "text": text}]
                elif role == "assistant":
                    item["content"] = [{"type": "output_text", "text": text}]
        elif isinstance(message.content, str):
            message.content = [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": message.content}],
                }
            ]

        # https://platform.openai.com/docs/guides/reasoning?api-mode=responses
        if message.thinking:
            if self.opt.model in GPTModel.REASONING_MODELS:
                message.thinking = {"effort": "medium", "summary": "detailed"}
            else:
                raise ValueError(f"{self.opt.model} is not reasoning model")
        else:
            message.thinking = None

        if message.system_prompt:
            message.content.insert(
                0,
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": message.system_prompt}],
                },
            )

        if self.opt.model in GPTModel.REASONING_MODELS:
            self.opt.temperature = None

        ##################################
        # post message
        ##################################

        response = self._post_message(message)

        if self.opt.stream:
            return self._prase_stream(response)
        else:
            return_data = response.to_dict()
            res: ModelOut = {"model": self.opt.model, "thinking": "", "output": ""}

            for item in return_data["output"]:
                if item["type"] == "reasoning":
                    for obj in item["summary"]:
                        res["thinking"] += obj["text"]
                elif item["type"] == "message":
                    res["output"] = item["content"][0]["text"]

            return res

    @pydantic.validate_call
    def _post_message(self, message: ModelIn) -> openai_Response:
        """
        Internal: make the HTTP POST.
        the argument message is provided from `chat` function
        """
        try:
            response: openai_Response = GPTModel.CLIENT.with_options(
                timeout=self.timeout
            ).responses.create(
                model=self.opt.model,
                input=message.content,
                temperature=self.opt.temperature,
                stream=self.opt.stream,
                reasoning=message.thinking,
                tool_choice="none",
            )
        except APITimeoutError as e:
            # https://github.com/openai/openai-python?tab=readme-ov-file#timeouts
            # logging if needed
            raise
        except BadRequestError as e:
            # invaid arg
            # logging if needed
            raise
        except Exception as e:
            # logging if needed
            raise

        # note: not real openai_Response if stream is true
        # see: https://github.com/openai/openai-python/blob/main/src/openai/resources/responses/responses.py
        return response

    @pydantic.validate_call
    def _prase_stream(self, response) -> ModelOut:
        """
        Internal: print the message AI generate in the stream

        Reference:
            - https://platform.openai.com/docs/guides/streaming-responses?api-mode=responses
        """
        result = {"model": self.opt.model, "thinking": "", "output": ""}

        once = True
        out = False

        # https://platform.openai.com/docs/guides/streaming-responses?api-mode=responses
        for event in response:
            # https://platform.openai.com/docs/api-reference/responses_streaming/response
            # https://github.com/openai/openai-python/blob/main/src/openai/types/responses/response_created_event.py
            event: dict = event.to_dict()

            t = event["type"]

            if t == "response.output_text.delta":
                if out:
                    print("\n</thinking>", flush=True)
                    out = False
                delta = event.get("delta")
                print(delta, end="", flush=True)

            elif t == "response.output_text.done":
                result["output"] = event["text"]

            elif t == "response.reasoning_summary_text.delta":
                if once:
                    print("<thinking>", flush=True)
                    once = False
                    out = True
                delta = event.get("delta")
                print(delta, end="", flush=True)

            elif t == "response.reasoning_summary_text.done":
                result["thinking"] = event["text"]

            elif t == "response.completed":
                break

            elif t == "error":
                # https://platform.openai.com/docs/api-reference/responses_streaming/error
                # logging here if needed
                break

        print(flush=True)
        return result

    def get_option(self) -> GPTOption:
        """
        Get the current GPTOption.
        """
        return self.opt

    @pydantic.validate_call
    def set_option(self, opt: GPTOption) -> None:
        """
        Set a new GPTOption. If None, resets to defaults.

        Args:
            opt (Optional[GPTOption]): new option for GPTModel
        """
        if opt:
            assert isinstance(opt, GPTOption)

        self.opt = opt if opt else GPTOption()

    def __repr__(self) -> str:
        return f"<GPTModel(model={self.opt.model}, timeout={self.timeout})>"
