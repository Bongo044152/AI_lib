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

# https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.pipeline
import json  # 處理 JSON 格式資料的標準套件 / Standard JSON library
import torch  # PyTorch 是一個深度學習框架 / PyTorch deep learning framework
from transformers import (  # Transformers 模型相關函式庫 / HuggingFace Transformers library
    AutoModelForCausalLM,  # 自動載入因果語言模型 / Auto loader for causal language models
    AutoTokenizer,  # 自動載入對應的 tokenizer / Auto loader for tokenizer
    BitsAndBytesConfig,  # 用來設定量化模型的參數 / Configuration for model quantization
    pipeline,  # 提供簡單的模型推論介面 / Easy interface for model inference
)

import logging, os
from .BaseModel import BaseModel, BaseOption
from typing import *
import pydantic

from .types import ModelIn, ModelOut

# https://huggingface.co/docs/huggingface_hub/en/package_reference/hf_api#huggingface_hub.HfApi.model_info
from huggingface_hub import model_info

# https://huggingface.co/docs/huggingface_hub/v1.0.0.rc5/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError
from huggingface_hub.utils import RepositoryNotFoundError

from .config import env_path
from dotenv import load_dotenv

# Load API key from .env
load_dotenv(env_path)
API_KEY = os.getenv("HF_TOKEN")
if not API_KEY:
    raise RuntimeError("HF_TOKEN is required in environment")

# https://huggingface.co/docs/huggingface_hub/v1.0.0.rc5/en/package_reference/authentication#huggingface_hub.login
# https://huggingface.co/docs/huggingface_hub/en/quick-start
# https://stackoverflow.max-everyday.com/2025/03/colab-hugging-face-api-token/
# https://huggingface.co/docs/transformers.js/en/guides/private
from huggingface_hub import login

login(token=API_KEY)  # HF_TOKEN


class HuggingFaceOption(BaseOption):
    """
    The option used to config HuggingFace model.
    """

    # __init__
    model: str = ""
    temperature: Union[int, float] = 0.8
    max_output_tokens: int = 2048
    stream: bool = False
    generation_config: dict = {}

    @pydantic.field_validator("model")
    @classmethod
    def _check_model(cls, model):
        try:
            assert model_info(model).pipeline_tag == "text-generation", "錯誤的任務類型"
        except RepositoryNotFoundError as e:
            assert False, "模型不存在"

        return model

    @pydantic.field_validator("temperature")
    def _check_temperature(temperature):
        assert 0 <= temperature <= 1, "temperature must be between 0 and 1"
        return temperature

    def to_dict(self) -> Dict[str, Any]:
        """Serialize options to the API payload."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "stream": self.stream,
            "kargs": self.generation_config,
        }

    def __repr__(self) -> str:
        return (
            f"<HuggingFaceOption("
            f"model={self.model}, "
            f"temperature={self.temperature}, "
            f"max_output_tokens={self.max_output_tokens}, "
            f"stream={self.stream})>"
            f"kargs={self.generation_config}"
        )


class HuggingFaceModel(BaseModel):

    # __init__
    opt: Optional[HuggingFaceOption] = HuggingFaceOption()

    @pydantic.validate_call
    def chat(self, message: ModelIn, **generation_config) -> ModelOut:
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
        if isinstance(message.content, str):
            message.content = [
                {
                    "role": "user",
                    "content": {"text": message.content},
                }
            ]

        if message.system_prompt:
            message.content.insert(
                0,
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": message.system_prompt}],
                },
            )

        temp_generation_config = self.opt.generation_config.copy()
        temp_generation_config.update(generation_config)
        generation_config = temp_generation_config

        ##################################
        # process message
        ##################################

        # 設定量化參數，減少記憶體使用 / Set quantization settings for smaller memory usage
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,  # 使用 4-bit 權重 / Use 4-bit weights
            bnb_4bit_use_double_quant=True,  # 啟用雙重量化 / Enable double quantization
            bnb_4bit_quant_type="nf4",  # 使用 nf4 量化類型 / Quantization type
            bnb_4bit_compute_dtype=torch.bfloat16,  # 使用 bfloat16 進行計算 / Use bfloat16 for compute
        )
        # 載入微調好的語言模型（Gemma）/ Load pretrained LLM
        llm = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path="unsloth/gemma-3-4b-it",
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        # 載入對應的 tokenizer / Load tokenizer for the model
        tokenizer = AutoTokenizer.from_pretrained(
            "unsloth/gemma-3-4b-it",
        )
        # 使用 pipeline 包裝模型推論介面 / Create a pipeline for generation
        llm_pipe = pipeline(
            "text-generation",  # 任務為文本生成 / Task type
            model=llm,  # 使用的模型 / LLM
            tokenizer=tokenizer,  # tokenizer
            max_new_tokens=self.opt.max_output_tokens,  # 回應最大長度 / Maximum new tokens
            do_sample=False,  # 不使用隨機 sampling（使用貪婪解碼）/ Greedy decoding
            device_map="auto",
        )

        res: ModelOut = {
            "model": self.opt.model,
            "output": llm_pipe(message.content, **temp_generation_config),
            "thinking": "",
        }

        return res

    def get_option(self) -> HuggingFaceOption:
        """
        Get the current GPTOption.
        """
        return self.opt

    @pydantic.validate_call
    def set_option(self, opt: HuggingFaceOption) -> None:
        """
        Set a new GPTOption. If None, resets to defaults.

        Args:
            opt (Optional[GPTOption]): new option for GPTModel
        """
        if opt:
            assert isinstance(opt, HuggingFaceOption)

        self.opt = opt if opt else HuggingFaceOption()

    def __repr__(self) -> str:
        return f"<HuggingFaceModel(model={self.opt.model})>"
