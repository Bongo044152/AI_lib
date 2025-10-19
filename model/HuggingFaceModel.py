"""
file: module/HuggingFaceModel.py

Hugging Face text generation model wrapper (e.g., BLOOM, LLaMA, Mistral).

This module provides `HuggingFaceModel` for unified text generation through
the Transformers pipeline, and `HuggingFaceOption` for configuration.

Features:
    - Supports quantized loading (4-bit) for reduced memory usage.
    - Compatible with models under "text-generation" and "text2text-generation".
    - Automatically handles chat templates (system / user / assistant).
    - Easy integration with `ModelIn` / `ModelOut` structures.

See:
    https://huggingface.co/docs/transformers/main/en/conversations
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

# -- Load HF_TOKEN from .env and login ---------------------------------------
# We do this at import-time for simplicity. If you want zero side-effects,
# move login() into a lazy init path (e.g., the first time chat() is called).
# see below:
## https://huggingface.co/docs/huggingface_hub/v1.0.0.rc5/en/package_reference/authentication#huggingface_hub.login
## https://stackoverflow.max-everyday.com/2025/03/colab-hugging-face-api-token/
## https://huggingface.co/docs/transformers.js/en/guides/private
from huggingface_hub import (
    login,
)  # HF auth; https://huggingface.co/docs/huggingface_hub/en/quick-start

login(token=API_KEY)  # HF_TOKEN


# -- Options ------------------------------------------------------------------


class HuggingFaceOption(BaseOption):
    """
    Options for configuring Hugging Face text generation.

    Notes:
    - temperature in [0,1]: when 0, we use greedy decoding (do_sample=False).
      when >0, we enable sampling (do_sample=True) and pass the temperature through.
    - generation_config: free-form kwargs forwarded to `pipeline(...)`
      (e.g., top_p, top_k, repetition_penalty, etc.)
    """

    # __init__
    temperature: Union[int, float] = 0.8
    max_output_tokens: int = 512
    stream: bool = False  # 功能尚未開放
    generation_config: Dict[str, Any] = {}

    @pydantic.field_validator("temperature")
    @classmethod
    def _check_temperature(cls, temperature: Union[int, float]) -> Union[int, float]:
        assert 0 <= temperature <= 1, "temperature must be between 0 and 1"
        return temperature

    def to_dict(self) -> Dict[str, Any]:
        """Serialize options to a dict (useful for logging/UI)."""
        return {
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "stream": self.stream,
            "generation_config": self.generation_config,
        }

    def __repr__(self) -> str:
        return (
            f"<HuggingFaceOption("
            f"temperature={self.temperature}, "
            f"max_output_tokens={self.max_output_tokens}, "
            f"stream={self.stream}, "
            f"generation_config={self.generation_config}"
            f")>"
        )


# -- Model --------------------------------------------------------------------


class HuggingFaceModel(BaseModel):
    """
    A lightweight wrapper around `transformers` text-generation pipeline.

    - Validates the model ID on construction.
    - Lazily loads and caches (quantized) model/tokenizer/pipeline on first call.
    - Converts `ModelIn` (string or multi-turn) into a prompt string (or chat template).

    Caveat:
    - If your target model is instruction-tuned with a specific chat template,
      results improve when `tokenizer.apply_chat_template` exists.
      see: https://huggingface.co/docs/transformers/en/chat_templating
    """

    model_config = pydantic.ConfigDict(validate_default=True)  # gpt: 讓預設值也會被驗證

    # __init__
    opt: Optional[HuggingFaceOption] = HuggingFaceOption()
    model: str = ""

    # Internal caches to avoid reloading every call.
    _llm: Optional[AutoModelForCausalLM] = None
    _tokenizer: Optional[AutoTokenizer] = None
    _has_chat_template = False
    _pipe: Optional[Any] = None  # transformers.Pipeline

    @pydantic.field_validator("model")
    @classmethod
    def _check_model(cls, model: str) -> str:
        """
        Accept models with pipeline_tag in {"text-generation", "text2text-generation"}.
        Provide clearer diagnostics for common failure modes.
        """
        try:
            info = model_info(model)
            tag = getattr(info, "pipeline_tag", None)
            if tag not in {"text-generation", "text2text-generation"}:
                raise AssertionError(
                    f"Unsupported pipeline_tag={tag!r}. "
                    f"Require 'text-generation' or 'text2text-generation'."
                )
        except RepositoryNotFoundError:
            raise AssertionError(f"模型不存在或未公開：{model}")
        except Exception as e:
            # Network/auth or other hub issues.
            raise AssertionError(f"無法驗證模型（可能為網路或權限問題）：{e}")

        return model

    # -- Lazy init helpers ----------------------------------------------------
    def _create_pipeline_lazy(self, **overrides: Any) -> None:
        """
        Lazily create and cache tokenizer/llm/pipeline.
        Uses 4-bit quantization by default to reduce memory.
        """
        if self._pipe is not None:
            return

        # 設定量化參數，減少記憶體使用 / Set quantization settings for smaller memory usage
        # https://huggingface.co/docs/transformers/en/conversations#performance-and-memory-usage
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,  # 使用 4-bit 權重 / Use 4-bit weights
            bnb_4bit_use_double_quant=True,  # 啟用雙重量化 / Enable double quantization
            bnb_4bit_quant_type="nf4",  # 使用 nf4 量化類型 / Quantization type
            bnb_4bit_compute_dtype=torch.bfloat16,  # 使用 bfloat16 進行計算 / Use bfloat16 for compute
        )
        # 載入微調好的語言模型（Gemma）/ Load pretrained LLM
        self._llm = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.model,
            quantization_config=bnb_config,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        # 載入對應的 tokenizer / Load tokenizer for the model
        self._tokenizer = AutoTokenizer.from_pretrained(self.model)

        # chat template
        # see: https://huggingface.co/docs/transformers/en/chat_templating_writing
        if not getattr(self._tokenizer, "chat_template", None):
            self._tokenizer.chat_template = """{% for message in messages %}
{% if message['role'] == 'system' -%}
System: {{ message['content'] }}
{% elif message['role'] == 'user' -%}
User: {{ message['content'] }}
{% elif message['role'] == 'assistant' -%}
Assistant: {{ message['content'] }}
{% endif %}
{% endfor %}
Assistant:""".strip()

        # --- Try to detect chat template ------------------------------------
        # If tokenizer has a chat template, we can later use:
        #   tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # doc: https://huggingface.co/docs/transformers/main/chat_templating
        self._has_chat_template = False
        try:
            tmpl = getattr(self._tokenizer, "chat_template", None)
            if tmpl:
                self._has_chat_template = True
                logging.info(
                    "[HF] Chat template detected (first 120 chars): %s",
                    str(tmpl)[:120].replace("\n", " "),
                )
            else:
                logging.info(
                    "[HF] No chat template found on tokenizer; will fallback to naive concat."
                )
        except Exception as e:
            logging.warning("[HF] Failed to read chat template: %s", e)

        # Base generation kwargs; allow per-call overrides later.
        base_kwargs = {
            "task": "text-generation",  # 任務為文本生成 / Task type
            "model": self._llm,  # 使用的模型 / LLM
            "tokenizer": self._tokenizer,  # tokenizer
            "max_new_tokens": self.opt.max_output_tokens,  # 回應最大長度 / Maximum new tokens
            # Sampling toggled later inside chat() according to temperature.
            "do_sample": False,  # 不使用隨機 sampling（使用貪婪解碼）/ Greedy decoding
            "device_map": "auto",  # accelerate 加速
        }
        base_kwargs.update(overrides)

        # 使用 pipeline 包裝模型推論介面 / Create a pipeline for generation
        self._pipe = pipeline(**base_kwargs)

    @pydantic.validate_call
    def chat(self, message: ModelIn, **generation_config: Any) -> ModelOut:
        """
        Execute text generation via HF pipeline and return ModelOut.

        Args:
            message: your unified input object.
            **generation_overrides: extra kwargs to override pipeline generation
                                    (e.g., top_p=0.9, repetition_penalty=1.1).
                                    https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.TextGenerationPipeline

        Returns:
            ModelOut:
            The message object with three fields:
                + model: the model which reply your message.
                + thinking: the process of model thinking, in text.
                + output: the model output, text only.

        Example: (ModelIn)
        ```
        {
            "model": model_name,
            "thinking": "ai thinking, not supported now",
            "output": "ai output (string)"
        }
        ```
        """

        temp_generation_config = self.opt.generation_config.copy()
        temp_generation_config.update(generation_config)
        generation_config = temp_generation_config

        # 直接輸出
        if isinstance(message.content, str) and not message.system_prompt:
            self._create_pipeline_lazy()
            model_out = self._pipe(message.content, **temp_generation_config)
            # 可能考慮的安全檢查 ... (略)

            res: ModelOut = {
                "model": self.model,
                "output": model_out,
                "thinking": "",
            }

            return res
        

        ##################################
        # format to the correct field
        ##################################

        if isinstance(message.content, str):
            message.content = [
                {
                    "role": "user",
                    "content": message.content,
                }
            ]

        if message.system_prompt:
            message.content.insert(
                0,
                {
                    "role": "system",
                    "content": message.system_prompt,
                },
            )

        # 別忘了更新 temperature
        if self.opt.temperature > 0:
            generation_config["do_sample"] = True
            generation_config["temperature"] = float(self.opt.temperature)

        ##################################
        # process message
        ##################################

        self._create_pipeline_lazy()

        model_out = self._pipe(message.content, **temp_generation_config)
        # 可能考慮的安全檢查 ... (略)

        res: ModelOut = {
            "model": self.model,
            "output": model_out,
            "thinking": "",
        }

        return res

    def get_option(self) -> HuggingFaceOption:
        """Return current option object."""
        return self.opt

    @pydantic.validate_call
    def set_option(self, opt: Optional[HuggingFaceOption]) -> None:
        """
        Set new options. Pass None to reset to defaults.
        Note: changing options after pipeline creation will not recreate the
        pipeline automatically. If you need a hard reset, ues function `rebuild_pipeline`
        insteaded.
        """
        if opt is not None:
            assert isinstance(opt, HuggingFaceOption)
            self.opt = opt
        else:
            self.opt = HuggingFaceOption()

    def rebuild_pipeline(self) -> None:
        """recreate the pipeline"""
        self._pipe = None
        self._ensure_pipeline()

    # def clear_all_hug_cache():
    #     """
    #     Clear all Hugging Face-related cache data.

    #     Note:
    #         This approach is not elegant — it removes *all* cached content used by Hugging Face,
    #         including models, tokenizers, and datasets across all projects.
    #         It may affect other codebases or applications sharing the same cache directory.

    #     Future Plan:
    #         A more refined method will be implemented later to selectively clear only
    #         the resources associated with the current object.

    #     Reference:
    #         https://huggingface.co/docs/huggingface_hub/en/guides/manage-cache#clean-your-cache
    #     """
    #     from huggingface_hub import scan_cache_dir
    #     cache_info = scan_cache_dir()
    #     cache_info.delete_revisions(cache_info.revisions)

    def set_chat_template(self, template: str) -> None:
        """
        Sets a new chat template for the tokenizer.

        Note:
            This will override the tokenizer’s existing default template.

        see: https://huggingface.co/docs/transformers/en/chat_templating_writing
        """
        self._tokenizer.chat_template = template

    def get_chat_template(self) -> str:
        """
        Returns the current chat template of the tokenizer.

        Note:
            Returns None if no template is set.
        """
        return getattr(self._tokenizer, "chat_template", None)

    def test_chat_template(self, context: str) -> str:
        if not self._tokenizer:
            self._create_pipeline_lazy()
        return self._tokenizer.apply_chat_template(context, tokenize=False);

    def __repr__(self) -> str:
        return f"<HuggingFaceModel(model={self.model})>"
