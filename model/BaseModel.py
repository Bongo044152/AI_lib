"""
file: module/BaseModel.py

Defines an abstract base class for unified AI controller interfaces.

References:
- Abstract Base Classes (ABC): https://docs.python.org/3/library/abc.html
"""

# using Abstract Base Classes provided by python offical
from abc import ABC, abstractmethod
import pydantic
from . import ModelIn, ModelOut
from typing import *


class BaseOption(pydantic.BaseModel, ABC):
    """
    Abstract base class for option configurations passed into BaseModel implementations.

    This class is intended to provide a unified interface for configuring different AI models.
    (e.g., model name, temperature, max tokens, etc.)

    Subclasses must implement:
    - `__repr__`: a string representation of the configuration object for debugging/logging.
    """

    @abstractmethod
    def __repr__(self) -> str:
        """
        Returns a human-readable string representation of this option configuration.

        Used for debugging, logging, or diagnostics.
        """
        pass


class BaseModel(pydantic.BaseModel, ABC):
    """
    Abstract base class for AI model.

    This interface defines the standard protocol for integrating various AI backend services
    (e.g., OpenAI, Anthropic, Claude). Subclasses must implement concrete logic for
    handling chat interactions and provide a descriptive representation of their internal state.

    Subclasses must implement:
        - chat: the core method to process input messages and generate AI responses.
        - get_option: get the Option in the model
        - set_option: set new Option for model
        - __repr__: useful for logging and debugging model configurations.
    """

    @abstractmethod
    @override
    def chat(self, message: ModelIn) -> ModelOut:
        """
        Abstract method to handle chat interactions with a specific AI implementation.

        Args:
            message (ModelIn): Unified model input object format.

        Returns:
            ModelOut:
            Unified model output object format.

        Note:
            This method must encapsulate all API logic (e.g., formatting, sending, parsing)
            necessary for the subclass’s backend AI system.
        """
        pass

    @abstractmethod
    def get_option(self) -> BaseOption:
        """
        Get the current Option.
        """
        pass

    @abstractmethod
    def set_option(self, opt: Optional[BaseOption] = None) -> None:
        """
        Set a new Option. If None, resets to defaults.

        Args:
            opt (Optional[Option]): new option instance or None
        """
        pass

    @abstractmethod
    @override
    def __repr__(self) -> str:
        """
        Returns a human-readable string representation of the model.

        Typically includes backend name, model configuration, or other internal state.
        Useful for debugging, logging, or diagnostics.
        """
        pass
