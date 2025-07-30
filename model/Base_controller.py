"""
file: module/Base_controller.py

Defines an abstract base class for unified AI controller interfaces.

References:
- Abstract Base Classes (ABC): https://docs.python.org/3/library/abc.html
"""

# using Abstract Base Classes provided by python offical
from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BaseOption(ABC):
    """
    Abstract base class for option configurations passed into BaseController implementations.

    This class is intended to provide a unified interface for configuring different AI models
    (e.g., model name, temperature, max tokens, etc.) in a controller-agnostic way.

    Subclasses must implement:
    - `to_dict`: serialization logic to prepare data for API requests.
    - `get_model_option`: a static method that returns all supported model identifiers.
    - `__repr__`: a string representation of the configuration object for debugging/logging.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the option configuration into a dictionary format suitable for API payloads.

        Returns:
            dict: A dictionary containing serialized configuration data.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_model_option() -> List[str]:
        """
        Returns a list of supported model names that this option class is compatible with.

        Returns:
            list[str]: A list of model identifiers (e.g., ["claude-3-haiku", "claude-3-sonnet"]).
        """
        pass
    
    @abstractmethod
    def __repr__(self) -> str:
        """
        Returns a human-readable string representation of this option configuration.

        Used for debugging, logging, or diagnostics.
        """
        pass


class BaseController(ABC):
    """
    Abstract base class for AI controllers.

    This interface defines the standard protocol for integrating various AI backend services
    (e.g., OpenAI, Anthropic, Google PaLM). Subclasses must implement concrete logic for
    handling chat interactions and provide a descriptive representation of their internal state.

    Typical usage:
        controller = ConcreteController()
        response = controller.chat(message)

    Subclasses must implement:
        - chat: the core method to process input messages and generate AI responses.
        - __repr__: useful for logging and debugging controller configurations.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def chat(self, message: dict) -> str:
        """
        Abstract method to handle chat interactions with a specific AI implementation.

        Args:
            message (dict): The input message payload. Usually a list of role-content dicts,
                            e.g., [{"role": "user", "content": "Hello!"}]

        Returns:
            str: The AI-generated response text.

        Note:
            This method must encapsulate all API logic (e.g., formatting, sending, parsing)
            necessary for the subclass’s backend AI system.
        """
        pass

    @abstractmethod
    def __repr__(self) -> str:
        """
        Returns a human-readable string representation of the controller.

        Typically includes backend name, model configuration, or other internal state.
        Useful for debugging, logging, or diagnostics.
        """
        pass
