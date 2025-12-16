"""
Abstract base class for LLM services.

This module defines the interface that all LLM providers must implement,
enabling easy switching between different providers (OpenAI, Gemini, etc.).
"""

from abc import ABC, abstractmethod
from typing import Any, Type, TypeVar, Optional
from pydantic import BaseModel

T = TypeVar('T', bound=BaseModel)


class LLMService(ABC):
    """
    Abstract base class for LLM service providers.
    
    All LLM implementations (OpenAI, Gemini, etc.) must inherit from this class
    and implement the required methods.
    """
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of the LLM provider (e.g., 'openai', 'gemini')."""
        pass
    
    @property
    @abstractmethod
    def model(self) -> str:
        """Return the default model name being used."""
        pass
    
    @abstractmethod
    async def generate_content(
        self,
        prompt: str,
        response_schema: Type[T],
        model: Optional[str] = None,
        **kwargs
    ) -> T:
        """
        Generate structured content matching a Pydantic schema.
        
        Args:
            prompt: The text prompt to send to the model.
            response_schema: A Pydantic model class defining the expected response structure.
            model: Optional model override. If not provided, uses the instance default.
            **kwargs: Additional config options to pass to the API.
        
        Returns:
            An instance of the response_schema model with generated content.
        """
        pass
    
    @abstractmethod
    async def generate_content_list(
        self,
        prompt: str,
        response_schema: Type[T],
        model: Optional[str] = None,
        **kwargs
    ) -> list[T]:
        """
        Generate structured content as a list matching a Pydantic schema.
        
        Args:
            prompt: The text prompt to send to the model.
            response_schema: A Pydantic model class defining each item's structure.
            model: Optional model override. If not provided, uses the instance default.
            **kwargs: Additional config options to pass to the API.
        
        Returns:
            A list of instances of the response_schema model.
        """
        pass
    
    @abstractmethod
    async def generate_content_raw(
        self,
        prompt: str,
        response_schema: Optional[Any] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate content and return raw text response.
        
        Args:
            prompt: The text prompt to send to the model.
            response_schema: Optional schema for structured output.
            model: Optional model override.
            **kwargs: Additional config options.
        
        Returns:
            Raw text response from the model.
        """
        pass

