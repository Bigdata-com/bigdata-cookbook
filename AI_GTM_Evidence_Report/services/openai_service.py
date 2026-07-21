"""
OpenAI Service for structured content generation.

This service provides async methods to interact with OpenAI's API,
with support for structured output using JSON schemas.
"""

import asyncio
import json
import os
from typing import Any, Type, TypeVar, Optional

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from pydantic import BaseModel

from .llm_service import LLMService


T = TypeVar('T', bound=BaseModel)


class OpenAIService(LLMService):
    """
    Service for interacting with OpenAI's API.
    
    Supports structured output generation using Pydantic models
    via JSON mode and response_format.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        **kwargs
    ):
        """
        Initialize the OpenAI service.
        
        Args:
            api_key: OpenAI API key. If not provided, will use OPENAI_API_KEY env var.
            model: Model to use for generation. Defaults to gpt-4o-mini.
            **kwargs: Additional arguments (ignored for now).
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI package not installed. Install it with: pip install openai"
            )
        
        self._model = model
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided. Set OPENAI_API_KEY environment variable "
                "or pass api_key to constructor."
            )
        
        self.client = AsyncOpenAI(api_key=self.api_key)
    
    @property
    def provider_name(self) -> str:
        """Return the name of the LLM provider."""
        return "openai"
    
    @property
    def model(self) -> str:
        """Return the default model name being used."""
        return self._model
    
    def _pydantic_to_json_schema(self, schema_class: Type[BaseModel]) -> dict:
        """
        Convert a Pydantic model to JSON schema for OpenAI.
        
        OpenAI requires additionalProperties: false for structured output.
        
        Args:
            schema_class: Pydantic model class
        
        Returns:
            JSON schema dictionary with additionalProperties set to false
        """
        schema = schema_class.model_json_schema()
        
        # OpenAI requires additionalProperties: false for structured output
        # Recursively add it to all object schemas
        def add_additional_properties(obj):
            if isinstance(obj, dict):
                if obj.get("type") == "object":
                    obj["additionalProperties"] = False
                # Recursively process nested objects
                for value in obj.values():
                    if isinstance(value, (dict, list)):
                        add_additional_properties(value)
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, (dict, list)):
                        add_additional_properties(item)
        
        add_additional_properties(schema)
        return schema
    
    async def generate_content(
        self,
        prompt: str,
        response_schema: Type[T],
        model: Optional[str] = None,
        **kwargs
    ) -> T:
        """
        Generate structured content using OpenAI API.
        
        Args:
            prompt: The text prompt to send to the model.
            response_schema: A Pydantic model class defining the expected response structure.
            model: Optional model override. If not provided, uses the instance default.
            **kwargs: Additional config options (temperature, max_tokens, etc.)
        
        Returns:
            An instance of the response_schema model with generated content.
        """
        model_name = model or self.model
        
        # Convert Pydantic schema to JSON schema
        json_schema = self._pydantic_to_json_schema(response_schema)
        
        # Build messages
        messages = [{"role": "user", "content": prompt}]
        
        # Build response format for structured output
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": response_schema.__name__,
                "schema": json_schema,
                "strict": True
            }
        }
        
        # Extract OpenAI-specific parameters
        temperature = kwargs.pop('temperature', None)
        max_tokens = kwargs.pop('max_tokens', None)
        
        # Build request parameters
        request_params = {
            "model": model_name,
            "messages": messages,
            "response_format": response_format,
        }
        
        if temperature is not None:
            request_params["temperature"] = temperature
        if max_tokens is not None:
            request_params["max_tokens"] = max_tokens
        
        # Add any remaining kwargs
        request_params.update(kwargs)
        
        # Make API call
        response = await self.client.chat.completions.create(**request_params)
        
        # Parse response
        if not response.choices or not response.choices[0].message.content:
            raise ValueError("No content in OpenAI response")
        
        content = response.choices[0].message.content
        
        # Parse JSON and create Pydantic model
        try:
            data = json.loads(content)
            return response_schema(**data)
        except (json.JSONDecodeError, TypeError) as e:
            raise ValueError(f"Failed to parse OpenAI response as JSON: {e}")
    
    async def generate_content_list(
        self,
        prompt: str,
        response_schema: Type[T],
        model: Optional[str] = None,
        **kwargs
    ) -> list[T]:
        """
        Generate structured content as a list using OpenAI API.
        
        OpenAI's structured output requires the root schema to be an object,
        so we wrap the array in an object with an 'items' property.
        
        Args:
            prompt: The text prompt to send to the model.
            response_schema: A Pydantic model class defining each item's structure.
            model: Optional model override. If not provided, uses the instance default.
            **kwargs: Additional config options
        
        Returns:
            A list of instances of the response_schema model.
        """
        model_name = model or self.model
        
        # Convert Pydantic schema to JSON schema for array items
        item_schema = self._pydantic_to_json_schema(response_schema)
        
        # OpenAI requires root schema to be an object, so wrap array in an object
        wrapper_schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": item_schema
                }
            },
            "required": ["items"],
            "additionalProperties": False
        }
        
        # Build messages
        messages = [{"role": "user", "content": prompt}]
        
        # Build response format for structured output
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": f"{response_schema.__name__}List",
                "schema": wrapper_schema,
                "strict": True
            }
        }
        
        # Extract OpenAI-specific parameters
        temperature = kwargs.pop('temperature', None)
        max_tokens = kwargs.pop('max_tokens', None)
        
        # Build request parameters
        request_params = {
            "model": model_name,
            "messages": messages,
            "response_format": response_format,
        }
        
        if temperature is not None:
            request_params["temperature"] = temperature
        if max_tokens is not None:
            request_params["max_tokens"] = max_tokens
        
        # Add any remaining kwargs
        request_params.update(kwargs)
        
        # Make API call
        response = await self.client.chat.completions.create(**request_params)
        
        # Parse response
        if not response.choices or not response.choices[0].message.content:
            raise ValueError("No content in OpenAI response")
        
        content = response.choices[0].message.content
        
        # Parse JSON and create list of Pydantic models
        try:
            data = json.loads(content)
            # OpenAI returns wrapped object with 'items' property
            if isinstance(data, dict) and "items" in data:
                items = data["items"]
            elif isinstance(data, list):
                # Fallback: if it's already a list, use it directly
                items = data
            else:
                raise ValueError(f"Expected object with 'items' property or list, got {type(data)}")
            
            if not isinstance(items, list):
                raise ValueError(f"Expected 'items' to be a list, got {type(items)}")
            
            return [response_schema(**item) for item in items]
        except (json.JSONDecodeError, TypeError) as e:
            raise ValueError(f"Failed to parse OpenAI response as JSON list: {e}")
    
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
            response_schema: Optional schema for structured output (ignored for raw).
            model: Optional model override.
            **kwargs: Additional config options.
        
        Returns:
            Raw text response from the model.
        """
        model_name = model or self.model
        
        # Build messages
        messages = [{"role": "user", "content": prompt}]
        
        # Extract OpenAI-specific parameters
        temperature = kwargs.pop('temperature', None)
        max_tokens = kwargs.pop('max_tokens', None)
        
        # Build request parameters
        request_params = {
            "model": model_name,
            "messages": messages,
        }
        
        if temperature is not None:
            request_params["temperature"] = temperature
        if max_tokens is not None:
            request_params["max_tokens"] = max_tokens
        
        # Add any remaining kwargs
        request_params.update(kwargs)
        
        # Make API call
        response = await self.client.chat.completions.create(**request_params)
        
        # Return raw text
        if not response.choices or not response.choices[0].message.content:
            raise ValueError("No content in OpenAI response")
        
        return response.choices[0].message.content

