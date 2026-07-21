"""
Gemini AI Service for structured content generation.

This service provides async methods to interact with Google's Gemini API,
with support for structured output using JSON schemas.
Supports both API key and Vertex AI service account authentication.
"""

import asyncio
import os
from typing import Any, Type, TypeVar, Optional
from functools import wraps

import google.genai as genai
from google.oauth2 import service_account
from pydantic import BaseModel

from .llm_service import LLMService


T = TypeVar('T', bound=BaseModel)


class GeminiService(LLMService):
    """
    Service for interacting with Google's Gemini API.
    
    Supports structured output generation using Pydantic models
    or custom JSON schemas.
    
    Supports two authentication methods:
    1. API Key: Traditional authentication using GEMINI_API_KEY
    2. Vertex AI: Service account authentication using credentials file
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        service_account_path: Optional[str] = None,
        project_id: Optional[str] = None,
        location: str = "us-central1",
        model: str = "gemini-2.5-flash"
    ):
        """
        Initialize the Gemini service.
        
        Args:
            api_key: Google AI API key. If not provided, will use GEMINI_API_KEY env var.
            service_account_path: Path to service account JSON file for Vertex AI.
                                 If not provided, will use GOOGLE_APPLICATION_CREDENTIALS env var.
            project_id: Google Cloud project ID for Vertex AI.
                       If not provided, will use GOOGLE_CLOUD_PROJECT env var.
            location: Google Cloud region for Vertex AI. Defaults to us-central1.
            model: Model to use for generation. Defaults to gemini-2.5-flash.
        
        Note: If service_account_path is provided, Vertex AI authentication will be used.
              Otherwise, will fall back to API key authentication.
        """
        self._model = model
        self.location = location
        
        # Determine authentication method
        service_account_file = service_account_path or os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        
        if service_account_file:
            # Use Vertex AI authentication
            self._init_vertex_ai(service_account_file, project_id)
        else:
            # Use API key authentication
            self._init_api_key(api_key)
    
    @property
    def provider_name(self) -> str:
        """Return the name of the LLM provider."""
        return "gemini"
    
    @property
    def model(self) -> str:
        """Return the default model name being used."""
        return self._model
    
    def _init_vertex_ai(self, service_account_file: str, project_id: Optional[str] = None):
        """Initialize client with Vertex AI service account credentials."""
        if not os.path.exists(service_account_file):
            raise ValueError(
                f"Service account file not found: {service_account_file}"
            )
        
        # Load credentials from service account file
        credentials = service_account.Credentials.from_service_account_file(
            service_account_file,
            scopes=['https://www.googleapis.com/auth/cloud-platform']
        )
        
        # Get project ID from file if not provided
        if not project_id:
            project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
            if not project_id:
                # Try to extract from service account file
                import json
                with open(service_account_file, 'r', encoding='utf-8') as f:
                    service_account_info = json.load(f)
                    project_id = service_account_info.get('project_id')
        
        if not project_id:
            raise ValueError(
                "Project ID not provided. Set GOOGLE_CLOUD_PROJECT environment variable "
                "or pass project_id to constructor."
            )
        
        self.project_id = project_id
        self.credentials = credentials
        self.auth_method = "vertex_ai"
        
        # Initialize Vertex AI client
        self.client = genai.Client(
            vertexai=True,
            project=project_id,
            location=self.location,
            credentials=credentials
        )
    
    def _init_api_key(self, api_key: Optional[str] = None):
        """Initialize client with API key authentication."""
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError(
                "Gemini API key not provided. Set GEMINI_API_KEY environment variable "
                "or pass api_key to constructor, or use service_account_path for Vertex AI."
            )
        
        self.auth_method = "api_key"
        self.client = genai.Client(api_key=self.api_key)
    
    async def generate_content(
        self,
        prompt: str,
        response_schema: Type[T],
        model: Optional[str] = None,
        **kwargs
    ) -> T:
        """
        Generate structured content using Gemini API.
        
        This async method generates content based on the prompt and returns
        a structured response matching the provided Pydantic schema.
        
        Args:
            prompt: The text prompt to send to the model.
            response_schema: A Pydantic model class defining the expected response structure.
            model: Optional model override. If not provided, uses the instance default.
            **kwargs: Additional config options to pass to the API.
        
        Returns:
            An instance of the response_schema model with generated content.
        
        Example:
            ```python
            from pydantic import BaseModel
            
            class Recipe(BaseModel):
                recipe_name: str
                ingredients: list[str]
            
            service = GeminiService()
            recipe = await service.generate_content(
                "Give me a chocolate chip cookie recipe",
                response_schema=Recipe
            )
            print(recipe.recipe_name)
            print(recipe.ingredients)
            ```
        """
        model_name = model or self.model
        
        # Build config
        config = {
            "response_mime_type": "application/json",
            "response_schema": response_schema,
            **kwargs
        }
        
        # Run the sync API call in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config
            )
        )
        
        # Check if response has parsed data
        if not hasattr(response, 'parsed') or response.parsed is None:
            # Try to get text and parse manually
            if hasattr(response, 'text') and response.text:
                import json
                try:
                    data = json.loads(response.text)
                    return response_schema(**data)
                except (json.JSONDecodeError, TypeError):
                    pass
            raise ValueError(f"No parsed data in response. Response attributes: {dir(response)}")
        
        # Return the parsed response
        return response.parsed
    
    async def generate_content_list(
        self,
        prompt: str,
        response_schema: Type[T],
        model: Optional[str] = None,
        **kwargs
    ) -> list[T]:
        """
        Generate structured content as a list using Gemini API.
        
        This is a convenience method for when you want a list of items
        matching a particular schema.
        
        Args:
            prompt: The text prompt to send to the model.
            response_schema: A Pydantic model class defining each item's structure.
            model: Optional model override. If not provided, uses the instance default.
            **kwargs: Additional config options to pass to the API.
        
        Returns:
            A list of instances of the response_schema model.
        
        Example:
            ```python
            from pydantic import BaseModel
            
            class Recipe(BaseModel):
                recipe_name: str
                ingredients: list[str]
            
            service = GeminiService()
            recipes = await service.generate_content_list(
                "Give me 3 popular cookie recipes",
                response_schema=Recipe
            )
            for recipe in recipes:
                print(recipe.recipe_name)
            ```
        """
        model_name = model or self.model
        
        # Build config with list schema
        config = {
            "response_mime_type": "application/json",
            "response_schema": list[response_schema],
            **kwargs
        }
        
        # Run the sync API call in a thread pool
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config
            )
        )
        
        # Check if response has parsed data
        if not hasattr(response, 'parsed') or response.parsed is None:
            # Try to get text and parse manually
            if hasattr(response, 'text') and response.text:
                import json
                try:
                    data = json.loads(response.text)
                    if isinstance(data, list):
                        return [response_schema(**item) for item in data]
                except (json.JSONDecodeError, TypeError):
                    pass
            raise ValueError(f"No parsed data in response. Response attributes: {dir(response)}")
        
        return response.parsed
    
    async def generate_content_raw(
        self,
        prompt: str,
        response_schema: Optional[Any] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate content and return raw text response.
        
        This method is useful when you need the raw response text
        instead of parsed objects.
        
        Args:
            prompt: The text prompt to send to the model.
            response_schema: Optional schema for structured output.
            model: Optional model override.
            **kwargs: Additional config options.
        
        Returns:
            Raw text response from the model.
        """
        model_name = model or self.model
        
        # Build config
        config = kwargs.copy()
        if response_schema:
            config["response_mime_type"] = "application/json"
            config["response_schema"] = response_schema
        
        # Run the sync API call in a thread pool
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config if config else None
            )
        )
        
        return response.text


# Convenience function for one-off requests
async def generate_structured_content(
    prompt: str,
    response_schema: Type[T],
    api_key: Optional[str] = None,
    model: str = "gemini-2.5-flash",
    **kwargs
) -> T:
    """
    Convenience function for generating structured content without instantiating a service.
    
    Args:
        prompt: The text prompt to send to the model.
        response_schema: A Pydantic model class defining the expected response structure.
        api_key: Optional API key. If not provided, uses GEMINI_API_KEY env var.
        model: Model to use. Defaults to gemini-2.5-flash.
        **kwargs: Additional config options.
    
    Returns:
        An instance of the response_schema model with generated content.
    """
    service = GeminiService(api_key=api_key, model=model)
    return await service.generate_content(prompt, response_schema, **kwargs)

