"""
LLM Service Factory for creating LLM service instances.

This factory provides a unified way to create LLM service instances
based on configuration, supporting multiple providers (OpenAI, Gemini, etc.).
"""

import os
import logging
from typing import Optional

from .llm_service import LLMService

logger = logging.getLogger(__name__)


class LLMServiceFactory:
    """
    Factory for creating LLM service instances.
    
    Supports multiple providers:
    - 'openai': OpenAI GPT models
    - 'gemini': Google Gemini models
    - 'auto': Automatically detect based on available credentials
    """
    
    @staticmethod
    def create(
        provider: Optional[str] = None,
        **kwargs
    ) -> LLMService:
        """
        Create an LLM service instance.
        
        Args:
            provider: Provider name ('openai', 'gemini', or 'auto').
                     If None or 'auto', will auto-detect based on env vars.
            **kwargs: Additional arguments passed to the service constructor.
        
        Returns:
            An LLMService instance.
        
        Raises:
            ValueError: If provider is invalid or credentials are missing.
        """
        # Get provider from env var if not specified
        if provider is None:
            provider = os.getenv('LLM_PROVIDER', 'auto')
        
        # Auto-detect provider if set to 'auto'
        if provider.lower() == 'auto':
            provider = LLMServiceFactory._detect_provider()
        
        provider = provider.lower()
        
        if provider == 'openai':
            return LLMServiceFactory._create_openai_service(**kwargs)
        elif provider == 'gemini':
            return LLMServiceFactory._create_gemini_service(**kwargs)
        else:
            raise ValueError(
                f"Unknown LLM provider: {provider}. "
                f"Supported providers: 'openai', 'gemini', 'auto'"
            )
    
    @staticmethod
    def _detect_provider() -> str:
        """
        Auto-detect LLM provider based on available environment variables.
        
        Priority:
        1. OPENAI_API_KEY -> 'openai'
        2. GEMINI_API_KEY or GOOGLE_APPLICATION_CREDENTIALS -> 'gemini'
        
        Returns:
            Provider name ('openai' or 'gemini')
        
        Raises:
            ValueError: If no provider credentials are found.
        """
        # Check for OpenAI
        if os.getenv('OPENAI_API_KEY'):
            logger.info("Auto-detected LLM provider: OpenAI")
            return 'openai'
        
        # Check for Gemini
        if os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
            logger.info("Auto-detected LLM provider: Gemini")
            return 'gemini'
        
        raise ValueError(
            "No LLM provider credentials found. Set one of:\n"
            "  - OPENAI_API_KEY for OpenAI\n"
            "  - GEMINI_API_KEY or GOOGLE_APPLICATION_CREDENTIALS for Gemini\n"
            "Or explicitly specify provider via LLM_PROVIDER env var."
        )
    
    @staticmethod
    def _create_openai_service(**kwargs) -> LLMService:
        """Create an OpenAI service instance."""
        # Import here to avoid circular dependencies
        from .openai_service import OpenAIService
        
        # Extract OpenAI-specific kwargs
        api_key = kwargs.pop('api_key', None) or os.getenv('OPENAI_API_KEY')
        model = kwargs.pop('model', None) or os.getenv('OPENAI_MODEL', 'gpt-4o-mini') #'gpt-5-mini' 
        
        if not api_key:
            raise ValueError(
                "OpenAI API key not provided. Set OPENAI_API_KEY environment variable "
                "or pass api_key to factory."
            )
        
        logger.info(f"Creating OpenAI service with model: {model}")
        return OpenAIService(api_key=api_key, model=model, **kwargs)
    
    @staticmethod
    def _create_gemini_service(**kwargs) -> LLMService:
        """Create a Gemini service instance."""
        # Import here to avoid circular dependencies
        from .gemini_service import GeminiService
        
        # Extract Gemini-specific kwargs
        api_key = kwargs.pop('api_key', None) or os.getenv('GEMINI_API_KEY')
        service_account_path = kwargs.pop('service_account_path', None) or os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        project_id = kwargs.pop('project_id', None) or os.getenv('GOOGLE_CLOUD_PROJECT')
        location = kwargs.pop('location', 'us-central1')
        model = kwargs.pop('model', None) or os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')
        
        logger.info(f"Creating Gemini service with model: {model}")
        return GeminiService(
            api_key=api_key,
            service_account_path=service_account_path,
            project_id=project_id,
            location=location,
            model=model,
            **kwargs
        )

