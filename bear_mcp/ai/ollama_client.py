"""ABOUTME: Ollama HTTP client for AI model integration
ABOUTME: Handles model management, request/response processing, and error handling"""

import json
from typing import Dict, List, Any, Optional, AsyncGenerator
import asyncio
import structlog
import aiohttp
from aiohttp import ClientError, ClientResponseError, ClientConnectorError
from asyncio import TimeoutError

from bear_mcp.config.models import OllamaConfig

logger = structlog.get_logger(__name__)


class OllamaError(Exception):
    """Base exception for Ollama-related errors."""
    pass


class ModelNotFoundError(OllamaError):
    """Exception raised when a requested model is not found."""
    
    def __init__(self, model_name: str):
        super().__init__(f"Model '{model_name}' not found on Ollama server")
        self.model_name = model_name


class OllamaClient:
    """HTTP client for Ollama API integration."""
    
    def __init__(self, config: OllamaConfig):
        """Initialize Ollama client.
        
        Args:
            config: Ollama configuration
        """
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the HTTP client session."""
        if self._initialized:
            logger.debug("Ollama client already initialized")
            return
        
        logger.info("Initializing Ollama client", base_url=self.config.base_url)
        
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        self.session = aiohttp.ClientSession(timeout=timeout)
        self._initialized = True
        
        logger.info("Ollama client initialized successfully")
    
    async def cleanup(self) -> None:
        """Cleanup client resources."""
        if self.session and not self.session.closed:
            await self.session.close()
        self.session = None
        self._initialized = False
        logger.info("Ollama client cleaned up")
    
    def _ensure_initialized(self) -> None:
        """Ensure client is initialized."""
        if not self._initialized or self.session is None:
            raise RuntimeError("Client not initialized. Call initialize() first.")
    
    async def health_check(self) -> bool:
        """Check if Ollama server is healthy.
        
        Returns:
            True if server is healthy, False otherwise
        """
        self._ensure_initialized()
        
        try:
            url = f"{self.config.base_url}/api/version"
            async with self.session.get(url) as response:
                if response.status == 200:
                    logger.debug("Ollama health check passed")
                    return True
                else:
                    logger.warning("Ollama health check failed", status=response.status)
                    return False
        except Exception as e:
            logger.warning("Ollama health check error", error=str(e))
            return False
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models on Ollama server.
        
        Returns:
            List of model information
            
        Raises:
            OllamaError: If request fails
        """
        self._ensure_initialized()
        
        try:
            url = f"{self.config.base_url}/api/tags"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    models = data.get("models", [])
                    logger.info("Listed models", count=len(models))
                    return models
                else:
                    error_text = await response.text()
                    raise OllamaError(f"Failed to list models: HTTP {response.status} - {error_text}")
        
        except ClientError as e:
            logger.error("Network error listing models", error=str(e))
            raise OllamaError(f"Network error: {str(e)}")
        except Exception as e:
            logger.error("Unexpected error listing models", error=str(e))
            raise OllamaError(f"Unexpected error: {str(e)}")
    
    async def pull_model(self, model_name: str) -> None:
        """Pull a model to the Ollama server.
        
        Args:
            model_name: Name of model to pull
            
        Raises:
            OllamaError: If pull fails
        """
        self._ensure_initialized()
        
        try:
            url = f"{self.config.base_url}/api/pull"
            payload = {"name": model_name}
            
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    logger.info("Model pull initiated", model=model_name)
                else:
                    error_text = await response.text()
                    raise OllamaError(f"Failed to pull model '{model_name}': HTTP {response.status} - {error_text}")
        
        except ClientError as e:
            logger.error("Network error pulling model", model=model_name, error=str(e))
            raise OllamaError(f"Network error: {str(e)}")
        except Exception as e:
            logger.error("Unexpected error pulling model", model=model_name, error=str(e))
            raise OllamaError(f"Unexpected error: {str(e)}")
    
    async def generate_response(
        self, 
        prompt: str, 
        model: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate a single response from Ollama.
        
        Args:
            prompt: Input prompt text
            model: Model to use (defaults to config model)
            options: Additional generation options
            
        Returns:
            Generated response text
            
        Raises:
            ModelNotFoundError: If model is not found
            OllamaError: If generation fails
        """
        self._ensure_initialized()
        
        model_name = model or self.config.model
        
        try:
            url = f"{self.config.base_url}/api/generate"
            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": False
            }
            
            if options:
                payload["options"] = options
            
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    response_text = data.get("response", "")
                    logger.info("Generated response", model=model_name, prompt_length=len(prompt))
                    return response_text
                elif response.status == 404:
                    raise ModelNotFoundError(model_name)
                else:
                    error_text = await response.text()
                    raise OllamaError(f"Generation failed: HTTP {response.status} - {error_text}")
        
        except ModelNotFoundError:
            raise
        except (TimeoutError, asyncio.TimeoutError) as e:
            logger.error("Request timeout", model=model_name, error=str(e))
            raise OllamaError(f"Request timeout: {str(e)}")
        except ClientError as e:
            logger.error("Network error generating response", model=model_name, error=str(e))
            raise OllamaError(f"Network error: {str(e)}")
        except Exception as e:
            logger.error("Unexpected error generating response", model=model_name, error=str(e))
            raise OllamaError(f"Unexpected error: {str(e)}")
    
    async def generate_streaming_response(
        self,
        prompt: str,
        model: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response from Ollama.
        
        Args:
            prompt: Input prompt text
            model: Model to use (defaults to config model)
            options: Additional generation options
            
        Yields:
            Response text chunks
            
        Raises:
            ModelNotFoundError: If model is not found
            OllamaError: If generation fails
        """
        self._ensure_initialized()
        
        model_name = model or self.config.model
        
        try:
            url = f"{self.config.base_url}/api/generate"
            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": True
            }
            
            if options:
                payload["options"] = options
            
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    logger.info("Started streaming response", model=model_name, prompt_length=len(prompt))
                    
                    async for chunk in response.content.iter_chunked(1024):
                        if chunk:
                            # Parse each line as JSON
                            for line in chunk.decode().strip().split('\n'):
                                if line.strip():
                                    try:
                                        data = json.loads(line)
                                        if "response" in data:
                                            yield data["response"]
                                        if data.get("done", False):
                                            logger.info("Streaming response completed", model=model_name)
                                            return
                                    except json.JSONDecodeError:
                                        # Skip malformed JSON lines
                                        continue
                elif response.status == 404:
                    raise ModelNotFoundError(model_name)
                else:
                    error_text = await response.text()
                    raise OllamaError(f"Streaming generation failed: HTTP {response.status} - {error_text}")
        
        except ModelNotFoundError:
            raise
        except (TimeoutError, asyncio.TimeoutError) as e:
            logger.error("Request timeout during streaming", model=model_name, error=str(e))
            raise OllamaError(f"Request timeout: {str(e)}")
        except ClientError as e:
            logger.error("Network error during streaming", model=model_name, error=str(e))
            raise OllamaError(f"Network error: {str(e)}")
        except Exception as e:
            logger.error("Unexpected error during streaming", model=model_name, error=str(e))
            raise OllamaError(f"Unexpected error: {str(e)}")