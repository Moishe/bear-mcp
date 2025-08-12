"""ABOUTME: Unit tests for Ollama HTTP client integration
ABOUTME: Tests model management, request/response handling, and error scenarios"""

import json
from unittest.mock import MagicMock, patch, AsyncMock
import pytest
import aiohttp
from aiohttp import ClientError
from asyncio import TimeoutError

from bear_mcp.ai.ollama_client import OllamaClient, OllamaError, ModelNotFoundError
from bear_mcp.config.models import OllamaConfig


@pytest.fixture
def ollama_config():
    """Create Ollama configuration for testing."""
    return OllamaConfig(
        base_url="http://localhost:11434",
        model="llama2",
        timeout=30.0,
        max_retries=2
    )


class TestOllamaClient:
    """Test Ollama client functionality."""

    def test_client_creation(self, ollama_config):
        """Test that Ollama client can be created."""
        client = OllamaClient(ollama_config)
        assert client.config == ollama_config
        assert client.session is None
        assert not client._initialized

    @pytest.mark.asyncio
    async def test_client_initialization(self, ollama_config):
        """Test client initialization."""
        client = OllamaClient(ollama_config)
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = MagicMock()
            mock_session_class.return_value = mock_session
            
            await client.initialize()
            
            mock_session_class.assert_called_once_with(
                timeout=aiohttp.ClientTimeout(total=ollama_config.timeout)
            )
            assert client.session == mock_session
            assert client._initialized

    @pytest.mark.asyncio
    async def test_client_cleanup(self, ollama_config):
        """Test client cleanup."""
        client = OllamaClient(ollama_config)
        
        # Mock session
        mock_session = MagicMock()
        mock_session.close = AsyncMock()
        mock_session.closed = False
        client.session = mock_session
        client._initialized = True
        
        await client.cleanup()
        
        mock_session.close.assert_called_once()
        assert client.session is None
        assert not client._initialized

    @pytest.mark.asyncio
    async def test_health_check_success(self, ollama_config):
        """Test successful health check."""
        client = OllamaClient(ollama_config)
        
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"status": "ok"})
        
        # Create async context manager
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=None)
        mock_session.get.return_value = mock_cm
        
        client.session = mock_session
        client._initialized = True
        
        is_healthy = await client.health_check()
        
        assert is_healthy
        mock_session.get.assert_called_once_with(f"{ollama_config.base_url}/api/version")

    @pytest.mark.asyncio
    async def test_health_check_failure(self, ollama_config):
        """Test health check failure."""
        client = OllamaClient(ollama_config)
        
        mock_session = AsyncMock()
        mock_session.get.side_effect = ClientError("Connection failed")
        
        client.session = mock_session
        client._initialized = True
        
        is_healthy = await client.health_check()
        
        assert not is_healthy

    @pytest.mark.asyncio
    async def test_list_models_success(self, ollama_config):
        """Test successful model listing."""
        client = OllamaClient(ollama_config)
        
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "models": [
                {"name": "llama2", "size": 123456789},
                {"name": "codellama", "size": 987654321}
            ]
        })
        
        # Create async context manager
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=None)
        mock_session.get.return_value = mock_cm
        
        client.session = mock_session
        client._initialized = True
        
        models = await client.list_models()
        
        assert len(models) == 2
        assert models[0]["name"] == "llama2"
        assert models[1]["name"] == "codellama"
        mock_session.get.assert_called_once_with(f"{ollama_config.base_url}/api/tags")

    @pytest.mark.asyncio
    async def test_pull_model_success(self, ollama_config):
        """Test successful model pulling."""
        client = OllamaClient(ollama_config)
        
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"status": "success"})
        
        # Create async context manager
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=None)
        mock_session.post.return_value = mock_cm
        
        client.session = mock_session
        client._initialized = True
        
        await client.pull_model("llama2")
        
        mock_session.post.assert_called_once_with(
            f"{ollama_config.base_url}/api/pull",
            json={"name": "llama2"}
        )

    @pytest.mark.asyncio
    async def test_generate_response_success(self, ollama_config):
        """Test successful response generation."""
        client = OllamaClient(ollama_config)
        
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "response": "This is a test response.",
            "done": True
        })
        
        # Create async context manager
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=None)
        mock_session.post.return_value = mock_cm
        
        client.session = mock_session
        client._initialized = True
        
        response = await client.generate_response("Test prompt", model="llama2")
        
        assert response == "This is a test response."
        mock_session.post.assert_called_once_with(
            f"{ollama_config.base_url}/api/generate",
            json={
                "model": "llama2",
                "prompt": "Test prompt",
                "stream": False
            }
        )

    @pytest.mark.asyncio
    async def test_generate_response_with_options(self, ollama_config):
        """Test response generation with custom options."""
        client = OllamaClient(ollama_config)
        
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "response": "Custom response.",
            "done": True
        })
        
        # Create async context manager
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=None)
        mock_session.post.return_value = mock_cm
        
        client.session = mock_session
        client._initialized = True
        
        options = {"temperature": 0.8, "top_p": 0.9}
        response = await client.generate_response(
            "Test prompt", 
            model="codellama", 
            options=options
        )
        
        assert response == "Custom response."
        mock_session.post.assert_called_once_with(
            f"{ollama_config.base_url}/api/generate",
            json={
                "model": "codellama",
                "prompt": "Test prompt",
                "stream": False,
                "options": options
            }
        )

    @pytest.mark.asyncio
    async def test_generate_streaming_response(self, ollama_config):
        """Test streaming response generation."""
        client = OllamaClient(ollama_config)
        
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status = 200
        
        # Mock streaming response
        stream_data = [
            b'{"response": "Hello", "done": false}\n',
            b'{"response": " world", "done": false}\n', 
            b'{"response": "!", "done": true}\n'
        ]
        
        async def mock_iter():
            for chunk in stream_data:
                yield chunk
                
        mock_response.content.iter_chunked.return_value = mock_iter()
        
        # Create async context manager
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=None)
        mock_session.post.return_value = mock_cm
        
        client.session = mock_session
        client._initialized = True
        
        responses = []
        async for chunk in client.generate_streaming_response("Test prompt"):
            responses.append(chunk)
        
        assert responses == ["Hello", " world", "!"]
        mock_session.post.assert_called_once_with(
            f"{ollama_config.base_url}/api/generate",
            json={
                "model": "llama2",  # Default model
                "prompt": "Test prompt",
                "stream": True
            }
        )

    @pytest.mark.asyncio
    async def test_generate_response_model_not_found(self, ollama_config):
        """Test response generation with non-existent model."""
        client = OllamaClient(ollama_config)
        
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status = 404
        mock_response.json = AsyncMock(return_value={"error": "model not found"})
        
        # Create async context manager
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=None)
        mock_session.post.return_value = mock_cm
        
        client.session = mock_session
        client._initialized = True
        
        with pytest.raises(ModelNotFoundError, match="Model 'nonexistent' not found"):
            await client.generate_response("Test prompt", model="nonexistent")

    @pytest.mark.asyncio
    async def test_generate_response_server_error(self, ollama_config):
        """Test response generation with server error."""
        client = OllamaClient(ollama_config)
        
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal server error")
        
        # Create async context manager
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=None)
        mock_session.post.return_value = mock_cm
        
        client.session = mock_session
        client._initialized = True
        
        with pytest.raises(OllamaError, match="HTTP 500"):
            await client.generate_response("Test prompt")

    @pytest.mark.asyncio
    async def test_generate_response_network_error(self, ollama_config):
        """Test response generation with network error."""
        client = OllamaClient(ollama_config)
        
        mock_session = MagicMock()
        mock_session.post.side_effect = ClientError("Connection failed")
        
        client.session = mock_session
        client._initialized = True
        
        with pytest.raises(OllamaError, match="Network error"):
            await client.generate_response("Test prompt")

    @pytest.mark.asyncio
    async def test_generate_response_timeout(self, ollama_config):
        """Test response generation with timeout."""
        client = OllamaClient(ollama_config)
        
        mock_session = MagicMock()
        mock_session.post.side_effect = TimeoutError("Request timeout")
        
        client.session = mock_session
        client._initialized = True
        
        with pytest.raises(OllamaError, match="Request timeout"):
            await client.generate_response("Test prompt")

    def test_uninitialized_client_operations(self, ollama_config):
        """Test that uninitialized client raises errors."""
        client = OllamaClient(ollama_config)
        
        with pytest.raises(RuntimeError, match="Client not initialized"):
            import asyncio
            asyncio.run(client.health_check())
        
        with pytest.raises(RuntimeError, match="Client not initialized"):
            import asyncio
            asyncio.run(client.list_models())
        
        with pytest.raises(RuntimeError, match="Client not initialized"):
            import asyncio
            asyncio.run(client.generate_response("test"))


# Integration tests
class TestOllamaClientIntegration:
    """Integration tests for Ollama client."""

    @pytest.mark.asyncio
    async def test_full_workflow(self, ollama_config):
        """Test complete workflow with initialization and cleanup."""
        client = OllamaClient(ollama_config)
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = MagicMock()
            mock_session.close = AsyncMock()
            mock_session.closed = False
            mock_session_class.return_value = mock_session
            
            # Mock health check response
            mock_health_response = MagicMock()
            mock_health_response.status = 200
            mock_health_response.json = AsyncMock(return_value={"status": "ok"})
            
            mock_health_cm = AsyncMock()
            mock_health_cm.__aenter__ = AsyncMock(return_value=mock_health_response)
            mock_health_cm.__aexit__ = AsyncMock(return_value=None)
            mock_session.get.return_value = mock_health_cm
            
            # Mock generate response
            mock_gen_response = MagicMock()
            mock_gen_response.status = 200
            mock_gen_response.json = AsyncMock(return_value={
                "response": "Generated text",
                "done": True
            })
            
            mock_gen_cm = AsyncMock()
            mock_gen_cm.__aenter__ = AsyncMock(return_value=mock_gen_response)
            mock_gen_cm.__aexit__ = AsyncMock(return_value=None)
            mock_session.post.return_value = mock_gen_cm
            
            # Test workflow
            await client.initialize()
            assert client._initialized
            
            # Health check
            is_healthy = await client.health_check()
            assert is_healthy
            
            # Generate response
            response = await client.generate_response("Test prompt")
            assert response == "Generated text"
            
            # Cleanup
            await client.cleanup()
            assert not client._initialized
            assert client.session is None

    def test_error_classes(self):
        """Test custom error classes."""
        base_error = OllamaError("Test error")
        assert str(base_error) == "Test error"
        
        model_error = ModelNotFoundError("test-model")
        assert "test-model" in str(model_error)
        assert isinstance(model_error, OllamaError)