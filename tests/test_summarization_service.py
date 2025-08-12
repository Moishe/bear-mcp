"""ABOUTME: Unit tests for AI summarization service integration
ABOUTME: Tests note summarization, streaming responses, and error handling"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from typing import AsyncGenerator

from bear_mcp.ai.summarization_service import (
    SummarizationService,
    SummarizationError,
    SummaryRequest,
    SummaryResponse,
    SummaryStyle
)
from bear_mcp.config.models import BearMCPConfig, OllamaConfig
from bear_mcp.bear_db.models import BearNote


@pytest.fixture
def config():
    """Create configuration for testing."""
    return BearMCPConfig(
        ollama=OllamaConfig(
            base_url="http://localhost:11434",
            model="llama2",
            timeout=30.0,
            max_retries=2
        )
    )


@pytest.fixture
def sample_note():
    """Create sample Bear note for testing."""
    return BearNote(
        z_pk=1,
        zuniqueidentifier="NOTE-123",
        ztitle="Machine Learning Introduction",
        ztext="# Machine Learning\n\nMachine learning is a field of AI that focuses on algorithms and statistical models. #ml #ai\n\nKey concepts:\n- Supervised learning\n- Unsupervised learning\n- Deep learning\n\nApplications include image recognition, natural language processing, and recommendation systems.",
        zcreationdate=725846400.0,
        zmodificationdate=725846400.0
    )


@pytest.fixture
def sample_notes(sample_note):
    """Create multiple sample notes."""
    note2 = BearNote(
        z_pk=2,
        zuniqueidentifier="NOTE-456",
        ztitle="Deep Learning Networks",
        ztext="# Deep Learning\n\nDeep learning uses neural networks with multiple layers. #deeplearning #neural\n\nArchitectures:\n- CNNs for image processing\n- RNNs for sequence data\n- Transformers for language",
        zcreationdate=725846500.0,
        zmodificationdate=725846500.0
    )
    return [sample_note, note2]


class TestSummaryRequest:
    """Test summary request model."""

    def test_request_creation_single_note(self, sample_note):
        """Test creating summary request for single note."""
        request = SummaryRequest(
            note=sample_note,
            style=SummaryStyle.BRIEF,
            max_length=50
        )
        
        assert request.note == sample_note
        assert request.style == SummaryStyle.BRIEF
        assert request.max_length == 50
        assert request.notes is None

    def test_request_creation_multi_notes(self, sample_notes):
        """Test creating summary request for multiple notes."""
        request = SummaryRequest(
            notes=sample_notes,
            style=SummaryStyle.STRUCTURED,
            theme="AI and Machine Learning"
        )
        
        assert request.notes == sample_notes
        assert request.style == SummaryStyle.STRUCTURED
        assert request.theme == "AI and Machine Learning"
        assert request.note is None

    def test_request_validation_error(self):
        """Test request validation with invalid input."""
        with pytest.raises(ValueError, match="Either note or notes must be provided"):
            SummaryRequest(style=SummaryStyle.BRIEF)

    def test_request_both_note_and_notes_error(self, sample_note, sample_notes):
        """Test error when both note and notes are provided."""
        with pytest.raises(ValueError, match="Cannot provide both note and notes"):
            SummaryRequest(
                note=sample_note,
                notes=sample_notes,
                style=SummaryStyle.BRIEF
            )


class TestSummarizationService:
    """Test summarization service functionality."""

    def test_service_creation(self, config):
        """Test that summarization service can be created."""
        service = SummarizationService(config)
        assert service.config == config
        assert service.ollama_client is not None
        assert service.template_manager is not None
        assert not service._initialized

    @pytest.mark.asyncio
    async def test_service_initialization(self, config):
        """Test service initialization."""
        service = SummarizationService(config)
        
        with patch.object(service.ollama_client, 'initialize') as mock_init:
            mock_init.return_value = AsyncMock()
            
            await service.initialize()
            
            mock_init.assert_called_once()
            assert service._initialized

    @pytest.mark.asyncio
    async def test_service_cleanup(self, config):
        """Test service cleanup."""
        service = SummarizationService(config)
        service._initialized = True
        
        with patch.object(service.ollama_client, 'cleanup') as mock_cleanup:
            mock_cleanup.return_value = AsyncMock()
            
            await service.cleanup()
            
            mock_cleanup.assert_called_once()
            assert not service._initialized

    @pytest.mark.asyncio
    async def test_summarize_note_brief(self, config, sample_note):
        """Test brief note summarization."""
        service = SummarizationService(config)
        service._initialized = True
        
        # Mock Ollama response
        mock_response = "Machine learning is a field of AI focusing on algorithms and statistical models for tasks like image recognition and NLP."
        
        with patch.object(service.ollama_client, 'generate_response', return_value=mock_response) as mock_generate:
            request = SummaryRequest(
                note=sample_note,
                style=SummaryStyle.BRIEF,
                max_length=30
            )
            
            response = await service.summarize(request)
            
            # Verify response
            assert isinstance(response, SummaryResponse)
            assert response.summary == mock_response
            assert response.style == SummaryStyle.BRIEF
            assert response.note_ids == [sample_note.zuniqueidentifier]
            
            # Verify Ollama was called
            mock_generate.assert_called_once()
            call_args = mock_generate.call_args
            assert sample_note.ztitle in call_args[0][0]  # prompt contains title
            assert sample_note.ztext in call_args[0][0]   # prompt contains content

    @pytest.mark.asyncio
    async def test_summarize_note_detailed(self, config, sample_note):
        """Test detailed note summarization."""
        service = SummarizationService(config)
        service._initialized = True
        
        mock_response = "Machine learning is a comprehensive field of artificial intelligence. It focuses on algorithms and statistical models. Key concepts include supervised, unsupervised, and deep learning approaches."
        
        with patch.object(service.ollama_client, 'generate_response', return_value=mock_response) as mock_generate:
            request = SummaryRequest(
                note=sample_note,
                style=SummaryStyle.DETAILED,
                max_length=5  # sentences
            )
            
            response = await service.summarize(request)
            
            assert response.summary == mock_response
            assert response.style == SummaryStyle.DETAILED
            mock_generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_summarize_note_structured(self, config, sample_note):
        """Test structured note summarization."""
        service = SummarizationService(config)
        service._initialized = True
        
        mock_response = """## Main Topic
Machine Learning and AI

## Key Points
- Supervised and unsupervised learning
- Deep learning applications
- Image recognition and NLP

## Tags
#ml #ai"""
        
        with patch.object(service.ollama_client, 'generate_response', return_value=mock_response) as mock_generate:
            request = SummaryRequest(
                note=sample_note,
                style=SummaryStyle.STRUCTURED,
                sections=["Main Topic", "Key Points", "Tags"]
            )
            
            response = await service.summarize(request)
            
            assert response.summary == mock_response
            assert response.style == SummaryStyle.STRUCTURED
            mock_generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_summarize_multiple_notes(self, config, sample_notes):
        """Test multi-note summarization."""
        service = SummarizationService(config)
        service._initialized = True
        
        mock_response = "Both notes cover machine learning and deep learning concepts. They discuss supervised learning, neural networks, and applications in AI."
        
        with patch.object(service.ollama_client, 'generate_response', return_value=mock_response) as mock_generate:
            request = SummaryRequest(
                notes=sample_notes,
                style=SummaryStyle.BRIEF,
                theme="AI Technologies",
                max_length=3  # paragraphs
            )
            
            response = await service.summarize(request)
            
            assert response.summary == mock_response
            assert response.style == SummaryStyle.BRIEF
            assert len(response.note_ids) == 2
            assert sample_notes[0].zuniqueidentifier in response.note_ids
            assert sample_notes[1].zuniqueidentifier in response.note_ids
            mock_generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_summarize_streaming(self, config, sample_note):
        """Test streaming summarization."""
        service = SummarizationService(config)
        service._initialized = True
        
        # Mock streaming response
        async def mock_streaming_response(prompt, **kwargs):
            yield "Machine learning"
            yield " is a field"
            yield " of AI that"
            yield " focuses on algorithms."
        
        with patch.object(service.ollama_client, 'generate_streaming_response', side_effect=mock_streaming_response) as mock_stream:
            request = SummaryRequest(
                note=sample_note,
                style=SummaryStyle.BRIEF,
                max_length=20
            )
            
            chunks = []
            async for chunk in service.summarize_streaming(request):
                chunks.append(chunk)
            
            assert chunks == ["Machine learning", " is a field", " of AI that", " focuses on algorithms."]
            mock_stream.assert_called_once()

    @pytest.mark.asyncio
    async def test_extract_keywords(self, config, sample_note):
        """Test keyword extraction."""
        service = SummarizationService(config)
        service._initialized = True
        
        mock_response = "machine learning, artificial intelligence, algorithms, supervised learning, neural networks"
        
        with patch.object(service.ollama_client, 'generate_response', return_value=mock_response) as mock_generate:
            keywords = await service.extract_keywords(sample_note, max_keywords=5)
            
            assert keywords == ["machine learning", "artificial intelligence", "algorithms", "supervised learning", "neural networks"]
            mock_generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check(self, config):
        """Test service health check."""
        service = SummarizationService(config)
        service._initialized = True
        
        with patch.object(service.ollama_client, 'health_check', return_value=True) as mock_health:
            is_healthy = await service.health_check()
            
            assert is_healthy
            mock_health.assert_called_once()

    @pytest.mark.asyncio
    async def test_summarize_uninitialized_service(self, config, sample_note):
        """Test that uninitialized service raises error."""
        service = SummarizationService(config)
        
        request = SummaryRequest(note=sample_note, style=SummaryStyle.BRIEF)
        
        with pytest.raises(SummarizationError, match="Service not initialized"):
            await service.summarize(request)

    @pytest.mark.asyncio
    async def test_summarize_ollama_error(self, config, sample_note):
        """Test handling Ollama errors during summarization."""
        service = SummarizationService(config)
        service._initialized = True
        
        with patch.object(service.ollama_client, 'generate_response', side_effect=Exception("Ollama connection failed")) as mock_generate:
            request = SummaryRequest(note=sample_note, style=SummaryStyle.BRIEF)
            
            with pytest.raises(SummarizationError, match="Summarization failed"):
                await service.summarize(request)

    @pytest.mark.asyncio
    async def test_get_service_stats(self, config):
        """Test getting service statistics."""
        service = SummarizationService(config)
        service._initialized = True
        
        # Mock some service usage
        service._summary_count = 42
        service._total_tokens_processed = 15000
        
        with patch.object(service.ollama_client, 'health_check', return_value=True):
            stats = await service.get_stats()
            
            assert stats["summaries_generated"] == 42
            assert stats["total_tokens_processed"] == 15000
            assert stats["model_name"] == config.ollama.model
            assert stats["is_healthy"] is True


# Integration tests
class TestSummarizationServiceIntegration:
    """Integration tests for summarization service."""

    @pytest.mark.asyncio
    async def test_full_workflow(self, config, sample_note):
        """Test complete workflow from initialization to summarization."""
        service = SummarizationService(config)
        
        # Mock all dependencies
        with patch.object(service.ollama_client, 'initialize') as mock_init, \
             patch.object(service.ollama_client, 'generate_response', return_value="Generated summary") as mock_generate, \
             patch.object(service.ollama_client, 'cleanup') as mock_cleanup:
            
            # Initialize
            await service.initialize()
            
            # Summarize
            request = SummaryRequest(note=sample_note, style=SummaryStyle.BRIEF)
            response = await service.summarize(request)
            
            assert response.summary == "Generated summary"
            
            # Cleanup
            await service.cleanup()
            
            mock_init.assert_called_once()
            mock_generate.assert_called_once()
            mock_cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_error_recovery(self, config, sample_note):
        """Test service error recovery."""
        service = SummarizationService(config)
        service._initialized = True
        
        # First call fails
        with patch.object(service.ollama_client, 'generate_response', side_effect=Exception("Network error")):
            request = SummaryRequest(note=sample_note, style=SummaryStyle.BRIEF)
            
            with pytest.raises(SummarizationError):
                await service.summarize(request)
        
        # Second call succeeds
        with patch.object(service.ollama_client, 'generate_response', return_value="Recovery summary"):
            response = await service.summarize(request)
            assert response.summary == "Recovery summary"

    def test_summary_response_attributes(self, sample_note):
        """Test summary response model."""
        response = SummaryResponse(
            summary="Test summary",
            style=SummaryStyle.BRIEF,
            note_ids=["NOTE-123"],
            model_used="llama2",
            tokens_processed=150,
            generation_time=2.5
        )
        
        assert response.summary == "Test summary"
        assert response.style == SummaryStyle.BRIEF
        assert response.note_ids == ["NOTE-123"]
        assert response.model_used == "llama2"
        assert response.tokens_processed == 150
        assert response.generation_time == 2.5