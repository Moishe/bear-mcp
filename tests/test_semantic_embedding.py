"""ABOUTME: Unit tests for semantic embedding pipeline and content processing
ABOUTME: Tests sentence-transformers integration, text preprocessing, and batch processing"""

import numpy as np
from unittest.mock import MagicMock, patch
import pytest

from bear_mcp.semantic.embedding import EmbeddingGenerator, TextPreprocessor
from bear_mcp.semantic.content_processing import ContentProcessor
from bear_mcp.config.models import EmbeddingConfig


@pytest.fixture
def embedding_config():
    """Create embedding configuration for testing."""
    return EmbeddingConfig(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        batch_size=4,
        max_length=256,
        cache_dir=None
    )


class TestTextPreprocessor:
    """Test text preprocessing functionality."""

    def test_preprocessor_creation(self):
        """Test that text preprocessor can be created."""
        preprocessor = TextPreprocessor()
        assert preprocessor is not None

    def test_clean_markdown_text(self):
        """Test cleaning markdown formatting from text."""
        preprocessor = TextPreprocessor()
        
        markdown_text = """# Title
        
        This is **bold** and *italic* text.
        
        - List item 1
        - List item 2
        
        [Link](https://example.com)
        
        `code snippet`
        """
        
        cleaned = preprocessor.clean_markdown(markdown_text)
        
        # Should remove markdown formatting but keep content
        assert "Title" in cleaned
        assert "bold" in cleaned
        assert "italic" in cleaned
        assert "List item 1" in cleaned
        assert "**" not in cleaned
        assert "##" not in cleaned
        assert "[Link]" not in cleaned

    def test_normalize_text(self):
        """Test text normalization."""
        preprocessor = TextPreprocessor()
        
        text = "  This   has   extra    whitespace\n\n\nand\t\ttabs  "
        normalized = preprocessor.normalize_text(text)
        
        # Should normalize whitespace
        assert normalized == "This has extra whitespace and tabs"

    def test_chunk_text_short(self):
        """Test chunking short text that doesn't need splitting."""
        preprocessor = TextPreprocessor()
        
        short_text = "This is a short text."
        chunks = preprocessor.chunk_text(short_text, max_chunk_size=100)
        
        assert len(chunks) == 1
        assert chunks[0] == short_text

    def test_chunk_text_long(self):
        """Test chunking long text into multiple chunks."""
        preprocessor = TextPreprocessor()
        
        # Create text longer than max_chunk_size
        long_text = " ".join(["word"] * 100)  # 100 words
        chunks = preprocessor.chunk_text(long_text, max_chunk_size=50)  # Force chunking
        
        assert len(chunks) > 1
        # Each chunk should be within size limit
        for chunk in chunks:
            assert len(chunk.split()) <= 50


class TestEmbeddingGenerator:
    """Test embedding generation functionality."""

    def test_generator_creation(self, embedding_config):
        """Test that embedding generator can be created."""
        generator = EmbeddingGenerator(embedding_config)
        assert generator.config == embedding_config

    @patch('bear_mcp.semantic.embedding.SentenceTransformer')
    def test_model_loading(self, mock_transformer, embedding_config):
        """Test that sentence transformer model is loaded correctly."""
        mock_model = MagicMock()
        mock_transformer.return_value = mock_model
        
        generator = EmbeddingGenerator(embedding_config)
        generator.load_model()
        
        mock_transformer.assert_called_once_with(embedding_config.model_name)
        assert generator.model == mock_model

    @patch('bear_mcp.semantic.embedding.SentenceTransformer')
    def test_single_text_embedding(self, mock_transformer, embedding_config):
        """Test generating embedding for a single text."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_transformer.return_value = mock_model
        
        generator = EmbeddingGenerator(embedding_config)
        generator.load_model()
        
        text = "This is a test text."
        embedding = generator.generate_embedding(text)
        
        mock_model.encode.assert_called_once_with([text])
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (3,)  # Should return 1D array for single text

    @patch('bear_mcp.semantic.embedding.SentenceTransformer')  
    def test_batch_text_embedding(self, mock_transformer, embedding_config):
        """Test generating embeddings for multiple texts in batch."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mock_transformer.return_value = mock_model
        
        generator = EmbeddingGenerator(embedding_config)
        generator.load_model()
        
        texts = ["First text", "Second text"]
        embeddings = generator.generate_embeddings_batch(texts)
        
        mock_model.encode.assert_called_once_with(texts)
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (2, 3)  # 2 texts, 3 dimensions

    def test_embedding_caching(self, embedding_config):
        """Test that embeddings can be cached."""
        generator = EmbeddingGenerator(embedding_config)
        
        text = "Test text for caching"
        # Mock embedding result
        fake_embedding = np.array([0.1, 0.2, 0.3])
        
        # First call should compute and cache
        with patch.object(generator, '_compute_embedding', return_value=fake_embedding) as mock_compute:
            result1 = generator.get_cached_embedding(text)
            mock_compute.assert_called_once()
        
        # Second call should use cache
        with patch.object(generator, '_compute_embedding', return_value=fake_embedding) as mock_compute:
            result2 = generator.get_cached_embedding(text)
            mock_compute.assert_not_called()  # Should not compute again
        
        np.testing.assert_array_equal(result1, result2)


class TestContentProcessor:
    """Test content processing for Bear notes."""

    def test_processor_creation(self):
        """Test that content processor can be created."""
        processor = ContentProcessor()
        assert processor is not None

    def test_extract_text_from_note(self):
        """Test extracting clean text from a Bear note."""
        from bear_mcp.bear_db.models import BearNote
        from datetime import datetime
        
        # Create test note with markdown content
        note = BearNote(
            z_pk=1,
            zuniqueidentifier="TEST-123",
            ztitle="Test Note",
            ztext="# My Note\n\nThis has **bold** text and [links](http://example.com).",
            zcreationdate=725846400.0,
            zmodificationdate=725846400.0
        )
        
        processor = ContentProcessor()
        clean_text = processor.extract_clean_text(note)
        
        # Should extract clean text without markdown
        assert "My Note" in clean_text
        assert "bold" in clean_text
        assert "**" not in clean_text
        assert "[links]" not in clean_text

    def test_extract_keywords_tfidf(self):
        """Test keyword extraction using TF-IDF."""
        processor = ContentProcessor()
        
        texts = [
            "Machine learning and artificial intelligence are transforming technology",
            "Natural language processing is a subset of machine learning",
            "Deep learning networks use artificial neural networks"
        ]
        
        keywords = processor.extract_keywords_tfidf(texts, top_k=5)
        
        assert len(keywords) <= 5
        assert isinstance(keywords, list)
        # Should find relevant keywords
        important_terms = ["machine", "learning", "artificial", "intelligence", "neural"]
        found_important = any(term in " ".join(keywords).lower() for term in important_terms)
        assert found_important

    def test_process_note_for_embedding(self):
        """Test full processing pipeline for a note."""
        from bear_mcp.bear_db.models import BearNote
        
        note = BearNote(
            z_pk=1,
            zuniqueidentifier="TEST-123", 
            ztitle="Test Note",
            ztext="# Important Note\n\nThis contains **important** information about machine learning.",
            zcreationdate=725846400.0,
            zmodificationdate=725846400.0
        )
        
        processor = ContentProcessor()
        processed = processor.process_note_for_embedding(note)
        
        # Should return processed text ready for embedding
        assert isinstance(processed, str)
        assert "Important Note" in processed
        assert "important" in processed
        assert "machine learning" in processed
        assert "**" not in processed  # Markdown removed


# Integration tests
class TestSemanticIntegration:
    """Integration tests for semantic pipeline components."""

    @patch('bear_mcp.semantic.embedding.SentenceTransformer')
    def test_end_to_end_embedding_pipeline(self, mock_transformer, embedding_config):
        """Test complete pipeline from note to embedding."""
        from bear_mcp.bear_db.models import BearNote
        
        # Setup mocks
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_transformer.return_value = mock_model
        
        # Create test note
        note = BearNote(
            z_pk=1,
            zuniqueidentifier="TEST-123",
            ztitle="Machine Learning",
            ztext="# ML Concepts\n\nMachine learning uses **algorithms** to learn patterns.",
            zcreationdate=725846400.0,
            zmodificationdate=725846400.0
        )
        
        # Process through pipeline
        processor = ContentProcessor()
        generator = EmbeddingGenerator(embedding_config)
        generator.load_model()
        
        # Extract and process text
        clean_text = processor.extract_clean_text(note)
        processed_text = processor.process_note_for_embedding(note)
        
        # Generate embedding
        embedding = generator.generate_embedding(processed_text)
        
        # Verify results
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (3,)
        mock_model.encode.assert_called()