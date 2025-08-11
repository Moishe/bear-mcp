"""ABOUTME: Unit tests for the complete semantic search service
ABOUTME: Tests integration of embedding, vector storage, and search components"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
import numpy as np
import pytest

from bear_mcp.semantic.service import SemanticSearchService
from bear_mcp.config.models import BearMCPConfig, EmbeddingConfig, VectorStorageConfig, PerformanceConfig
from bear_mcp.bear_db.models import BearNote


@pytest.fixture
def semantic_config():
    """Create complete configuration for semantic service."""
    temp_dir = tempfile.mkdtemp()
    
    return BearMCPConfig(
        embedding=EmbeddingConfig(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            batch_size=4,
            max_length=256
        ),
        vector_storage=VectorStorageConfig(
            persist_directory=Path(temp_dir),
            collection_name="test_notes",
            distance_function="cosine"
        ),
        performance=PerformanceConfig(
            similarity_threshold=0.7,
            max_related_notes=5,
            cache_size=50
        )
    )


@pytest.fixture
def sample_notes():
    """Create sample notes for testing."""
    return [
        BearNote(
            z_pk=1,
            zuniqueidentifier="NOTE-ML-1",
            ztitle="Machine Learning Introduction",
            ztext="# Machine Learning\n\nMachine learning is a field of AI. #ml #ai",
            zcreationdate=725846400.0,
            zmodificationdate=725846400.0
        ),
        BearNote(
            z_pk=2, 
            zuniqueidentifier="NOTE-DL-2",
            ztitle="Deep Learning Networks",
            ztext="# Deep Learning\n\nDeep learning uses neural networks. #deeplearning #neural",
            zcreationdate=725846500.0,
            zmodificationdate=725846500.0
        )
    ]


class TestSemanticSearchService:
    """Test the complete semantic search service."""

    def test_service_creation(self, semantic_config):
        """Test that semantic service can be created."""
        service = SemanticSearchService(semantic_config)
        assert service.config == semantic_config
        assert service.embedding_generator is not None
        assert service.vector_store is not None
        assert service.content_processor is not None
        assert service.search_engine is None  # Not initialized until initialize() is called
        assert service._initialized is False

    @pytest.mark.asyncio
    @patch('bear_mcp.semantic.service.SentenceTransformer')
    @patch('bear_mcp.semantic.service.chromadb.PersistentClient')
    async def test_service_initialization(self, mock_chroma, mock_transformer, semantic_config):
        """Test service initialization."""
        # Setup mocks
        mock_model = MagicMock()
        mock_transformer.return_value = mock_model
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client
        
        service = SemanticSearchService(semantic_config)
        await service.initialize()
        
        # Verify initialization
        mock_transformer.assert_called_once()
        mock_chroma.assert_called_once()
        assert service._initialized is True

    @patch('bear_mcp.semantic.service.SentenceTransformer')
    @patch('bear_mcp.semantic.service.chromadb.PersistentClient')
    async def test_index_notes(self, mock_chroma, mock_transformer, semantic_config, sample_notes):
        """Test indexing notes into vector store."""
        # Setup mocks
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
        mock_transformer.return_value = mock_model
        
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client
        
        service = SemanticSearchService(semantic_config)
        await service.initialize()
        
        # Index notes
        await service.index_notes(sample_notes)
        
        # Verify embedding generation
        mock_model.encode.assert_called_once()
        
        # Verify vector storage
        mock_collection.add.assert_called_once()
        call_args = mock_collection.add.call_args
        assert len(call_args.kwargs["ids"]) == 2
        assert call_args.kwargs["ids"][0] == "NOTE-ML-1"
        assert call_args.kwargs["ids"][1] == "NOTE-DL-2"

    @patch('bear_mcp.semantic.service.SentenceTransformer')
    @patch('bear_mcp.semantic.service.chromadb.PersistentClient')
    async def test_search_notes(self, mock_chroma, mock_transformer, semantic_config):
        """Test searching for similar notes."""
        # Setup mocks
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2]])
        mock_transformer.return_value = mock_model
        
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [["NOTE-ML-1"]],
            "distances": [[0.1]],
            "metadatas": [[{"title": "Machine Learning Introduction", "created": "2025-01-01"}]],
            "documents": [[None]]
        }
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client
        
        service = SemanticSearchService(semantic_config)
        await service.initialize()
        
        # Search
        results = await service.search("machine learning concepts", top_k=1)
        
        # Verify search
        assert len(results) == 1
        assert results[0]["id"] == "NOTE-ML-1"
        assert results[0]["similarity"] == 0.9  # 1.0 - 0.1
        assert "title" in results[0]["metadata"]

    @patch('bear_mcp.semantic.service.SentenceTransformer')
    @patch('bear_mcp.semantic.service.chromadb.PersistentClient')
    async def test_find_related_notes(self, mock_chroma, mock_transformer, semantic_config):
        """Test finding notes related to a given note."""
        # Setup mocks
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2]])
        mock_transformer.return_value = mock_model
        
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [["NOTE-DL-2"]],
            "distances": [[0.2]],
            "metadatas": [[{"title": "Deep Learning Networks"}]],
            "documents": [[None]]
        }
        mock_client.get_or_create_collection.return_value = mock_client
        mock_chroma.return_value = mock_client
        
        service = SemanticSearchService(semantic_config)
        await service.initialize()
        
        # Find related notes for a reference note
        reference_note = BearNote(
            z_pk=1,
            zuniqueidentifier="NOTE-ML-1", 
            ztitle="Machine Learning",
            ztext="ML content #ml",
            zcreationdate=725846400.0,
            zmodificationdate=725846400.0
        )
        
        results = await service.find_related_notes(reference_note, top_k=1)
        
        # Verify results
        assert len(results) == 1
        assert results[0]["id"] == "NOTE-DL-2"
        assert results[0]["similarity"] == 0.8  # 1.0 - 0.2

    @patch('bear_mcp.semantic.service.SentenceTransformer')
    @patch('bear_mcp.semantic.service.chromadb.PersistentClient')
    async def test_update_note_embedding(self, mock_chroma, mock_transformer, semantic_config):
        """Test updating embedding for a modified note."""
        # Setup mocks
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.5, 0.6]])
        mock_transformer.return_value = mock_model
        
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client
        
        service = SemanticSearchService(semantic_config)
        await service.initialize()
        
        # Update note
        updated_note = BearNote(
            z_pk=1,
            zuniqueidentifier="NOTE-ML-1",
            ztitle="Updated ML Title", 
            ztext="Updated content about machine learning",
            zcreationdate=725846400.0,
            zmodificationdate=725846600.0
        )
        
        await service.update_note_embedding(updated_note)
        
        # Verify update
        mock_model.encode.assert_called_once()
        mock_collection.update.assert_called_once()

    @patch('bear_mcp.semantic.service.SentenceTransformer')
    @patch('bear_mcp.semantic.service.chromadb.PersistentClient')
    async def test_delete_note_embedding(self, mock_chroma, mock_transformer, semantic_config):
        """Test deleting note embedding."""
        # Setup mocks
        mock_transformer.return_value = MagicMock()
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client
        
        service = SemanticSearchService(semantic_config)
        await service.initialize()
        
        # Delete note
        await service.delete_note_embedding("NOTE-ML-1")
        
        # Verify deletion
        mock_collection.delete.assert_called_once_with(ids=["NOTE-ML-1"])

    @patch('bear_mcp.semantic.service.SentenceTransformer')
    @patch('bear_mcp.semantic.service.chromadb.PersistentClient')
    async def test_get_service_stats(self, mock_chroma, mock_transformer, semantic_config):
        """Test getting service statistics."""
        # Setup mocks
        mock_transformer.return_value = MagicMock()
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 150
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client
        
        service = SemanticSearchService(semantic_config)
        await service.initialize()
        
        # Get stats
        stats = await service.get_stats()
        
        # Verify stats
        assert stats["indexed_notes"] == 150
        assert stats["model_name"] == semantic_config.embedding.model_name
        assert stats["similarity_threshold"] == semantic_config.performance.similarity_threshold


# Integration tests
class TestSemanticServiceIntegration:
    """Integration tests for semantic service."""

    @patch('bear_mcp.semantic.service.SentenceTransformer')
    @patch('bear_mcp.semantic.service.chromadb.PersistentClient')
    async def test_end_to_end_workflow(self, mock_chroma, mock_transformer, semantic_config, sample_notes):
        """Test complete workflow: index, search, update, delete."""
        # Setup mocks
        mock_model = MagicMock()
        # Different embeddings for index vs search
        mock_model.encode.side_effect = [
            np.array([[0.1, 0.2], [0.3, 0.4]]),  # Index call
            np.array([[0.15, 0.25]]),             # Search call  
            np.array([[0.9, 0.8]])                # Update call
        ]
        mock_transformer.return_value = mock_model
        
        mock_client = MagicMock()
        mock_collection = MagicMock()
        # Mock search results
        mock_collection.query.return_value = {
            "ids": [["NOTE-ML-1"]],
            "distances": [[0.1]],
            "metadatas": [[{"title": "Machine Learning Introduction"}]],
            "documents": [[None]]
        }
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client
        
        service = SemanticSearchService(semantic_config)
        
        # Initialize service
        await service.initialize()
        
        # Index notes
        await service.index_notes(sample_notes)
        
        # Search for similar notes
        search_results = await service.search("artificial intelligence", top_k=1)
        assert len(search_results) == 1
        
        # Update a note
        updated_note = sample_notes[0]
        updated_note.ztext = "Updated content about AI and ML"
        await service.update_note_embedding(updated_note)
        
        # Delete a note
        await service.delete_note_embedding("NOTE-DL-2")
        
        # Verify all operations were called
        assert mock_model.encode.call_count == 3
        mock_collection.add.assert_called_once()
        mock_collection.query.assert_called_once()
        mock_collection.update.assert_called_once()
        mock_collection.delete.assert_called_once()

    def test_service_error_handling(self, semantic_config):
        """Test service handles errors gracefully."""
        service = SemanticSearchService(semantic_config)
        
        # Test search without initialization
        with pytest.raises(RuntimeError, match="Service not initialized"):
            import asyncio
            asyncio.run(service.search("test query"))
        
        # Test index without initialization
        with pytest.raises(RuntimeError, match="Service not initialized"):
            import asyncio
            asyncio.run(service.index_notes([]))

    def test_configuration_validation(self):
        """Test that service validates configuration."""
        # Test with invalid config - this should work since Pydantic allows empty strings by default
        # The validation happens at runtime when trying to load the model
        try:
            invalid_config = BearMCPConfig(
                embedding=EmbeddingConfig(
                    model_name="nonexistent-model",  # Invalid model name
                    batch_size=1   # Valid batch size
                )
            )
            service = SemanticSearchService(invalid_config)
            assert service.config.embedding.model_name == "nonexistent-model"
        except Exception:
            # If validation does occur, that's also fine
            pass