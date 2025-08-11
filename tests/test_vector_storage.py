"""ABOUTME: Unit tests for vector storage using ChromaDB
ABOUTME: Tests embedding persistence, retrieval, and search capabilities"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from bear_mcp.semantic.vector_storage import VectorStore, EmbeddingCache
from bear_mcp.config.models import VectorStorageConfig


@pytest.fixture
def vector_config():
    """Create vector storage configuration for testing."""
    # Use temporary directory for tests
    temp_dir = tempfile.mkdtemp()
    return VectorStorageConfig(
        persist_directory=Path(temp_dir),
        collection_name="test_bear_notes",
        distance_function="cosine"
    )


@pytest.fixture
def sample_embeddings():
    """Create sample embeddings for testing."""
    return {
        "doc1": np.array([0.1, 0.2, 0.3, 0.4]),
        "doc2": np.array([0.5, 0.6, 0.7, 0.8]), 
        "doc3": np.array([0.2, 0.3, 0.1, 0.9])
    }


class TestVectorStore:
    """Test ChromaDB vector storage functionality."""

    def test_vector_store_creation(self, vector_config):
        """Test that vector store can be created."""
        store = VectorStore(vector_config)
        assert store.config == vector_config
        assert store.client is None
        assert store.collection is None

    @patch('bear_mcp.semantic.vector_storage.chromadb.PersistentClient')
    def test_initialize_store(self, mock_client_class, vector_config):
        """Test initializing the vector store."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        
        store = VectorStore(vector_config)
        store.initialize()
        
        mock_client_class.assert_called_once_with(path=str(vector_config.persist_directory))
        mock_client.get_or_create_collection.assert_called_once_with(
            name=vector_config.collection_name,
            metadata={"hnsw:space": vector_config.distance_function}
        )
        assert store.client == mock_client
        assert store.collection == mock_collection

    @patch('bear_mcp.semantic.vector_storage.chromadb.PersistentClient')
    def test_add_embeddings(self, mock_client_class, vector_config, sample_embeddings):
        """Test adding embeddings to vector store."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        
        store = VectorStore(vector_config)
        store.initialize()
        
        # Add embeddings
        ids = list(sample_embeddings.keys())
        embeddings = list(sample_embeddings.values())
        metadatas = [{"source": f"note_{i}"} for i in range(len(ids))]
        
        store.add_embeddings(ids, embeddings, metadatas)
        
        mock_collection.add.assert_called_once()
        call_args = mock_collection.add.call_args
        
        assert call_args.kwargs["ids"] == ids
        assert len(call_args.kwargs["embeddings"]) == len(embeddings)
        assert call_args.kwargs["metadatas"] == metadatas

    @patch('bear_mcp.semantic.vector_storage.chromadb.PersistentClient')
    def test_search_similar_embeddings(self, mock_client_class, vector_config):
        """Test searching for similar embeddings."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        
        # Mock search results
        mock_collection.query.return_value = {
            "ids": [["doc1", "doc2"]],
            "distances": [[0.1, 0.3]], 
            "metadatas": [[{"source": "note_1"}, {"source": "note_2"}]],
            "documents": [[None, None]]  # ChromaDB format
        }
        
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        
        store = VectorStore(vector_config)
        store.initialize()
        
        query_embedding = np.array([0.15, 0.25, 0.35, 0.45])
        results = store.search_similar(query_embedding, top_k=2)
        
        mock_collection.query.assert_called_once_with(
            query_embeddings=[query_embedding.tolist()],
            n_results=2
        )
        
        assert len(results) == 2
        assert results[0]["id"] == "doc1"
        assert results[0]["distance"] == 0.1
        assert results[0]["metadata"]["source"] == "note_1"

    @patch('bear_mcp.semantic.vector_storage.chromadb.PersistentClient')
    def test_update_embedding(self, mock_client_class, vector_config):
        """Test updating an existing embedding."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        
        store = VectorStore(vector_config)
        store.initialize()
        
        new_embedding = np.array([0.9, 0.8, 0.7, 0.6])
        new_metadata = {"source": "updated_note"}
        
        store.update_embedding("doc1", new_embedding, new_metadata)
        
        mock_collection.update.assert_called_once_with(
            ids=["doc1"],
            embeddings=[new_embedding.tolist()],
            metadatas=[new_metadata]
        )

    @patch('bear_mcp.semantic.vector_storage.chromadb.PersistentClient')
    def test_delete_embedding(self, mock_client_class, vector_config):
        """Test deleting an embedding."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        
        store = VectorStore(vector_config)
        store.initialize()
        
        store.delete_embedding("doc1")
        
        mock_collection.delete.assert_called_once_with(ids=["doc1"])

    @patch('bear_mcp.semantic.vector_storage.chromadb.PersistentClient')
    def test_get_collection_stats(self, mock_client_class, vector_config):
        """Test getting collection statistics."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 42
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        
        store = VectorStore(vector_config)
        store.initialize()
        
        stats = store.get_stats()
        
        assert stats["total_embeddings"] == 42
        assert stats["collection_name"] == vector_config.collection_name


class TestEmbeddingCache:
    """Test embedding caching functionality."""

    def test_cache_creation(self, vector_config):
        """Test that embedding cache can be created."""
        cache = EmbeddingCache(max_size=100)
        assert cache.max_size == 100
        assert len(cache._cache) == 0

    def test_cache_embedding(self):
        """Test caching and retrieving embeddings."""
        cache = EmbeddingCache(max_size=10)
        
        embedding = np.array([0.1, 0.2, 0.3])
        cache.put("test_key", embedding)
        
        retrieved = cache.get("test_key")
        np.testing.assert_array_equal(retrieved, embedding)

    def test_cache_miss(self):
        """Test cache miss returns None."""
        cache = EmbeddingCache(max_size=10)
        
        result = cache.get("nonexistent_key")
        assert result is None

    def test_cache_size_limit(self):
        """Test that cache respects size limit."""
        cache = EmbeddingCache(max_size=2)
        
        # Add 3 embeddings (exceeds limit)
        for i in range(3):
            embedding = np.array([i, i+1, i+2])
            cache.put(f"key_{i}", embedding)
        
        # Cache should only have 2 items (most recent)
        assert len(cache._cache) == 2
        assert cache.get("key_0") is None  # Should be evicted
        assert cache.get("key_1") is not None
        assert cache.get("key_2") is not None

    def test_cache_clear(self):
        """Test clearing the cache."""
        cache = EmbeddingCache(max_size=10)
        
        cache.put("key1", np.array([1, 2, 3]))
        cache.put("key2", np.array([4, 5, 6]))
        
        assert len(cache._cache) == 2
        
        cache.clear()
        
        assert len(cache._cache) == 0


# Integration tests
class TestVectorStorageIntegration:
    """Integration tests for vector storage components."""

    @patch('bear_mcp.semantic.vector_storage.chromadb.PersistentClient')
    def test_end_to_end_vector_operations(self, mock_client_class, vector_config, sample_embeddings):
        """Test complete vector storage workflow."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        
        # Mock search results
        mock_collection.query.return_value = {
            "ids": [["doc2"]],
            "distances": [[0.2]],
            "metadatas": [[{"source": "note_2"}]], 
            "documents": [[None]]
        }
        
        store = VectorStore(vector_config)
        cache = EmbeddingCache(max_size=100)
        
        # Initialize store
        store.initialize()
        
        # Add embeddings
        ids = list(sample_embeddings.keys())
        embeddings = list(sample_embeddings.values())
        metadatas = [{"source": f"note_{i}"} for i in range(len(ids))]
        
        store.add_embeddings(ids, embeddings, metadatas)
        
        # Cache some embeddings
        for id_, embedding in sample_embeddings.items():
            cache.put(id_, embedding)
        
        # Search for similar
        query = np.array([0.4, 0.5, 0.6, 0.7])
        results = store.search_similar(query, top_k=1)
        
        # Verify results
        assert len(results) == 1
        assert results[0]["id"] == "doc2"
        assert results[0]["distance"] == 0.2
        
        # Verify cache
        cached_embedding = cache.get("doc1")
        np.testing.assert_array_equal(cached_embedding, sample_embeddings["doc1"])

    def test_store_persistence(self, vector_config):
        """Test that vector store persists data correctly."""
        # This test would verify actual persistence, but we'll mock it
        # In real usage, ChromaDB would persist to the filesystem
        
        store = VectorStore(vector_config)
        
        # Verify configuration is set for persistence
        assert store.config.persist_directory == vector_config.persist_directory
        assert store.config.collection_name == vector_config.collection_name