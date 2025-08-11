"""ABOUTME: Unit tests for similarity computation and hybrid scoring algorithms
ABOUTME: Tests cosine similarity, hashtag matching, and search result ranking"""

import numpy as np
from unittest.mock import MagicMock, patch
import pytest

from bear_mcp.semantic.similarity import SimilarityEngine, HybridScorer, SearchEngine
from bear_mcp.config.models import PerformanceConfig
from bear_mcp.bear_db.models import BearNote
from datetime import datetime


@pytest.fixture
def performance_config():
    """Create performance configuration for testing."""
    return PerformanceConfig(
        similarity_threshold=0.7,
        max_related_notes=10,
        cache_size=100
    )


@pytest.fixture
def sample_notes():
    """Create sample Bear notes for testing."""
    return [
        BearNote(
            z_pk=1,
            zuniqueidentifier="NOTE-1",
            ztitle="Machine Learning Basics",
            ztext="# ML Overview\n\nMachine learning is a subset of AI. #ml #ai #tech",
            zcreationdate=725846400.0,
            zmodificationdate=725846400.0
        ),
        BearNote(
            z_pk=2,
            zuniqueidentifier="NOTE-2", 
            ztitle="Deep Learning Neural Networks",
            ztext="# Neural Networks\n\nDeep learning uses neural networks. #ml #deeplearning #neural",
            zcreationdate=725846500.0,
            zmodificationdate=725846500.0
        ),
        BearNote(
            z_pk=3,
            zuniqueidentifier="NOTE-3",
            ztitle="Cooking Recipes",
            ztext="# Pasta Recipe\n\nHow to make delicious pasta. #cooking #food #recipes",
            zcreationdate=725846600.0,
            zmodificationdate=725846600.0
        )
    ]


class TestSimilarityEngine:
    """Test similarity computation functionality."""

    def test_similarity_engine_creation(self, performance_config):
        """Test that similarity engine can be created."""
        engine = SimilarityEngine(performance_config)
        assert engine.config == performance_config
        assert engine._similarity_cache == {}

    def test_cosine_similarity_identical(self):
        """Test cosine similarity with identical vectors."""
        engine = SimilarityEngine(PerformanceConfig())
        
        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([1.0, 2.0, 3.0])
        
        similarity = engine.cosine_similarity(vec1, vec2)
        assert abs(similarity - 1.0) < 1e-6  # Should be 1.0 for identical vectors

    def test_cosine_similarity_orthogonal(self):
        """Test cosine similarity with orthogonal vectors."""
        engine = SimilarityEngine(PerformanceConfig())
        
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])
        
        similarity = engine.cosine_similarity(vec1, vec2)
        assert abs(similarity - 0.0) < 1e-6  # Should be 0.0 for orthogonal vectors

    def test_cosine_similarity_opposite(self):
        """Test cosine similarity with opposite vectors."""
        engine = SimilarityEngine(PerformanceConfig())
        
        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([-1.0, -2.0, -3.0])
        
        similarity = engine.cosine_similarity(vec1, vec2)
        assert abs(similarity - (-1.0)) < 1e-6  # Should be -1.0 for opposite vectors

    def test_batch_similarity_computation(self):
        """Test computing similarities for multiple embeddings."""
        engine = SimilarityEngine(PerformanceConfig())
        
        query = np.array([1.0, 0.0, 0.0])
        embeddings = np.array([
            [1.0, 0.0, 0.0],  # Identical
            [0.0, 1.0, 0.0],  # Orthogonal
            [0.5, 0.5, 0.0],  # 45 degrees
        ])
        
        similarities = engine.compute_similarities(query, embeddings)
        
        assert len(similarities) == 3
        assert abs(similarities[0] - 1.0) < 1e-6  # Identical
        assert abs(similarities[1] - 0.0) < 1e-6  # Orthogonal
        assert 0.0 < similarities[2] < 1.0  # 45 degrees

    def test_similarity_caching(self):
        """Test that similarity calculations are cached."""
        engine = SimilarityEngine(PerformanceConfig(cache_size=10))
        
        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([2.0, 3.0, 4.0])
        
        # First call should compute
        with patch.object(engine, '_compute_cosine_similarity', return_value=0.5) as mock_compute:
            result1 = engine.get_cached_similarity(vec1, vec2)
            mock_compute.assert_called_once()
        
        # Second call should use cache
        with patch.object(engine, '_compute_cosine_similarity', return_value=0.5) as mock_compute:
            result2 = engine.get_cached_similarity(vec1, vec2)
            mock_compute.assert_not_called()
        
        assert result1 == result2 == 0.5


class TestHybridScorer:
    """Test hybrid scoring functionality."""

    def test_hybrid_scorer_creation(self, performance_config):
        """Test that hybrid scorer can be created."""
        scorer = HybridScorer(performance_config)
        assert scorer.config == performance_config

    def test_extract_hashtags(self):
        """Test hashtag extraction from text."""
        scorer = HybridScorer(PerformanceConfig())
        
        text = "This is about #machinelearning and #ai. Also #deeplearning."
        hashtags = scorer.extract_hashtags(text)
        
        expected = ["machinelearning", "ai", "deeplearning"]
        assert set(hashtags) == set(expected)

    def test_hashtag_overlap_score(self):
        """Test hashtag overlap scoring."""
        scorer = HybridScorer(PerformanceConfig())
        
        hashtags1 = ["ml", "ai", "tech"]
        hashtags2 = ["ml", "ai", "data"]  # 2 out of 3 overlap
        
        score = scorer.hashtag_overlap_score(hashtags1, hashtags2)
        
        # Should be Jaccard similarity: intersection / union
        # intersection = {ml, ai} = 2, union = {ml, ai, tech, data} = 4
        expected_score = 2.0 / 4.0  # 0.5
        assert abs(score - expected_score) < 1e-6

    def test_hashtag_overlap_no_match(self):
        """Test hashtag overlap with no matches."""
        scorer = HybridScorer(PerformanceConfig())
        
        hashtags1 = ["ml", "ai"]
        hashtags2 = ["cooking", "food"]
        
        score = scorer.hashtag_overlap_score(hashtags1, hashtags2)
        assert score == 0.0

    def test_compute_hybrid_score(self):
        """Test computing hybrid score combining semantic and hashtag similarity."""
        scorer = HybridScorer(PerformanceConfig())
        
        # Mock semantic similarity
        semantic_score = 0.8
        
        # Mock hashtag data
        query_text = "Learning about #ml and #ai"
        candidate_text = "Machine learning and #ai are important. #tech"
        
        hybrid_score = scorer.compute_hybrid_score(
            semantic_score=semantic_score,
            query_text=query_text,
            candidate_text=candidate_text,
            semantic_weight=0.7,
            hashtag_weight=0.3
        )
        
        # Should combine semantic (0.8) and hashtag overlap
        assert 0.0 <= hybrid_score <= 1.0
        assert hybrid_score >= semantic_score * 0.7  # Minimum from semantic component


class TestSearchEngine:
    """Test search engine functionality."""

    def test_search_engine_creation(self, performance_config):
        """Test that search engine can be created."""
        # Mock dependencies
        mock_embedding_gen = MagicMock()
        mock_vector_store = MagicMock()
        mock_content_processor = MagicMock()
        
        engine = SearchEngine(
            performance_config,
            mock_embedding_gen,
            mock_vector_store,
            mock_content_processor
        )
        
        assert engine.config == performance_config
        assert engine.embedding_generator == mock_embedding_gen
        assert engine.vector_store == mock_vector_store
        assert engine.content_processor == mock_content_processor

    @patch('bear_mcp.semantic.similarity.SimilarityEngine')
    @patch('bear_mcp.semantic.similarity.HybridScorer')
    def test_semantic_search_basic(self, mock_scorer_class, mock_similarity_class, performance_config):
        """Test basic semantic search functionality."""
        # Setup mocks
        mock_embedding_gen = MagicMock()
        mock_vector_store = MagicMock()
        mock_content_processor = MagicMock()
        
        # Mock query embedding generation
        query_embedding = np.array([0.1, 0.2, 0.3])
        mock_embedding_gen.generate_embedding.return_value = query_embedding
        
        # Mock vector search results (both should be above threshold 0.7)
        mock_vector_store.search_similar.return_value = [
            {"id": "NOTE-1", "distance": 0.2, "metadata": {"title": "ML Basics"}},    # similarity = 0.8
            {"id": "NOTE-2", "distance": 0.1, "metadata": {"title": "Neural Networks"}} # similarity = 0.9
        ]
        
        engine = SearchEngine(
            performance_config,
            mock_embedding_gen,
            mock_vector_store,
            mock_content_processor
        )
        
        results = engine.semantic_search("machine learning query", top_k=2)
        
        # Verify calls
        mock_embedding_gen.generate_embedding.assert_called_once()
        mock_vector_store.search_similar.assert_called_once_with(query_embedding, top_k=2)
        
        # Verify results
        assert len(results) == 2
        assert results[0]["id"] == "NOTE-2"  # Should be first (highest similarity 0.9)
        assert results[0]["similarity"] == 1.0 - 0.1  # Convert distance to similarity
        assert results[1]["id"] == "NOTE-1"  # Should be second (similarity 0.8)
        assert results[1]["similarity"] == 1.0 - 0.2

    def test_filter_by_threshold(self, performance_config):
        """Test filtering results by similarity threshold."""
        mock_embedding_gen = MagicMock()
        mock_vector_store = MagicMock()
        mock_content_processor = MagicMock()
        
        engine = SearchEngine(
            performance_config,
            mock_embedding_gen,
            mock_vector_store,
            mock_content_processor
        )
        
        # Mock results with varying similarities
        raw_results = [
            {"id": "NOTE-1", "similarity": 0.9},  # Above threshold
            {"id": "NOTE-2", "similarity": 0.6},  # Below threshold
            {"id": "NOTE-3", "similarity": 0.8},  # Above threshold
        ]
        
        filtered = engine._filter_by_threshold(raw_results, threshold=0.7)
        
        assert len(filtered) == 2
        assert filtered[0]["id"] == "NOTE-1"
        assert filtered[1]["id"] == "NOTE-3"

    def test_rank_results(self, performance_config):
        """Test ranking search results by similarity score."""
        mock_embedding_gen = MagicMock()
        mock_vector_store = MagicMock()
        mock_content_processor = MagicMock()
        
        engine = SearchEngine(
            performance_config,
            mock_embedding_gen,
            mock_vector_store,
            mock_content_processor
        )
        
        # Mock unordered results
        results = [
            {"id": "NOTE-1", "similarity": 0.6},
            {"id": "NOTE-2", "similarity": 0.9},  # Should be first
            {"id": "NOTE-3", "similarity": 0.8},  # Should be second
        ]
        
        ranked = engine._rank_results(results)
        
        assert len(ranked) == 3
        assert ranked[0]["id"] == "NOTE-2"  # Highest similarity
        assert ranked[1]["id"] == "NOTE-3"  # Second highest
        assert ranked[2]["id"] == "NOTE-1"  # Lowest


# Integration tests
class TestSimilarityIntegration:
    """Integration tests for similarity and search components."""

    def test_end_to_end_similarity_pipeline(self, performance_config, sample_notes):
        """Test complete similarity computation pipeline."""
        # This would test the full pipeline but we'll mock the heavy components
        
        similarity_engine = SimilarityEngine(performance_config)
        hybrid_scorer = HybridScorer(performance_config)
        
        # Test similarity computation
        vec1 = np.array([0.1, 0.2, 0.3])
        vec2 = np.array([0.2, 0.3, 0.4])
        
        similarity = similarity_engine.cosine_similarity(vec1, vec2)
        assert 0.0 <= similarity <= 1.0
        
        # Test hybrid scoring
        query_text = "Machine learning and #ai"
        candidate_text = "Deep learning neural networks #ai #ml"
        
        hybrid_score = hybrid_scorer.compute_hybrid_score(
            semantic_score=similarity,
            query_text=query_text,
            candidate_text=candidate_text
        )
        
        assert 0.0 <= hybrid_score <= 1.0