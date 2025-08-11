"""ABOUTME: Similarity computation and hybrid scoring for semantic search
ABOUTME: Handles cosine similarity, hashtag matching, result ranking and search engine logic"""

import re
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from collections import OrderedDict
import numpy as np

from bear_mcp.config.models import PerformanceConfig


class SimilarityEngine:
    """Computes similarity between embeddings with caching."""
    
    def __init__(self, config: PerformanceConfig):
        """Initialize similarity engine.
        
        Args:
            config: Performance configuration
        """
        self.config = config
        self._similarity_cache: Dict[str, float] = {}
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score (-1 to 1)
        """
        # Normalize vectors
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Compute dot product and normalize
        dot_product = np.dot(vec1, vec2)
        similarity = dot_product / (norm1 * norm2)
        
        return float(similarity)
    
    def compute_similarities(self, query: np.ndarray, embeddings: np.ndarray) -> List[float]:
        """Compute similarities between query and multiple embeddings.
        
        Args:
            query: Query embedding vector
            embeddings: Matrix of embedding vectors (n_embeddings x embedding_dim)
            
        Returns:
            List of similarity scores
        """
        similarities = []
        for embedding in embeddings:
            similarity = self.cosine_similarity(query, embedding)
            similarities.append(similarity)
        
        return similarities
    
    def _compute_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Internal method to compute cosine similarity (for testing)."""
        return self.cosine_similarity(vec1, vec2)
    
    def _create_cache_key(self, vec1: np.ndarray, vec2: np.ndarray) -> str:
        """Create cache key for vector pair."""
        # Create deterministic key from vector content
        vec1_str = ','.join(map(str, vec1.flatten()))
        vec2_str = ','.join(map(str, vec2.flatten()))
        combined = f"{vec1_str}|{vec2_str}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get_cached_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Get similarity with caching support.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cached or newly computed similarity
        """
        cache_key = self._create_cache_key(vec1, vec2)
        
        if cache_key in self._similarity_cache:
            return self._similarity_cache[cache_key]
        
        similarity = self._compute_cosine_similarity(vec1, vec2)
        
        # Add to cache (with size limit)
        if len(self._similarity_cache) >= self.config.cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self._similarity_cache))
            del self._similarity_cache[oldest_key]
        
        self._similarity_cache[cache_key] = similarity
        return similarity


class HybridScorer:
    """Combines semantic and metadata-based scoring."""
    
    def __init__(self, config: PerformanceConfig):
        """Initialize hybrid scorer.
        
        Args:
            config: Performance configuration
        """
        self.config = config
    
    def extract_hashtags(self, text: str) -> List[str]:
        """Extract hashtags from text.
        
        Args:
            text: Text to extract hashtags from
            
        Returns:
            List of hashtag strings (without # symbol)
        """
        if not text:
            return []
        
        # Find all hashtags (word characters after #)
        hashtags = re.findall(r'#(\w+)', text.lower())
        return hashtags
    
    def hashtag_overlap_score(self, hashtags1: List[str], hashtags2: List[str]) -> float:
        """Compute Jaccard similarity between hashtag sets.
        
        Args:
            hashtags1: First set of hashtags
            hashtags2: Second set of hashtags
            
        Returns:
            Jaccard similarity (0 to 1)
        """
        if not hashtags1 and not hashtags2:
            return 0.0
        
        set1 = set(hashtags1)
        set2 = set(hashtags2)
        
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        
        if len(union) == 0:
            return 0.0
        
        return len(intersection) / len(union)
    
    def compute_hybrid_score(
        self,
        semantic_score: float,
        query_text: str,
        candidate_text: str,
        semantic_weight: float = 0.8,
        hashtag_weight: float = 0.2
    ) -> float:
        """Compute hybrid score combining semantic and hashtag similarity.
        
        Args:
            semantic_score: Semantic similarity score (0 to 1)
            query_text: Query text
            candidate_text: Candidate text
            semantic_weight: Weight for semantic component
            hashtag_weight: Weight for hashtag component
            
        Returns:
            Combined hybrid score (0 to 1)
        """
        # Extract hashtags
        query_hashtags = self.extract_hashtags(query_text)
        candidate_hashtags = self.extract_hashtags(candidate_text)
        
        # Compute hashtag overlap
        hashtag_score = self.hashtag_overlap_score(query_hashtags, candidate_hashtags)
        
        # Combine scores
        hybrid_score = (
            semantic_weight * semantic_score +
            hashtag_weight * hashtag_score
        )
        
        return min(hybrid_score, 1.0)  # Cap at 1.0


class SearchEngine:
    """Main search engine combining all components."""
    
    def __init__(
        self,
        config: PerformanceConfig,
        embedding_generator,
        vector_store,
        content_processor
    ):
        """Initialize search engine.
        
        Args:
            config: Performance configuration
            embedding_generator: Embedding generation component
            vector_store: Vector storage component
            content_processor: Content processing component
        """
        self.config = config
        self.embedding_generator = embedding_generator
        self.vector_store = vector_store
        self.content_processor = content_processor
        
        # Initialize sub-components
        self.similarity_engine = SimilarityEngine(config)
        self.hybrid_scorer = HybridScorer(config)
    
    def semantic_search(
        self, 
        query: str, 
        top_k: int = None,
        use_threshold: bool = True
    ) -> List[Dict[str, Any]]:
        """Perform semantic search.
        
        Args:
            query: Search query text
            top_k: Maximum number of results to return
            use_threshold: Whether to filter by similarity threshold
            
        Returns:
            List of search results with similarity scores
        """
        if top_k is None:
            top_k = self.config.max_related_notes
        
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embedding(query)
        
        # Search vector store
        raw_results = self.vector_store.search_similar(query_embedding, top_k=top_k)
        
        # Convert distance to similarity (assuming distance is in [0, 2] for cosine)
        for result in raw_results:
            result["similarity"] = 1.0 - min(result["distance"], 1.0)
        
        # Filter by threshold
        if use_threshold:
            results = self._filter_by_threshold(raw_results, self.config.similarity_threshold)
        else:
            results = raw_results
        
        # Rank results
        ranked_results = self._rank_results(results)
        
        return ranked_results[:top_k]
    
    def _filter_by_threshold(
        self, 
        results: List[Dict[str, Any]], 
        threshold: float
    ) -> List[Dict[str, Any]]:
        """Filter results by similarity threshold.
        
        Args:
            results: Raw search results
            threshold: Minimum similarity threshold
            
        Returns:
            Filtered results
        """
        return [
            result for result in results 
            if result.get("similarity", 0.0) >= threshold
        ]
    
    def _rank_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank results by similarity score.
        
        Args:
            results: Search results to rank
            
        Returns:
            Results sorted by similarity (descending)
        """
        return sorted(
            results, 
            key=lambda x: x.get("similarity", 0.0), 
            reverse=True
        )