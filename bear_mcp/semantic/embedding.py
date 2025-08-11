"""ABOUTME: Embedding generation pipeline using sentence-transformers
ABOUTME: Handles text preprocessing, model loading, and batch embedding generation"""

import re
from typing import List, Optional, Union, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer

from bear_mcp.config.models import EmbeddingConfig


class TextPreprocessor:
    """Handles text preprocessing for embedding generation."""
    
    def __init__(self):
        """Initialize text preprocessor."""
        pass
    
    def clean_markdown(self, text: str) -> str:
        """Remove markdown formatting from text.
        
        Args:
            text: Raw markdown text
            
        Returns:
            Clean text without markdown formatting
        """
        # Remove headers
        text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
        
        # Remove bold and italic
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        
        # Remove links but keep text
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        
        # Remove code blocks and inline code
        text = re.sub(r'```[^`]*```', '', text, flags=re.DOTALL)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        
        # Remove list markers
        text = re.sub(r'^[\s]*[-*+]\s+', '', text, flags=re.MULTILINE)
        
        return text
    
    def normalize_text(self, text: str) -> str:
        """Normalize whitespace and clean text.
        
        Args:
            text: Raw text
            
        Returns:
            Normalized text
        """
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing whitespace
        return text.strip()
    
    def chunk_text(self, text: str, max_chunk_size: int = 500) -> List[str]:
        """Split long text into smaller chunks.
        
        Args:
            text: Text to chunk
            max_chunk_size: Maximum words per chunk
            
        Returns:
            List of text chunks
        """
        words = text.split()
        
        if len(words) <= max_chunk_size:
            return [text]
        
        chunks = []
        current_chunk = []
        
        for word in words:
            current_chunk.append(word)
            
            if len(current_chunk) >= max_chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
        
        # Add remaining words
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks


class EmbeddingGenerator:
    """Generates embeddings using sentence-transformers."""
    
    def __init__(self, config: EmbeddingConfig):
        """Initialize embedding generator.
        
        Args:
            config: Embedding configuration
        """
        self.config = config
        self.model: Optional[SentenceTransformer] = None
        self._cache: Dict[str, np.ndarray] = {}
    
    def load_model(self) -> None:
        """Load the sentence transformer model."""
        self.model = SentenceTransformer(self.config.model_name)
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector as numpy array
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        embeddings = self.model.encode([text])
        return embeddings[0]  # Return 1D array for single text
    
    def generate_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts in batch.
        
        Args:
            texts: List of input texts
            
        Returns:
            2D numpy array of embeddings
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        return self.model.encode(texts)
    
    def _compute_embedding(self, text: str) -> np.ndarray:
        """Compute embedding for text (internal method)."""
        return self.generate_embedding(text)
    
    def get_cached_embedding(self, text: str) -> np.ndarray:
        """Get embedding with caching support.
        
        Args:
            text: Input text
            
        Returns:
            Cached or newly computed embedding
        """
        if text in self._cache:
            return self._cache[text]
        
        embedding = self._compute_embedding(text)
        self._cache[text] = embedding
        return embedding