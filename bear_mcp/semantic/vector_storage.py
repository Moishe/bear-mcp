"""ABOUTME: Vector storage using ChromaDB for embedding persistence and retrieval
ABOUTME: Provides embedding storage, similarity search, and caching functionality"""

from typing import List, Dict, Any, Optional, Union
from collections import OrderedDict
import numpy as np
import chromadb
from chromadb.config import Settings

from bear_mcp.config.models import VectorStorageConfig


class EmbeddingCache:
    """LRU cache for embedding vectors."""
    
    def __init__(self, max_size: int = 1000):
        """Initialize embedding cache.
        
        Args:
            max_size: Maximum number of embeddings to cache
        """
        self.max_size = max_size
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
    
    def get(self, key: str) -> Optional[np.ndarray]:
        """Get embedding from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached embedding or None if not found
        """
        if key in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return self._cache[key]
        return None
    
    def put(self, key: str, embedding: np.ndarray) -> None:
        """Store embedding in cache.
        
        Args:
            key: Cache key
            embedding: Embedding vector to cache
        """
        if key in self._cache:
            # Update existing entry
            self._cache[key] = embedding
            self._cache.move_to_end(key)
        else:
            # Add new entry
            self._cache[key] = embedding
            
            # Remove oldest if over capacity
            if len(self._cache) > self.max_size:
                self._cache.popitem(last=False)  # Remove oldest (FIFO)
    
    def clear(self) -> None:
        """Clear all cached embeddings."""
        self._cache.clear()


class VectorStore:
    """ChromaDB-based vector storage for embeddings."""
    
    def __init__(self, config: VectorStorageConfig):
        """Initialize vector store.
        
        Args:
            config: Vector storage configuration
        """
        self.config = config
        self.client: Optional[chromadb.Client] = None
        self.collection: Optional[chromadb.Collection] = None
    
    def initialize(self) -> None:
        """Initialize ChromaDB client and collection."""
        # Create persistent client
        self.client = chromadb.PersistentClient(path=str(self.config.persist_directory))
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=self.config.collection_name,
            metadata={"hnsw:space": self.config.distance_function}
        )
    
    def add_embeddings(
        self, 
        ids: List[str], 
        embeddings: List[np.ndarray], 
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Add embeddings to the vector store.
        
        Args:
            ids: Unique identifiers for embeddings
            embeddings: List of embedding vectors
            metadatas: Optional metadata for each embedding
        """
        if self.collection is None:
            raise RuntimeError("Vector store not initialized. Call initialize() first.")
        
        # Convert numpy arrays to lists for ChromaDB
        embeddings_list = [emb.tolist() for emb in embeddings]
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings_list,
            metadatas=metadatas
        )
    
    def search_similar(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Search for similar embeddings.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of similar embeddings to return
            
        Returns:
            List of similar embeddings with metadata
        """
        if self.collection is None:
            raise RuntimeError("Vector store not initialized. Call initialize() first.")
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        
        # Format results
        formatted_results = []
        for i, id_ in enumerate(results["ids"][0]):
            result = {
                "id": id_,
                "distance": results["distances"][0][i],
                "metadata": results["metadatas"][0][i] if results["metadatas"][0] else {}
            }
            formatted_results.append(result)
        
        return formatted_results
    
    def update_embedding(
        self, 
        id_: str, 
        embedding: np.ndarray, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update an existing embedding.
        
        Args:
            id_: Embedding identifier
            embedding: New embedding vector
            metadata: Optional new metadata
        """
        if self.collection is None:
            raise RuntimeError("Vector store not initialized. Call initialize() first.")
        
        update_kwargs = {
            "ids": [id_],
            "embeddings": [embedding.tolist()]
        }
        
        if metadata is not None:
            update_kwargs["metadatas"] = [metadata]
        
        self.collection.update(**update_kwargs)
    
    def delete_embedding(self, id_: str) -> None:
        """Delete an embedding.
        
        Args:
            id_: Embedding identifier to delete
        """
        if self.collection is None:
            raise RuntimeError("Vector store not initialized. Call initialize() first.")
        
        self.collection.delete(ids=[id_])
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics.
        
        Returns:
            Dictionary with collection stats
        """
        if self.collection is None:
            raise RuntimeError("Vector store not initialized. Call initialize() first.")
        
        return {
            "total_embeddings": self.collection.count(),
            "collection_name": self.config.collection_name,
            "distance_function": self.config.distance_function
        }