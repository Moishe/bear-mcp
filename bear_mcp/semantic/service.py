"""ABOUTME: Complete semantic search service integrating all Phase 2 components
ABOUTME: Provides high-level API for indexing, searching, and managing note embeddings"""

from typing import List, Dict, Any, Optional
import asyncio
import structlog
# Import for mocking in tests
import chromadb
from sentence_transformers import SentenceTransformer

from bear_mcp.config.models import BearMCPConfig
from bear_mcp.bear_db.models import BearNote
from bear_mcp.semantic.embedding import EmbeddingGenerator
from bear_mcp.semantic.vector_storage import VectorStore
from bear_mcp.semantic.content_processing import ContentProcessor
from bear_mcp.semantic.similarity import SearchEngine

logger = structlog.get_logger(__name__)


class SemanticSearchService:
    """High-level semantic search service."""
    
    def __init__(self, config: BearMCPConfig):
        """Initialize semantic search service.
        
        Args:
            config: Complete application configuration
        """
        self.config = config
        self._initialized = False
        
        # Initialize components
        self.embedding_generator = EmbeddingGenerator(config.embedding)
        self.vector_store = VectorStore(config.vector_storage)
        self.content_processor = ContentProcessor()
        
        # Will be initialized in initialize() method
        self.search_engine: Optional[SearchEngine] = None
    
    async def initialize(self) -> None:
        """Initialize all service components."""
        if self._initialized:
            logger.debug("Service already initialized")
            return
        
        logger.info("Initializing semantic search service")
        
        try:
            # Load embedding model
            logger.info("Loading embedding model", model=self.config.embedding.model_name)
            self.embedding_generator.load_model()
            
            # Initialize vector store
            logger.info("Initializing vector store", path=self.config.vector_storage.persist_directory)
            self.vector_store.initialize()
            
            # Initialize search engine with all components
            self.search_engine = SearchEngine(
                self.config.performance,
                self.embedding_generator,
                self.vector_store,
                self.content_processor
            )
            
            self._initialized = True
            logger.info("Semantic search service initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize semantic search service", error=str(e))
            raise
    
    def _ensure_initialized(self) -> None:
        """Ensure service is initialized."""
        if not self._initialized:
            raise RuntimeError("Service not initialized. Call initialize() first.")
    
    async def index_notes(self, notes: List[BearNote]) -> None:
        """Index notes into the vector store.
        
        Args:
            notes: List of Bear notes to index
        """
        self._ensure_initialized()
        
        if not notes:
            logger.info("No notes to index")
            return
        
        logger.info("Indexing notes", count=len(notes))
        
        try:
            # Process notes and generate embeddings
            processed_texts = []
            note_ids = []
            metadatas = []
            
            for note in notes:
                # Process note content
                processed_text = self.content_processor.process_note_for_embedding(note)
                processed_texts.append(processed_text)
                note_ids.append(note.zuniqueidentifier)
                
                # Create metadata
                metadata = {
                    "title": note.ztitle or "Untitled",
                    "created": str(note.creation_date) if note.creation_date else None,
                    "modified": str(note.modification_date) if note.modification_date else None,
                    "has_title": bool(note.ztitle),
                    "text_length": len(note.ztext or "")
                }
                metadatas.append(metadata)
            
            # Generate embeddings in batch
            embeddings = self.embedding_generator.generate_embeddings_batch(processed_texts)
            
            # Add to vector store
            self.vector_store.add_embeddings(note_ids, embeddings, metadatas)
            
            logger.info("Successfully indexed notes", count=len(notes))
            
        except Exception as e:
            logger.error("Failed to index notes", error=str(e), count=len(notes))
            raise
    
    async def search(
        self, 
        query: str, 
        top_k: int = None,
        use_threshold: bool = True
    ) -> List[Dict[str, Any]]:
        """Search for notes similar to query.
        
        Args:
            query: Search query text
            top_k: Maximum number of results
            use_threshold: Whether to apply similarity threshold
            
        Returns:
            List of search results with similarity scores
        """
        self._ensure_initialized()
        
        if not self.search_engine:
            raise RuntimeError("Search engine not initialized")
        
        if not query.strip():
            logger.warning("Empty search query provided")
            return []
        
        if top_k is None:
            top_k = self.config.performance.max_related_notes
        
        logger.info("Performing semantic search", query=query[:50], top_k=top_k)
        
        try:
            results = self.search_engine.semantic_search(
                query, 
                top_k=top_k, 
                use_threshold=use_threshold
            )
            
            logger.info("Search completed", query=query[:50], results_count=len(results))
            return results
            
        except Exception as e:
            logger.error("Search failed", error=str(e), query=query[:50])
            raise
    
    async def find_related_notes(
        self, 
        reference_note: BearNote, 
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """Find notes related to a given reference note.
        
        Args:
            reference_note: Note to find related notes for
            top_k: Maximum number of related notes
            
        Returns:
            List of related notes with similarity scores
        """
        self._ensure_initialized()
        
        # Process reference note to get search text
        processed_text = self.content_processor.process_note_for_embedding(reference_note)
        
        logger.info(
            "Finding related notes", 
            reference_id=reference_note.zuniqueidentifier,
            title=reference_note.ztitle or "Untitled"
        )
        
        # Use the processed text as search query
        results = await self.search(processed_text, top_k=top_k)
        
        # Filter out the reference note itself if present in results
        filtered_results = [
            result for result in results 
            if result.get("id") != reference_note.zuniqueidentifier
        ]
        
        logger.info(
            "Found related notes",
            reference_id=reference_note.zuniqueidentifier,
            related_count=len(filtered_results)
        )
        
        return filtered_results
    
    async def update_note_embedding(self, note: BearNote) -> None:
        """Update embedding for a modified note.
        
        Args:
            note: Updated note
        """
        self._ensure_initialized()
        
        logger.info("Updating note embedding", note_id=note.zuniqueidentifier)
        
        try:
            # Process note content
            processed_text = self.content_processor.process_note_for_embedding(note)
            
            # Generate new embedding
            embedding = self.embedding_generator.generate_embedding(processed_text)
            
            # Create updated metadata
            metadata = {
                "title": note.ztitle or "Untitled",
                "created": str(note.creation_date) if note.creation_date else None,
                "modified": str(note.modification_date) if note.modification_date else None,
                "has_title": bool(note.ztitle),
                "text_length": len(note.ztext or "")
            }
            
            # Update in vector store
            self.vector_store.update_embedding(note.zuniqueidentifier, embedding, metadata)
            
            logger.info("Successfully updated note embedding", note_id=note.zuniqueidentifier)
            
        except Exception as e:
            logger.error("Failed to update note embedding", error=str(e), note_id=note.zuniqueidentifier)
            raise
    
    async def delete_note_embedding(self, note_id: str) -> None:
        """Delete embedding for a note.
        
        Args:
            note_id: Unique identifier of note to delete
        """
        self._ensure_initialized()
        
        logger.info("Deleting note embedding", note_id=note_id)
        
        try:
            self.vector_store.delete_embedding(note_id)
            logger.info("Successfully deleted note embedding", note_id=note_id)
            
        except Exception as e:
            logger.error("Failed to delete note embedding", error=str(e), note_id=note_id)
            raise
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get service statistics.
        
        Returns:
            Dictionary with service statistics
        """
        self._ensure_initialized()
        
        try:
            vector_stats = self.vector_store.get_stats()
            
            stats = {
                "indexed_notes": vector_stats["total_embeddings"],
                "collection_name": vector_stats["collection_name"],
                "model_name": self.config.embedding.model_name,
                "similarity_threshold": self.config.performance.similarity_threshold,
                "max_related_notes": self.config.performance.max_related_notes,
                "cache_size": self.config.performance.cache_size,
                "distance_function": vector_stats["distance_function"]
            }
            
            logger.info("Retrieved service stats", **stats)
            return stats
            
        except Exception as e:
            logger.error("Failed to get service stats", error=str(e))
            raise