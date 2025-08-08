"""Configuration models for Bear MCP Server."""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, validator


class BearDatabaseConfig(BaseModel):
    """Configuration for Bear database connection."""
    
    path: Optional[Path] = Field(
        default=None,
        description="Path to Bear database file. If None, will auto-detect."
    )
    read_only: bool = Field(
        default=True,
        description="Whether to open database in read-only mode"
    )
    timeout: float = Field(
        default=30.0,
        description="Database connection timeout in seconds"
    )
    check_same_thread: bool = Field(
        default=False,
        description="SQLite check_same_thread parameter"
    )


class EmbeddingConfig(BaseModel):
    """Configuration for embedding generation."""
    
    model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Name of the sentence transformer model"
    )
    batch_size: int = Field(
        default=32,
        description="Batch size for embedding generation"
    )
    max_length: int = Field(
        default=512,
        description="Maximum token length for embeddings"
    )
    cache_dir: Optional[Path] = Field(
        default=None,
        description="Directory for caching embeddings"
    )


class VectorStorageConfig(BaseModel):
    """Configuration for vector storage (ChromaDB)."""
    
    persist_directory: Path = Field(
        default=Path("./chroma_db"),
        description="Directory for ChromaDB persistence"
    )
    collection_name: str = Field(
        default="bear_notes",
        description="Name of the ChromaDB collection"
    )
    distance_function: str = Field(
        default="cosine",
        description="Distance function for similarity search"
    )


class OllamaConfig(BaseModel):
    """Configuration for Ollama AI integration."""
    
    base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for Ollama API"
    )
    model: str = Field(
        default="llama2",
        description="Default model for summarization"
    )
    timeout: float = Field(
        default=60.0,
        description="Request timeout in seconds"
    )
    max_retries: int = Field(
        default=3,
        description="Maximum number of retries for failed requests"
    )


class MCPServerConfig(BaseModel):
    """Configuration for MCP server."""
    
    name: str = Field(
        default="bear-notes-mcp",
        description="Name of the MCP server"
    )
    version: str = Field(
        default="0.1.0",
        description="Version of the MCP server"
    )
    max_resources: int = Field(
        default=1000,
        description="Maximum number of resources to return"
    )


class LoggingConfig(BaseModel):
    """Configuration for logging."""
    
    level: str = Field(
        default="INFO",
        description="Logging level"
    )
    format: str = Field(
        default="json",
        description="Logging format (json or text)"
    )
    file_path: Optional[Path] = Field(
        default=None,
        description="Optional log file path"
    )

    @validator("level")
    def validate_level(cls, v: str) -> str:
        """Validate logging level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid logging level: {v}. Must be one of {valid_levels}")
        return v.upper()


class PerformanceConfig(BaseModel):
    """Configuration for performance optimization."""
    
    cache_size: int = Field(
        default=1000,
        description="LRU cache size for embeddings"
    )
    similarity_threshold: float = Field(
        default=0.7,
        description="Minimum similarity threshold for related notes"
    )
    max_related_notes: int = Field(
        default=10,
        description="Maximum number of related notes to return"
    )
    refresh_debounce_seconds: float = Field(
        default=2.0,
        description="Debounce time for database refresh triggers"
    )


class BearMCPConfig(BaseModel):
    """Main configuration for Bear MCP Server."""
    
    bear_db: BearDatabaseConfig = Field(default_factory=BearDatabaseConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_storage: VectorStorageConfig = Field(default_factory=VectorStorageConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    mcp_server: MCPServerConfig = Field(default_factory=MCPServerConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)

    class Config:
        """Pydantic configuration."""
        extra = "forbid"  # Prevent extra fields
        validate_assignment = True  # Validate on assignment