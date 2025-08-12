"""ABOUTME: AI-powered summarization service for Bear notes
ABOUTME: Integrates Ollama client with prompt templates for note summarization"""

import time
from enum import Enum
from typing import List, Dict, Any, Optional, AsyncGenerator
from dataclasses import dataclass
import structlog

from bear_mcp.config.models import BearMCPConfig
from bear_mcp.bear_db.models import BearNote
from bear_mcp.ai.ollama_client import OllamaClient, OllamaError
from bear_mcp.ai.prompt_templates import PromptTemplateManager

logger = structlog.get_logger(__name__)


class SummarizationError(Exception):
    """Exception raised for summarization-related errors."""
    pass


class SummaryStyle(Enum):
    """Enumeration of available summary styles."""
    BRIEF = "brief"
    DETAILED = "detailed"
    STRUCTURED = "structured"
    KEYWORDS = "keywords"


@dataclass
class SummaryRequest:
    """Request for note summarization."""
    style: SummaryStyle
    note: Optional[BearNote] = None
    notes: Optional[List[BearNote]] = None
    max_length: Optional[int] = None
    theme: Optional[str] = None
    sections: Optional[List[str]] = None
    custom_prompt: Optional[str] = None
    
    def __post_init__(self):
        """Validate request after initialization."""
        if not self.note and not self.notes:
            raise ValueError("Either note or notes must be provided")
        
        if self.note and self.notes:
            raise ValueError("Cannot provide both note and notes")


@dataclass
class SummaryResponse:
    """Response from note summarization."""
    summary: str
    style: SummaryStyle
    note_ids: List[str]
    model_used: str = ""
    tokens_processed: int = 0
    generation_time: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}


class SummarizationService:
    """AI-powered summarization service for Bear notes."""
    
    def __init__(self, config: BearMCPConfig):
        """Initialize summarization service.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.ollama_client = OllamaClient(config.ollama)
        self.template_manager = PromptTemplateManager()
        self._initialized = False
        
        # Service statistics
        self._summary_count = 0
        self._total_tokens_processed = 0
        self._total_generation_time = 0.0
    
    async def initialize(self) -> None:
        """Initialize the summarization service."""
        if self._initialized:
            logger.debug("Summarization service already initialized")
            return
        
        logger.info("Initializing summarization service")
        
        try:
            await self.ollama_client.initialize()
            self._initialized = True
            logger.info("Summarization service initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize summarization service", error=str(e))
            raise SummarizationError(f"Service initialization failed: {e}")
    
    async def cleanup(self) -> None:
        """Cleanup service resources."""
        if self.ollama_client:
            await self.ollama_client.cleanup()
        self._initialized = False
        logger.info("Summarization service cleaned up")
    
    def _ensure_initialized(self) -> None:
        """Ensure service is initialized."""
        if not self._initialized:
            raise SummarizationError("Service not initialized. Call initialize() first.")
    
    async def summarize(self, request: SummaryRequest) -> SummaryResponse:
        """Summarize note(s) according to request.
        
        Args:
            request: Summarization request
            
        Returns:
            Summary response
            
        Raises:
            SummarizationError: If summarization fails
        """
        self._ensure_initialized()
        
        start_time = time.time()
        
        try:
            # Generate appropriate prompt
            prompt = self._generate_prompt(request)
            
            # Get model name
            model_name = self.config.ollama.model
            
            # Generate summary
            logger.info("Generating summary", style=request.style.value, model=model_name)
            summary_text = await self.ollama_client.generate_response(prompt, model=model_name)
            
            # Calculate metrics
            generation_time = time.time() - start_time
            tokens_processed = len(prompt.split()) + len(summary_text.split())
            
            # Update service statistics
            self._summary_count += 1
            self._total_tokens_processed += tokens_processed
            self._total_generation_time += generation_time
            
            # Get note IDs
            if request.note:
                note_ids = [request.note.zuniqueidentifier]
            else:
                note_ids = [note.zuniqueidentifier for note in request.notes]
            
            # Create response
            response = SummaryResponse(
                summary=summary_text,
                style=request.style,
                note_ids=note_ids,
                model_used=model_name,
                tokens_processed=tokens_processed,
                generation_time=generation_time,
                metadata={
                    "prompt_template": self._get_template_name(request.style),
                    "request_params": self._extract_request_params(request)
                }
            )
            
            logger.info("Summary generated successfully", 
                       style=request.style.value,
                       tokens=tokens_processed,
                       time=f"{generation_time:.2f}s")
            
            return response
            
        except OllamaError as e:
            logger.error("Ollama error during summarization", error=str(e))
            raise SummarizationError(f"Failed to generate summary: {e}")
        except Exception as e:
            logger.error("Unexpected error during summarization", error=str(e))
            raise SummarizationError(f"Summarization failed: {e}")
    
    async def summarize_streaming(self, request: SummaryRequest) -> AsyncGenerator[str, None]:
        """Generate streaming summary response.
        
        Args:
            request: Summarization request
            
        Yields:
            Summary text chunks
            
        Raises:
            SummarizationError: If summarization fails
        """
        self._ensure_initialized()
        
        try:
            # Generate appropriate prompt
            prompt = self._generate_prompt(request)
            model_name = self.config.ollama.model
            
            logger.info("Starting streaming summary", style=request.style.value, model=model_name)
            
            # Stream response
            async for chunk in self.ollama_client.generate_streaming_response(prompt, model=model_name):
                yield chunk
            
            logger.info("Streaming summary completed", style=request.style.value)
            
        except OllamaError as e:
            logger.error("Ollama error during streaming", error=str(e))
            raise SummarizationError(f"Failed to stream summary: {e}")
        except Exception as e:
            logger.error("Unexpected error during streaming", error=str(e))
            raise SummarizationError(f"Streaming failed: {e}")
    
    async def extract_keywords(self, note: BearNote, max_keywords: int = 10) -> List[str]:
        """Extract keywords from a note.
        
        Args:
            note: Note to extract keywords from
            max_keywords: Maximum number of keywords to extract
            
        Returns:
            List of extracted keywords
        """
        self._ensure_initialized()
        
        try:
            prompt = self.template_manager.render_template(
                "keyword_extraction",
                content=note.ztext or "",
                max_keywords=str(max_keywords)
            )
            
            response = await self.ollama_client.generate_response(prompt)
            
            # Parse comma-separated keywords
            keywords = [kw.strip() for kw in response.split(",") if kw.strip()]
            
            logger.info("Keywords extracted", note_id=note.zuniqueidentifier, count=len(keywords))
            return keywords
            
        except Exception as e:
            logger.error("Failed to extract keywords", note_id=note.zuniqueidentifier, error=str(e))
            raise SummarizationError(f"Keyword extraction failed: {e}")
    
    async def health_check(self) -> bool:
        """Check if service is healthy.
        
        Returns:
            True if service is healthy
        """
        try:
            if not self._initialized:
                return False
            
            return await self.ollama_client.health_check()
        except Exception as e:
            logger.warning("Health check failed", error=str(e))
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get service statistics.
        
        Returns:
            Dictionary with service statistics
        """
        is_healthy = await self.health_check() if self._initialized else False
        
        stats = {
            "is_initialized": self._initialized,
            "is_healthy": is_healthy,
            "summaries_generated": self._summary_count,
            "total_tokens_processed": self._total_tokens_processed,
            "total_generation_time": round(self._total_generation_time, 2),
            "average_generation_time": round(
                self._total_generation_time / max(self._summary_count, 1), 2
            ),
            "model_name": self.config.ollama.model,
            "ollama_base_url": self.config.ollama.base_url,
            "available_templates": len(self.template_manager.list_templates())
        }
        
        logger.info("Retrieved service stats", **stats)
        return stats
    
    def _generate_prompt(self, request: SummaryRequest) -> str:
        """Generate prompt for summarization request.
        
        Args:
            request: Summarization request
            
        Returns:
            Generated prompt string
        """
        if request.custom_prompt:
            return request.custom_prompt
        
        # Single note summarization
        if request.note:
            note = request.note
            title = note.ztitle or "Untitled"
            content = note.ztext or "No content"
            
            if request.style == SummaryStyle.BRIEF:
                max_words = request.max_length or 50
                return self.template_manager.render_template(
                    "brief_summary",
                    title=title,
                    content=content,
                    max_words=str(max_words)
                )
            
            elif request.style == SummaryStyle.DETAILED:
                max_sentences = request.max_length or 5
                return self.template_manager.render_template(
                    "detailed_summary",
                    title=title,
                    content=content,
                    max_sentences=str(max_sentences)
                )
            
            elif request.style == SummaryStyle.STRUCTURED:
                sections = request.sections or ["Main Topic", "Key Points", "Summary"]
                return self.template_manager.render_template(
                    "structured_summary",
                    title=title,
                    content=content,
                    sections=", ".join(sections)
                )
            
            elif request.style == SummaryStyle.KEYWORDS:
                max_keywords = request.max_length or 10
                return self.template_manager.render_template(
                    "keyword_extraction",
                    content=content,
                    max_keywords=str(max_keywords)
                )
        
        # Multiple notes summarization
        elif request.notes:
            theme = request.theme or "Related Notes"
            max_paragraphs = request.max_length or 3
            
            return self.template_manager.render_template(
                "multi_note_summary",
                notes=request.notes,
                theme=theme,
                max_paragraphs=str(max_paragraphs)
            )
        
        raise SummarizationError("Invalid request: no notes provided")
    
    def _get_template_name(self, style: SummaryStyle) -> str:
        """Get template name for summary style."""
        template_map = {
            SummaryStyle.BRIEF: "brief_summary",
            SummaryStyle.DETAILED: "detailed_summary", 
            SummaryStyle.STRUCTURED: "structured_summary",
            SummaryStyle.KEYWORDS: "keyword_extraction"
        }
        return template_map.get(style, "unknown")
    
    def _extract_request_params(self, request: SummaryRequest) -> Dict[str, Any]:
        """Extract parameters from request for metadata."""
        params = {
            "style": request.style.value,
            "max_length": request.max_length,
            "theme": request.theme,
            "sections": request.sections,
            "has_custom_prompt": bool(request.custom_prompt)
        }
        
        if request.note:
            params["note_count"] = 1
            params["note_id"] = request.note.zuniqueidentifier
        elif request.notes:
            params["note_count"] = len(request.notes)
            params["note_ids"] = [n.zuniqueidentifier for n in request.notes]
        
        return params