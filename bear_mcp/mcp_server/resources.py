"""Resource management for Bear Notes MCP Server."""

import json
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse

import structlog
from mcp.types import Resource, TextContent

from bear_mcp.config.models import BearMCPConfig
from bear_mcp.bear_db.connection import BearDatabaseRepository
from bear_mcp.bear_db.models import BearNote, NoteMetadata

logger = structlog.get_logger()


class BearNotesResourceManager:
    """Manages MCP resources for Bear Notes."""
    
    def __init__(self, repository: BearDatabaseRepository, config: BearMCPConfig):
        """Initialize the resource manager.
        
        Args:
            repository: Database repository
            config: Server configuration
        """
        self.repository = repository
        self.config = config
        
        # Resource cache
        self._notes_cache: Optional[List[BearNote]] = None
        self._metadata_cache: Optional[List[NoteMetadata]] = None
        self._cache_valid = False
    
    async def list_resources(self) -> List[Resource]:
        """List all available resources.
        
        Returns:
            List of available resources
        """
        resources = []
        
        # Add the "all notes" resource
        resources.append(
            Resource(
                uri="notes://all",
                name="All Notes",
                description="List of all active notes in Bear",
                mimeType="application/json"
            )
        )
        
        # Add individual note resources
        try:
            notes_metadata = await self._get_notes_metadata()
            
            # Limit the number of resources returned
            max_resources = self.config.mcp_server.max_resources
            limited_notes = notes_metadata[:max_resources]
            
            for note in limited_notes:
                resources.append(
                    Resource(
                        uri=f"note:///{note.id}",
                        name=note.title,
                        description=f"Note: {note.title} (modified: {note.modification_date})",
                        mimeType="text/markdown"
                    )
                )
            
            if len(notes_metadata) > max_resources:
                logger.info(
                    "Limited resources returned", 
                    total=len(notes_metadata), 
                    returned=max_resources
                )
        
        except Exception as e:
            logger.error("Error getting note resources", error=str(e))
            # Return at least the "all notes" resource
        
        logger.debug("Listed resources", count=len(resources))
        return resources
    
    async def read_resource(self, uri: str) -> List[TextContent]:
        """Read a specific resource by URI.
        
        Args:
            uri: Resource URI
            
        Returns:
            List of text content
            
        Raises:
            ValueError: If URI is invalid or resource not found
        """
        parsed = urlparse(uri)
        
        if parsed.scheme == "notes" and parsed.path == "//all":
            return await self._read_all_notes_resource()
        elif parsed.scheme == "note":
            note_id = parsed.path.lstrip("/")
            return await self._read_note_resource(note_id)
        else:
            raise ValueError(f"Unknown resource URI: {uri}")
    
    async def _read_all_notes_resource(self) -> List[TextContent]:
        """Read the 'all notes' resource.
        
        Returns:
            List containing JSON content of all notes metadata
        """
        try:
            notes_metadata = await self._get_notes_metadata()
            
            # Convert to serializable format
            notes_data = []
            for note in notes_metadata:
                note_data = note.model_dump()
                # Convert datetime objects to ISO strings
                if note_data["creation_date"]:
                    note_data["creation_date"] = note_data["creation_date"].isoformat()
                if note_data["modification_date"]:
                    note_data["modification_date"] = note_data["modification_date"].isoformat()
                notes_data.append(note_data)
            
            content = {
                "notes": notes_data,
                "total_count": len(notes_data),
                "resource_uri": "notes://all"
            }
            
            return [
                TextContent(
                    type="text",
                    text=json.dumps(content, indent=2)
                )
            ]
            
        except Exception as e:
            logger.error("Error reading all notes resource", error=str(e))
            raise ValueError(f"Failed to read all notes resource: {e}")
    
    async def _read_note_resource(self, note_id: str) -> List[TextContent]:
        """Read a specific note resource.
        
        Args:
            note_id: Note unique identifier
            
        Returns:
            List containing the note content
        """
        try:
            note = self.repository.get_note_by_id(note_id)
            
            if not note:
                raise ValueError(f"Note not found: {note_id}")
            
            # Build markdown content
            content_parts = []
            
            # Add title
            if note.ztitle:
                content_parts.append(f"# {note.ztitle}")
                content_parts.append("")
            
            # Add metadata
            metadata_lines = [
                "<!-- Note Metadata -->",
                f"- **ID:** {note.zuniqueidentifier}",
            ]
            
            if note.creation_date:
                metadata_lines.append(f"- **Created:** {note.creation_date.isoformat()}")
            
            if note.modification_date:
                metadata_lines.append(f"- **Modified:** {note.modification_date.isoformat()}")
            
            if note.zpinned:
                metadata_lines.append("- **Pinned:** Yes")
            
            if note.ztrashed:
                metadata_lines.append("- **Status:** Trashed")
            elif note.zarchived:
                metadata_lines.append("- **Status:** Archived")
            else:
                metadata_lines.append("- **Status:** Active")
            
            content_parts.extend(metadata_lines)
            content_parts.append("")
            content_parts.append("---")
            content_parts.append("")
            
            # Add note content
            if note.ztext:
                content_parts.append(note.ztext)
            else:
                content_parts.append("*[No content]*")
            
            content = "\n".join(content_parts)
            
            return [
                TextContent(
                    type="text",
                    text=content
                )
            ]
            
        except Exception as e:
            logger.error("Error reading note resource", note_id=note_id, error=str(e))
            raise ValueError(f"Failed to read note {note_id}: {e}")
    
    async def _get_notes_metadata(self) -> List[NoteMetadata]:
        """Get notes metadata with caching.
        
        Returns:
            List of note metadata
        """
        if not self._cache_valid or self._metadata_cache is None:
            logger.debug("Refreshing notes metadata cache")
            self._metadata_cache = self.repository.get_note_metadata(
                include_trashed=False, 
                include_archived=False
            )
            self._cache_valid = True
        
        return self._metadata_cache
    
    async def refresh_cache(self) -> None:
        """Refresh the internal cache."""
        logger.info("Refreshing resource cache")
        self._cache_valid = False
        self._notes_cache = None
        self._metadata_cache = None
        
        # Pre-load cache
        await self._get_notes_metadata()
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        logger.debug("Cleaning up resource manager")
        self._notes_cache = None
        self._metadata_cache = None
        self._cache_valid = False