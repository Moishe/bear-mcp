"""Tool management for Bear Notes MCP Server."""

import json
from typing import List, Dict, Any, Optional

import structlog
from mcp.types import Tool, TextContent, CallToolResult

from bear_mcp.config.models import BearMCPConfig
from bear_mcp.bear_db.connection import BearDatabaseRepository
from bear_mcp.bear_db.models import BearNote, DatabaseStats

logger = structlog.get_logger()


class BearNotesToolManager:
    """Manages MCP tools for Bear Notes."""
    
    def __init__(self, repository: BearDatabaseRepository, config: BearMCPConfig):
        """Initialize the tool manager.
        
        Args:
            repository: Database repository
            config: Server configuration
        """
        self.repository = repository
        self.config = config
        
        # Tool cache
        self._cache_valid = False
    
    async def list_tools(self) -> List[Tool]:
        """List all available tools.
        
        Returns:
            List of available tools
        """
        tools = []
        
        # Get note content tool
        tools.append(
            Tool(
                name="get_note",
                description="Get the full content of a specific note by its ID",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "note_id": {
                            "type": "string",
                            "description": "The unique identifier of the note"
                        }
                    },
                    "required": ["note_id"]
                }
            )
        )
        
        # Search notes tool
        tools.append(
            Tool(
                name="search_notes",
                description="Search for notes by title or content",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query to match against note titles and content"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results to return",
                            "default": 10
                        },
                        "include_trashed": {
                            "type": "boolean",
                            "description": "Whether to include trashed notes in results",
                            "default": False
                        }
                    },
                    "required": ["query"]
                }
            )
        )
        
        # Get database stats tool
        tools.append(
            Tool(
                name="get_database_stats",
                description="Get statistics about the Bear database",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False
                }
            )
        )
        
        # List notes tool
        tools.append(
            Tool(
                name="list_notes",
                description="List notes with optional filtering",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer", 
                            "description": "Maximum number of notes to return",
                            "default": 50
                        },
                        "include_trashed": {
                            "type": "boolean",
                            "description": "Whether to include trashed notes",
                            "default": False
                        },
                        "include_archived": {
                            "type": "boolean", 
                            "description": "Whether to include archived notes",
                            "default": False
                        }
                    },
                    "additionalProperties": False
                }
            )
        )
        
        logger.debug("Listed tools", count=len(tools))
        return tools
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> CallToolResult:
        """Call a specific tool with given arguments.
        
        Args:
            name: Tool name
            arguments: Tool arguments
            
        Returns:
            Tool call result
            
        Raises:
            ValueError: If tool name is unknown or arguments are invalid
        """
        logger.debug("Calling tool", name=name, arguments=arguments)
        
        try:
            if name == "get_note":
                return await self._get_note_tool(arguments)
            elif name == "search_notes":
                return await self._search_notes_tool(arguments)
            elif name == "get_database_stats":
                return await self._get_database_stats_tool(arguments)
            elif name == "list_notes":
                return await self._list_notes_tool(arguments)
            else:
                raise ValueError(f"Unknown tool: {name}")
                
        except Exception as e:
            logger.error("Tool call failed", name=name, error=str(e))
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"Error calling tool '{name}': {str(e)}"
                    )
                ]
            )
    
    async def _get_note_tool(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Handle get_note tool calls."""
        note_id = arguments.get("note_id")
        if not note_id:
            raise ValueError("note_id is required")
        
        note = self.repository.get_note_by_id(note_id)
        if not note:
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"Note not found: {note_id}"
                    )
                ]
            )
        
        # Format note data
        note_data = {
            "id": note.zuniqueidentifier,
            "title": note.ztitle or "Untitled",
            "content": note.ztext or "",
            "creation_date": note.creation_date.isoformat() if note.creation_date else None,
            "modification_date": note.modification_date.isoformat() if note.modification_date else None,
            "is_pinned": note.zpinned,
            "is_trashed": note.ztrashed,
            "is_archived": note.zarchived,
        }
        
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=json.dumps(note_data, indent=2)
                )
            ]
        )
    
    async def _search_notes_tool(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Handle search_notes tool calls."""
        query = arguments.get("query")
        if not query:
            raise ValueError("query is required")
        
        limit = arguments.get("limit", 10)
        include_trashed = arguments.get("include_trashed", False)
        
        # Get all notes
        notes = self.repository.get_notes(
            include_trashed=include_trashed,
            include_archived=False,  # Don't include archived by default
            limit=None  # We'll filter and limit after search
        )
        
        # Simple text search (case-insensitive)
        query_lower = query.lower()
        matching_notes = []
        
        for note in notes:
            # Search in title
            title_match = note.ztitle and query_lower in note.ztitle.lower()
            # Search in content
            content_match = note.ztext and query_lower in note.ztext.lower()
            
            if title_match or content_match:
                matching_notes.append({
                    "id": note.zuniqueidentifier,
                    "title": note.ztitle or "Untitled",
                    "creation_date": note.creation_date.isoformat() if note.creation_date else None,
                    "modification_date": note.modification_date.isoformat() if note.modification_date else None,
                    "is_pinned": note.zpinned,
                    "match_type": "title" if title_match else "content"
                })
        
        # Limit results
        matching_notes = matching_notes[:limit]
        
        result_data = {
            "query": query,
            "results": matching_notes,
            "total_matches": len(matching_notes)
        }
        
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=json.dumps(result_data, indent=2)
                )
            ]
        )
    
    async def _get_database_stats_tool(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Handle get_database_stats tool calls."""
        stats = self.repository.get_database_stats()
        
        stats_data = {
            "total_notes": stats.total_notes,
            "active_notes": stats.active_notes,
            "trashed_notes": stats.trashed_notes,
            "archived_notes": stats.archived_notes,
            "total_tags": stats.total_tags,
            "last_updated": stats.last_updated.isoformat()
        }
        
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=json.dumps(stats_data, indent=2)
                )
            ]
        )
    
    async def _list_notes_tool(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Handle list_notes tool calls."""
        limit = arguments.get("limit", 50)
        include_trashed = arguments.get("include_trashed", False)
        include_archived = arguments.get("include_archived", False)
        
        notes = self.repository.get_notes(
            include_trashed=include_trashed,
            include_archived=include_archived,
            limit=limit
        )
        
        notes_data = []
        for note in notes:
            notes_data.append({
                "id": note.zuniqueidentifier,
                "title": note.ztitle or "Untitled",
                "creation_date": note.creation_date.isoformat() if note.creation_date else None,
                "modification_date": note.modification_date.isoformat() if note.modification_date else None,
                "is_pinned": note.zpinned,
                "is_trashed": note.ztrashed,
                "is_archived": note.zarchived,
                "word_count": len(note.ztext.split()) if note.ztext else 0
            })
        
        result_data = {
            "notes": notes_data,
            "total_count": len(notes_data),
            "filters": {
                "include_trashed": include_trashed,
                "include_archived": include_archived,
                "limit": limit
            }
        }
        
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=json.dumps(result_data, indent=2)
                )
            ]
        )
    
    async def refresh_cache(self) -> None:
        """Refresh the internal cache."""
        logger.info("Refreshing tool cache")
        self._cache_valid = False
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        logger.debug("Cleaning up tool manager")
        self._cache_valid = False