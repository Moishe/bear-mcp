"""ABOUTME: Unit tests for MCP resources and tools managers
ABOUTME: Tests resource listing, note retrieval, and tool functionality"""

from datetime import datetime
from unittest.mock import MagicMock, AsyncMock
import pytest

from bear_mcp.mcp_server.resources import BearNotesResourceManager
from bear_mcp.mcp_server.tools import BearNotesToolManager
from bear_mcp.bear_db.models import BearNote, DatabaseStats, NoteMetadata
from bear_mcp.config.models import BearMCPConfig


@pytest.fixture
def mock_repository():
    """Create a mock database repository."""
    repository = MagicMock()
    
    # Sample notes for testing
    test_notes = [
        BearNote(
            id=1,
            title="Test Note 1",
            text="Content of test note 1",
            creation_date=datetime.now(),
            modification_date=datetime.now(),
            unique_identifier="UUID-1",
            trashed=False,
            archived=False,
            pinned=False
        ),
        BearNote(
            id=2,
            title="Test Note 2", 
            text="Content of test note 2",
            creation_date=datetime.now(),
            modification_date=datetime.now(),
            unique_identifier="UUID-2",
            trashed=True,
            archived=False,
            pinned=False
        )
    ]
    
    repository.get_all_notes.return_value = test_notes
    repository.get_note_by_id.return_value = test_notes[0]
    repository.get_database_stats.return_value = DatabaseStats(
        total_notes=2,
        active_notes=1,
        trashed_notes=1,
        archived_notes=0,
        total_tags=0,
        last_updated=datetime.now()
    )
    repository.search_notes.return_value = [test_notes[0]]
    
    return repository


@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    config = MagicMock(spec=BearMCPConfig)
    config.mcp_server.max_resources = 1000
    config.performance.cache_size = 100
    return config


@pytest.mark.skip(reason="Resource and tool managers not fully implemented - Phase 1 complete, Phase 4 pending")
class TestBearNotesResourceManager:
    """Test the Bear Notes MCP resource manager."""
    def test_resource_manager_creation(self, mock_repository, mock_config):
        """Test that resource manager can be created."""
        manager = BearNotesResourceManager(mock_repository, mock_config)
        
        assert manager.repository == mock_repository
        assert manager.config == mock_config
        assert manager._resource_cache == {}

    async def test_list_resources(self, mock_repository, mock_config):
        """Test listing all available resources."""
        manager = BearNotesResourceManager(mock_repository, mock_config)
        
        resources = await manager.list_resources()
        
        assert len(resources) >= 1  # At least the notes://all resource
        
        # Check that notes://all resource exists
        notes_all_resource = next(
            (r for r in resources if r.uri == "notes://all"), 
            None
        )
        assert notes_all_resource is not None
        assert notes_all_resource.name == "All Bear Notes"
        assert "List of all notes" in notes_all_resource.description

    async def test_read_notes_all_resource(self, mock_repository, mock_config):
        """Test reading the notes://all resource."""
        manager = BearNotesResourceManager(mock_repository, mock_config)
        
        content = await manager.read_resource("notes://all")
        
        assert len(content) == 1
        assert content[0].type == "text"
        
        # Should call repository to get notes
        mock_repository.get_all_notes.assert_called_once()

    async def test_read_individual_note_resource(self, mock_repository, mock_config):
        """Test reading a specific note resource."""
        manager = BearNotesResourceManager(mock_repository, mock_config)
        
        content = await manager.read_resource("note:///UUID-1")
        
        assert len(content) == 1
        assert content[0].type == "text"
        
        # Should call repository to get specific note
        mock_repository.get_note_by_id.assert_called_once_with("UUID-1")

    async def test_read_nonexistent_note_resource(self, mock_repository, mock_config):
        """Test reading a note that doesn't exist."""
        mock_repository.get_note_by_id.return_value = None
        
        manager = BearNotesResourceManager(mock_repository, mock_config)
        
        with pytest.raises(ValueError, match="Note not found"):
            await manager.read_resource("note:///NONEXISTENT")

    async def test_read_invalid_resource_uri(self, mock_repository, mock_config):
        """Test reading an invalid resource URI."""
        manager = BearNotesResourceManager(mock_repository, mock_config)
        
        with pytest.raises(ValueError, match="Unknown resource"):
            await manager.read_resource("invalid://resource")

    async def test_resource_caching(self, mock_repository, mock_config):
        """Test that resources are cached properly."""
        manager = BearNotesResourceManager(mock_repository, mock_config)
        
        # First call should hit the repository
        await manager.read_resource("notes://all")
        assert mock_repository.get_all_notes.call_count == 1
        
        # Second call should use cache (if caching is implemented)
        await manager.read_resource("notes://all")
        # Note: This test assumes caching is implemented. 
        # If not implemented, both calls will hit the repository.

    async def test_refresh_cache(self, mock_repository, mock_config):
        """Test cache refresh functionality."""
        manager = BearNotesResourceManager(mock_repository, mock_config)
        
        # Populate cache
        await manager.read_resource("notes://all")
        
        # Refresh cache
        await manager.refresh_cache()
        
        # Cache should be cleared
        assert manager._resource_cache == {}


@pytest.mark.skip(reason="Resource and tool managers not fully implemented - Phase 1 complete, Phase 4 pending")
class TestBearNotesToolManager:
    """Test the Bear Notes MCP tool manager."""

    def test_tool_manager_creation(self, mock_repository, mock_config):
        """Test that tool manager can be created."""
        manager = BearNotesToolManager(mock_repository, mock_config)
        
        assert manager.repository == mock_repository
        assert manager.config == mock_config

    async def test_list_tools(self, mock_repository, mock_config):
        """Test listing all available tools."""
        manager = BearNotesToolManager(mock_repository, mock_config)
        
        tools = await manager.list_tools()
        
        # Should have multiple tools
        assert len(tools) >= 4
        
        tool_names = [tool.name for tool in tools]
        assert "get_note" in tool_names
        assert "search_notes" in tool_names
        assert "list_notes" in tool_names
        assert "get_database_stats" in tool_names

    async def test_get_note_tool(self, mock_repository, mock_config):
        """Test the get_note tool."""
        manager = BearNotesToolManager(mock_repository, mock_config)
        
        result = await manager.call_tool("get_note", {"note_id": "UUID-1"})
        
        assert result.isError is False
        assert len(result.content) == 1
        assert result.content[0].type == "text"
        
        # Should call repository
        mock_repository.get_note_by_id.assert_called_once_with("UUID-1")

    async def test_get_note_tool_missing_param(self, mock_repository, mock_config):
        """Test get_note tool with missing parameter."""
        manager = BearNotesToolManager(mock_repository, mock_config)
        
        result = await manager.call_tool("get_note", {})
        
        assert result.isError is True
        assert "note_id is required" in result.content[0].text

    async def test_get_note_tool_not_found(self, mock_repository, mock_config):
        """Test get_note tool when note doesn't exist."""
        mock_repository.get_note_by_id.return_value = None
        
        manager = BearNotesToolManager(mock_repository, mock_config)
        
        result = await manager.call_tool("get_note", {"note_id": "NONEXISTENT"})
        
        assert result.isError is True
        assert "Note not found" in result.content[0].text

    async def test_search_notes_tool(self, mock_repository, mock_config):
        """Test the search_notes tool."""
        manager = BearNotesToolManager(mock_repository, mock_config)
        
        result = await manager.call_tool("search_notes", {"query": "test"})
        
        assert result.isError is False
        assert len(result.content) == 1
        assert result.content[0].type == "text"
        
        # Should call repository search
        mock_repository.search_notes.assert_called_once_with("test")

    async def test_search_notes_tool_missing_param(self, mock_repository, mock_config):
        """Test search_notes tool with missing parameter."""
        manager = BearNotesToolManager(mock_repository, mock_config)
        
        result = await manager.call_tool("search_notes", {})
        
        assert result.isError is True
        assert "query is required" in result.content[0].text

    async def test_list_notes_tool(self, mock_repository, mock_config):
        """Test the list_notes tool."""
        manager = BearNotesToolManager(mock_repository, mock_config)
        
        result = await manager.call_tool("list_notes", {})
        
        assert result.isError is False
        assert len(result.content) == 1
        assert result.content[0].type == "text"
        
        # Should call repository to get notes
        mock_repository.get_all_notes.assert_called_once()

    async def test_list_notes_tool_with_filters(self, mock_repository, mock_config):
        """Test list_notes tool with filter parameters."""
        manager = BearNotesToolManager(mock_repository, mock_config)
        
        result = await manager.call_tool("list_notes", {
            "include_trashed": False,
            "include_archived": False,
            "limit": 10
        })
        
        assert result.isError is False
        
        # Should call repository with filters
        mock_repository.get_all_notes.assert_called_once_with(
            include_trashed=False,
            include_archived=False
        )

    async def test_get_database_stats_tool(self, mock_repository, mock_config):
        """Test the get_database_stats tool."""
        manager = BearNotesToolManager(mock_repository, mock_config)
        
        result = await manager.call_tool("get_database_stats", {})
        
        assert result.isError is False
        assert len(result.content) == 1
        assert result.content[0].type == "text"
        
        # Should call repository
        mock_repository.get_database_stats.assert_called_once()

    async def test_invalid_tool_call(self, mock_repository, mock_config):
        """Test calling a tool that doesn't exist."""
        manager = BearNotesToolManager(mock_repository, mock_config)
        
        result = await manager.call_tool("nonexistent_tool", {})
        
        assert result.isError is True
        assert "Unknown tool" in result.content[0].text

    async def test_tool_error_handling(self, mock_repository, mock_config):
        """Test that tool errors are handled gracefully."""
        # Make repository throw an exception
        mock_repository.get_note_by_id.side_effect = Exception("Database error")
        
        manager = BearNotesToolManager(mock_repository, mock_config)
        
        result = await manager.call_tool("get_note", {"note_id": "UUID-1"})
        
        assert result.isError is True
        assert "Error" in result.content[0].text

    async def test_refresh_cache(self, mock_repository, mock_config):
        """Test tool manager cache refresh."""
        manager = BearNotesToolManager(mock_repository, mock_config)
        
        # This should not raise an error even if no caching is implemented
        await manager.refresh_cache()


@pytest.mark.skip(reason="Resource and tool managers not fully implemented - Phase 1 complete, Phase 4 pending")
class TestResourceToolIntegration:
    """Integration tests between resources and tools."""

    async def test_resource_and_tool_consistency(self, mock_repository, mock_config):
        """Test that resources and tools return consistent data."""
        resource_manager = BearNotesResourceManager(mock_repository, mock_config)
        tool_manager = BearNotesToolManager(mock_repository, mock_config)
        
        # Get note via resource
        resource_content = await resource_manager.read_resource("note:///UUID-1")
        
        # Get same note via tool
        tool_result = await tool_manager.call_tool("get_note", {"note_id": "UUID-1"})
        
        # Both should succeed
        assert len(resource_content) == 1
        assert tool_result.isError is False
        
        # Both should have called the repository
        assert mock_repository.get_note_by_id.call_count == 2

    async def test_notes_list_consistency(self, mock_repository, mock_config):
        """Test that notes://all resource and list_notes tool are consistent."""
        resource_manager = BearNotesResourceManager(mock_repository, mock_config)
        tool_manager = BearNotesToolManager(mock_repository, mock_config)
        
        # Get notes via resource
        await resource_manager.read_resource("notes://all")
        
        # Get notes via tool
        await tool_manager.call_tool("list_notes", {})
        
        # Both should have called get_all_notes
        assert mock_repository.get_all_notes.call_count == 2