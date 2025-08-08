"""MCP Server implementation for Bear Notes."""

import asyncio
from typing import Any, Dict, List, Optional

import structlog
from mcp import ClientSession
from mcp.server import Server, InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource, 
    Tool, 
    TextContent, 
    CallToolResult,
    ListResourcesResult,
    ListToolsResult,
    ReadResourceResult,
    ServerCapabilities,
    ToolsCapability,
    ResourcesCapability,
)

from bear_mcp.config.models import BearMCPConfig
from bear_mcp.bear_db.connection import BearDatabaseConnection, BearDatabaseRepository
from bear_mcp.bear_db.monitor import BearDatabaseMonitor
from bear_mcp.mcp_server.resources import BearNotesResourceManager
from bear_mcp.mcp_server.tools import BearNotesToolManager

logger = structlog.get_logger()


class BearMCPServer:
    """Main MCP Server for Bear Notes."""
    
    def __init__(self, config: BearMCPConfig):
        """Initialize the Bear MCP Server.
        
        Args:
            config: Server configuration
        """
        self.config = config
        
        # Core components
        self.db_connection: Optional[BearDatabaseConnection] = None
        self.db_repository: Optional[BearDatabaseRepository] = None
        self.db_monitor: Optional[BearDatabaseMonitor] = None
        
        # MCP components
        self.server = Server(self.config.mcp_server.name)
        self.resource_manager: Optional[BearNotesResourceManager] = None
        self.tool_manager: Optional[BearNotesToolManager] = None
        
        # State
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize all server components."""
        if self._initialized:
            logger.debug("Server already initialized")
            return
        
        logger.info("Initializing Bear MCP Server")
        
        try:
            # Initialize database connection
            self.db_connection = BearDatabaseConnection(self.config.bear_db)
            
            # Test database connection
            if not self.db_connection.test_connection():
                raise RuntimeError("Failed to connect to Bear database")
            
            logger.info("Database connection established")
            
            # Initialize repository
            self.db_repository = BearDatabaseRepository(self.db_connection)
            
            # Initialize resource and tool managers
            self.resource_manager = BearNotesResourceManager(
                self.db_repository, 
                self.config
            )
            self.tool_manager = BearNotesToolManager(
                self.db_repository, 
                self.config
            )
            
            # Set up MCP server handlers
            self._setup_handlers()
            
            # Initialize database monitoring
            if self.config.bear_db.path:
                self.db_monitor = BearDatabaseMonitor(
                    self.config.bear_db.path,
                    self.config.performance.refresh_debounce_seconds
                )
                
                # Add refresh callback
                self.db_monitor.add_callback(self._on_database_change)
                
                # Start monitoring
                self.db_monitor.start()
                logger.info("Database monitoring started")
            
            self._initialized = True
            logger.info("Bear MCP Server initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize server", error=str(e))
            await self.cleanup()
            raise
    
    def _setup_handlers(self) -> None:
        """Set up MCP server request handlers."""
        
        @self.server.list_resources()
        async def list_resources() -> ListResourcesResult:
            """Handle list_resources requests."""
            try:
                if not self.resource_manager:
                    raise RuntimeError("Resource manager not initialized")
                
                resources = await self.resource_manager.list_resources()
                logger.debug("Listed resources", count=len(resources))
                
                return ListResourcesResult(resources=resources)
                
            except Exception as e:
                logger.error("Error listing resources", error=str(e))
                raise
        
        @self.server.read_resource()
        async def read_resource(uri: str) -> ReadResourceResult:
            """Handle read_resource requests."""
            try:
                if not self.resource_manager:
                    raise RuntimeError("Resource manager not initialized")
                
                content = await self.resource_manager.read_resource(uri)
                logger.debug("Read resource", uri=uri)
                
                return ReadResourceResult(contents=content)
                
            except Exception as e:
                logger.error("Error reading resource", uri=uri, error=str(e))
                raise
        
        @self.server.list_tools()
        async def list_tools() -> ListToolsResult:
            """Handle list_tools requests."""
            try:
                if not self.tool_manager:
                    raise RuntimeError("Tool manager not initialized")
                
                tools = await self.tool_manager.list_tools()
                logger.debug("Listed tools", count=len(tools))
                
                return ListToolsResult(tools=tools)
                
            except Exception as e:
                logger.error("Error listing tools", error=str(e))
                raise
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> CallToolResult:
            """Handle call_tool requests."""
            try:
                if not self.tool_manager:
                    raise RuntimeError("Tool manager not initialized")
                
                result = await self.tool_manager.call_tool(name, arguments)
                logger.debug("Called tool", name=name)
                
                return result
                
            except Exception as e:
                logger.error("Error calling tool", name=name, error=str(e))
                raise
    
    def _on_database_change(self) -> None:
        """Handle database change notifications."""
        logger.info("Database changed, refreshing caches")
        
        # Refresh resource manager cache
        if self.resource_manager:
            asyncio.create_task(self.resource_manager.refresh_cache())
        
        # Refresh tool manager cache
        if self.tool_manager:
            asyncio.create_task(self.tool_manager.refresh_cache())
    
    async def run(self) -> None:
        """Run the MCP server."""
        await self.initialize()
        
        logger.info(
            "Starting MCP server", 
            name=self.config.mcp_server.name,
            version=self.config.mcp_server.version
        )
        
        try:
            # Configure initialization options
            initialization_options = InitializationOptions(
                server_name=self.config.mcp_server.name,
                server_version=self.config.mcp_server.version,
                capabilities=ServerCapabilities(
                    tools=ToolsCapability(listChanged=True),
                    resources=ResourcesCapability(subscribe=False, listChanged=True)
                )
            )
            
            # Run the stdio server
            async with stdio_server() as (read_stream, write_stream):
                await self.server.run(
                    read_stream, 
                    write_stream, 
                    initialization_options,
                    raise_exceptions=False
                )
        finally:
            await self.cleanup()
    
    async def cleanup(self) -> None:
        """Clean up server resources."""
        logger.info("Cleaning up server resources")
        
        # Stop database monitoring
        if self.db_monitor:
            self.db_monitor.stop()
            self.db_monitor = None
        
        # Clean up managers
        if self.resource_manager:
            await self.resource_manager.cleanup()
            self.resource_manager = None
        
        if self.tool_manager:
            await self.tool_manager.cleanup()
            self.tool_manager = None
        
        # Close database connection
        self.db_connection = None
        self.db_repository = None
        
        self._initialized = False
        logger.info("Server cleanup completed")
    
    def get_server_info(self) -> Dict[str, Any]:
        """Get server information and status.
        
        Returns:
            Dictionary containing server information
        """
        return {
            "name": self.config.mcp_server.name,
            "version": self.config.mcp_server.version,
            "initialized": self._initialized,
            "database_connected": self.db_connection is not None,
            "monitoring_active": self.db_monitor.is_running() if self.db_monitor else False,
            "config": self.config.model_dump()
        }