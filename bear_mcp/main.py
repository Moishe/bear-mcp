"""Main entry point for Bear MCP Server."""

import asyncio
import sys
from pathlib import Path

import structlog

from bear_mcp.config.settings import load_config
from bear_mcp.mcp_server.server import BearMCPServer
from bear_mcp.utils import setup_logging

logger = structlog.get_logger()


async def main_async() -> None:
    """Main entry point for the Bear MCP Server."""
    try:
        # Load configuration
        config = load_config()
        
        # Set up logging
        setup_logging(config.logging)
        logger.info("Configuration loaded", config=config.model_dump())
        
        # Initialize and start the MCP server
        server = BearMCPServer(config)
        await server.run()
        
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error("Server error", error=str(e), exc_info=True)
        sys.exit(1)


def main() -> None:
    """Synchronous wrapper for main function."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()