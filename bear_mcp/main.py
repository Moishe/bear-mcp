"""Main entry point for Bear MCP Server."""

import asyncio
import logging
import sys
from pathlib import Path

import structlog

# Configure basic logging to stderr immediately
logging.basicConfig(
    format="%(message)s",
    stream=sys.stderr,
    level=logging.INFO,
)

# Configure structlog to use stderr by default
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

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