"""Utility functions for Bear MCP Server."""

import sys
from pathlib import Path

import structlog


def setup_logging(config) -> None:
    """Set up structured logging based on configuration.
    
    Args:
        config: Logging configuration
    """
    # Configure structlog
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    
    if config.format.lower() == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Set log level
    import logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, config.level),
    )


def validate_bear_installation() -> bool:
    """Check if Bear is installed and accessible.
    
    Returns:
        bool: True if Bear appears to be installed
    """
    # Check common Bear installation paths
    bear_paths = [
        Path("/Applications/Bear.app"),
        Path.home() / "Applications" / "Bear.app",
    ]
    
    for path in bear_paths:
        if path.exists():
            return True
    
    return False


def get_bear_database_paths() -> list[Path]:
    """Get potential Bear database paths.
    
    Returns:
        List of potential database file paths
    """
    potential_paths = [
        # Bear 2.x
        Path.home() / "Library" / "Group Containers" / "9K33E3U3T4.net.shinyfrog.bear" / "Application Data" / "database.sqlite",
        Path.home() / "Library" / "Containers" / "net.shinyfrog.bear" / "Data" / "Documents" / "Application Data" / "database.sqlite",
        
        # Legacy Bear 1.x (for reference)
        Path.home() / "Library" / "Containers" / "net.shinyfrog.bear" / "Data" / "Library" / "Application Support" / "database.sqlite",
    ]
    
    return potential_paths