"""Command line interface for Bear MCP Server."""

import argparse
import json
import sys
from pathlib import Path

import structlog

from bear_mcp.config.settings import load_config, create_default_config_file
from bear_mcp.bear_db.connection import BearDatabaseConnection, BearDatabaseRepository
from bear_mcp.utils import setup_logging, validate_bear_installation, get_bear_database_paths

logger = structlog.get_logger()


def cmd_test_db(args):
    """Test database connection."""
    config = load_config()
    setup_logging(config.logging)
    
    if not config.bear_db.path:
        logger.error("No database path configured")
        sys.exit(1)
    
    try:
        db_conn = BearDatabaseConnection(config.bear_db)
        
        if db_conn.test_connection():
            logger.info("Database connection successful")
            
            # Get database info
            db_info = db_conn.get_database_info()
            logger.info("Database info", **db_info)
            
            # Get some basic stats
            repository = BearDatabaseRepository(db_conn)
            stats = repository.get_database_stats()
            logger.info("Database stats", **stats.model_dump())
            
        else:
            logger.error("Database connection failed")
            sys.exit(1)
            
    except Exception as e:
        logger.error("Database test failed", error=str(e))
        sys.exit(1)


def cmd_list_notes(args):
    """List notes from the database."""
    config = load_config()
    setup_logging(config.logging)
    
    try:
        db_conn = BearDatabaseConnection(config.bear_db)
        repository = BearDatabaseRepository(db_conn)
        
        notes = repository.get_notes(
            include_trashed=args.include_trashed,
            include_archived=args.include_archived,
            limit=args.limit
        )
        
        for note in notes:
            print(f"ID: {note.zuniqueidentifier}")
            print(f"Title: {note.ztitle or 'Untitled'}")
            print(f"Created: {note.creation_date}")
            print(f"Modified: {note.modification_date}")
            print("---")
            
    except Exception as e:
        logger.error("Failed to list notes", error=str(e))
        sys.exit(1)


def cmd_find_db(args):
    """Find Bear database files."""
    logger.info("Searching for Bear database files...")
    
    paths = get_bear_database_paths()
    found_paths = []
    
    for path in paths:
        if path.exists():
            found_paths.append(path)
            logger.info("Found database", path=str(path))
    
    if not found_paths:
        logger.warning("No Bear database files found")
        
        # Check if Bear is installed
        if validate_bear_installation():
            logger.info("Bear is installed but database not found")
            logger.info("Make sure Bear has been run at least once")
        else:
            logger.warning("Bear does not appear to be installed")
    
    return found_paths


def cmd_create_config(args):
    """Create a default configuration file."""
    config_path = Path(args.output)
    
    if config_path.exists() and not args.force:
        logger.error("Configuration file already exists", path=str(config_path))
        logger.info("Use --force to overwrite")
        sys.exit(1)
    
    try:
        create_default_config_file(config_path)
        logger.info("Created configuration file", path=str(config_path))
    except Exception as e:
        logger.error("Failed to create configuration file", error=str(e))
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Bear MCP Server CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Test database command
    test_db_parser = subparsers.add_parser("test-db", help="Test database connection")
    
    # List notes command
    list_notes_parser = subparsers.add_parser("list-notes", help="List notes from database")
    list_notes_parser.add_argument("--limit", type=int, default=10, help="Maximum notes to list")
    list_notes_parser.add_argument("--include-trashed", action="store_true", help="Include trashed notes")
    list_notes_parser.add_argument("--include-archived", action="store_true", help="Include archived notes")
    
    # Find database command
    find_db_parser = subparsers.add_parser("find-db", help="Find Bear database files")
    
    # Create config command
    create_config_parser = subparsers.add_parser("create-config", help="Create default configuration file")
    create_config_parser.add_argument("--output", "-o", default="config/config.yaml", help="Output path")
    create_config_parser.add_argument("--force", action="store_true", help="Overwrite existing file")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Map commands to functions
    commands = {
        "test-db": cmd_test_db,
        "list-notes": cmd_list_notes,
        "find-db": cmd_find_db,
        "create-config": cmd_create_config,
    }
    
    command_func = commands.get(args.command)
    if command_func:
        command_func(args)
    else:
        logger.error("Unknown command", command=args.command)
        sys.exit(1)


if __name__ == "__main__":
    main()