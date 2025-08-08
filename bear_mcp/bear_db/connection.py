"""Database connection and access layer for Bear Notes."""

import sqlite3
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import AsyncGenerator, Dict, Generator, List, Optional, Any

import structlog

from bear_mcp.config.models import BearDatabaseConfig
from bear_mcp.bear_db.models import BearNote, BearTag, NoteMetadata, DatabaseStats

logger = structlog.get_logger()


class BearDatabaseError(Exception):
    """Custom exception for Bear database errors."""
    pass


class BearDatabaseConnection:
    """Manages connection to the Bear Notes SQLite database."""
    
    def __init__(self, config: BearDatabaseConfig):
        """Initialize the database connection manager.
        
        Args:
            config: Database configuration
            
        Raises:
            BearDatabaseError: If database path is invalid or inaccessible
        """
        self.config = config
        self._connection: Optional[sqlite3.Connection] = None
        
        # Validate database path
        if not self.config.path:
            raise BearDatabaseError("Database path not configured")
        
        if not self.config.path.exists():
            raise BearDatabaseError(f"Database file not found: {self.config.path}")
        
        if not self.config.path.is_file():
            raise BearDatabaseError(f"Database path is not a file: {self.config.path}")
    
    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a database connection (synchronous context manager).
        
        Yields:
            sqlite3.Connection: Database connection
            
        Raises:
            BearDatabaseError: If connection fails
        """
        connection = None
        try:
            # Configure connection
            connection = sqlite3.connect(
                database=str(self.config.path),
                timeout=self.config.timeout,
                check_same_thread=self.config.check_same_thread,
            )
            
            # Enable read-only mode if configured
            if self.config.read_only:
                connection.execute("PRAGMA query_only = ON")
            
            # Configure row factory for easier access
            connection.row_factory = sqlite3.Row
            
            # Enable foreign keys
            connection.execute("PRAGMA foreign_keys = ON")
            
            logger.debug("Database connection established", path=str(self.config.path))
            yield connection
            
        except sqlite3.Error as e:
            logger.error("Database connection error", error=str(e), path=str(self.config.path))
            raise BearDatabaseError(f"Failed to connect to database: {e}")
        finally:
            if connection:
                connection.close()
                logger.debug("Database connection closed")
    
    def test_connection(self) -> bool:
        """Test if the database connection is working.
        
        Returns:
            bool: True if connection is successful
        """
        try:
            with self.get_connection() as conn:
                # Test basic query
                cursor = conn.execute("SELECT COUNT(*) FROM sqlite_master")
                cursor.fetchone()
                return True
        except Exception as e:
            logger.error("Database connection test failed", error=str(e))
            return False
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get information about the database.
        
        Returns:
            Dict containing database information
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT name, type, sql 
                    FROM sqlite_master 
                    WHERE type IN ('table', 'index')
                    ORDER BY type, name
                """)
                
                schema_info = []
                for row in cursor:
                    schema_info.append({
                        "name": row["name"],
                        "type": row["type"], 
                        "sql": row["sql"]
                    })
                
                # Get database file info
                db_stat = self.config.path.stat()
                
                return {
                    "path": str(self.config.path),
                    "size_bytes": db_stat.st_size,
                    "modified": db_stat.st_mtime,
                    "schema": schema_info,
                    "read_only": self.config.read_only
                }
                
        except Exception as e:
            logger.error("Failed to get database info", error=str(e))
            raise BearDatabaseError(f"Failed to get database info: {e}")
    
    def execute_query(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """Execute a SELECT query and return results.
        
        Args:
            query: SQL query string
            params: Optional query parameters
            
        Returns:
            List of dictionaries representing rows
            
        Raises:
            BearDatabaseError: If query fails
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.execute(query, params or ())
                
                # Convert sqlite3.Row objects to dictionaries
                results = []
                for row in cursor:
                    results.append(dict(row))
                
                return results
                
        except sqlite3.Error as e:
            logger.error("Query execution failed", query=query, error=str(e))
            raise BearDatabaseError(f"Query failed: {e}")
    
    def get_table_columns(self, table_name: str) -> List[Dict[str, Any]]:
        """Get column information for a table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            List of column information dictionaries
        """
        query = f"PRAGMA table_info({table_name})"
        return self.execute_query(query)
    
    def list_tables(self) -> List[str]:
        """Get list of all tables in the database.
        
        Returns:
            List of table names
        """
        query = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        results = self.execute_query(query)
        return [row["name"] for row in results]


class BearDatabaseRepository:
    """High-level repository for accessing Bear Notes data."""
    
    def __init__(self, connection: BearDatabaseConnection):
        """Initialize the repository.
        
        Args:
            connection: Database connection manager
        """
        self.connection = connection
        self._cached_schema: Optional[Dict[str, List[Dict[str, Any]]]] = None
    
    def get_notes(
        self, 
        include_trashed: bool = False, 
        include_archived: bool = False,
        limit: Optional[int] = None
    ) -> List[BearNote]:
        """Get all notes from the database.
        
        Args:
            include_trashed: Whether to include trashed notes
            include_archived: Whether to include archived notes
            limit: Maximum number of notes to return
            
        Returns:
            List of BearNote objects
        """
        # Build WHERE clause
        conditions = []
        if not include_trashed:
            conditions.append("(ZTRASHED IS NULL OR ZTRASHED = 0)")
        if not include_archived:
            conditions.append("(ZARCHIVED IS NULL OR ZARCHIVED = 0)")
        
        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)
        
        # Build LIMIT clause
        limit_clause = f"LIMIT {limit}" if limit else ""
        
        query = f"""
            SELECT 
                Z_PK as z_pk,
                ZUNIQUEIDENTIFIER as zuniqueidentifier,
                ZTITLE as ztitle,
                ZTEXT as ztext,
                ZCREATIONDATE as zcreationdate,
                ZMODIFICATIONDATE as zmodificationdate,
                ZTRASHED as ztrashed,
                ZARCHIVED as zarchived,
                ZPINNED as zpinned,
                ZLASTEDITINGDEVICE as zlasteditingdevice
            FROM ZSFNOTE 
            {where_clause}
            ORDER BY ZMODIFICATIONDATE DESC
            {limit_clause}
        """
        
        results = self.connection.execute_query(query)
        
        notes = []
        for row in results:
            # Convert boolean fields
            row["ztrashed"] = bool(row.get("ztrashed", 0))
            row["zarchived"] = bool(row.get("zarchived", 0))
            row["zpinned"] = bool(row.get("zpinned", 0))
            
            notes.append(BearNote(**row))
        
        return notes
    
    def get_note_by_id(self, note_id: str) -> Optional[BearNote]:
        """Get a specific note by its unique identifier.
        
        Args:
            note_id: The unique identifier of the note
            
        Returns:
            BearNote object or None if not found
        """
        query = """
            SELECT 
                Z_PK as z_pk,
                ZUNIQUEIDENTIFIER as zuniqueidentifier,
                ZTITLE as ztitle,
                ZTEXT as ztext,
                ZCREATIONDATE as zcreationdate,
                ZMODIFICATIONDATE as zmodificationdate,
                ZTRASHED as ztrashed,
                ZARCHIVED as zarchived,
                ZPINNED as zpinned,
                ZLASTEDITINGDEVICE as zlasteditingdevice
            FROM ZSFNOTE 
            WHERE ZUNIQUEIDENTIFIER = ?
        """
        
        results = self.connection.execute_query(query, (note_id,))
        
        if not results:
            return None
        
        row = results[0]
        # Convert boolean fields
        row["ztrashed"] = bool(row.get("ztrashed", 0))
        row["zarchived"] = bool(row.get("zarchived", 0))
        row["zpinned"] = bool(row.get("zpinned", 0))
        
        return BearNote(**row)
    
    def get_note_metadata(
        self, 
        include_trashed: bool = False, 
        include_archived: bool = False
    ) -> List[NoteMetadata]:
        """Get simplified note metadata for all notes.
        
        Args:
            include_trashed: Whether to include trashed notes
            include_archived: Whether to include archived notes
            
        Returns:
            List of NoteMetadata objects
        """
        notes = self.get_notes(include_trashed, include_archived)
        
        metadata_list = []
        for note in notes:
            # Calculate word count (approximate)
            word_count = len(note.ztext.split()) if note.ztext else 0
            
            metadata = NoteMetadata(
                id=note.zuniqueidentifier,
                title=note.ztitle or "Untitled",
                creation_date=note.creation_date,
                modification_date=note.modification_date,
                tags=[],  # TODO: Implement tag loading
                is_pinned=note.zpinned,
                is_trashed=note.ztrashed,
                is_archived=note.zarchived,
                word_count=word_count
            )
            metadata_list.append(metadata)
        
        return metadata_list
    
    def get_database_stats(self) -> DatabaseStats:
        """Get statistics about the database.
        
        Returns:
            DatabaseStats object
        """
        # Count notes by status
        stats_query = """
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN (ZTRASHED IS NULL OR ZTRASHED = 0) AND (ZARCHIVED IS NULL OR ZARCHIVED = 0) THEN 1 ELSE 0 END) as active,
                SUM(CASE WHEN ZTRASHED = 1 THEN 1 ELSE 0 END) as trashed,
                SUM(CASE WHEN ZARCHIVED = 1 THEN 1 ELSE 0 END) as archived
            FROM ZSFNOTE
        """
        
        results = self.connection.execute_query(stats_query)
        note_stats = results[0] if results else {}
        
        # Count tags (if table exists)
        try:
            tag_query = "SELECT COUNT(*) as total FROM ZSFNOTETAG"
            tag_results = self.connection.execute_query(tag_query)
            total_tags = tag_results[0]["total"] if tag_results else 0
        except BearDatabaseError:
            total_tags = 0
        
        return DatabaseStats(
            total_notes=note_stats.get("total", 0),
            active_notes=note_stats.get("active", 0),
            trashed_notes=note_stats.get("trashed", 0),
            archived_notes=note_stats.get("archived", 0),
            total_tags=total_tags,
        )