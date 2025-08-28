"""File system monitoring for Bear database changes."""

import asyncio
from pathlib import Path
from typing import Callable, Optional, Set

import structlog
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

logger = structlog.get_logger()


class DatabaseChangeHandler(FileSystemEventHandler):
    """Handles file system events for the Bear database."""
    
    def __init__(
        self, 
        database_path: Path, 
        callback: Callable[[], None],
        debounce_seconds: float = 2.0
    ):
        """Initialize the change handler.
        
        Args:
            database_path: Path to the Bear database file
            callback: Function to call when database changes
            debounce_seconds: Time to wait before triggering callback
        """
        super().__init__()
        self.database_path = database_path
        self.callback = callback
        self.debounce_seconds = debounce_seconds
        
        # Debouncing mechanism
        self._pending_refresh = False
        self._refresh_task: Optional[asyncio.Task] = None
        
        # Track relevant file extensions
        self._relevant_extensions = {".sqlite", ".sqlite3", ".db"}
    
    def on_modified(self, event: FileSystemEvent) -> None:
        """Handle file modification events."""
        if event.is_directory:
            return
        
        event_path = Path(event.src_path)
        
        # Check if the modified file is the database file
        if event_path.resolve() == self.database_path.resolve():
            logger.debug("Database file modified", path=event.src_path)
            self._schedule_refresh()
        
        # Also check for related files (WAL, journal, etc.)
        elif event_path.parent == self.database_path.parent:
            if (event_path.suffix in self._relevant_extensions or
                event_path.name.startswith(self.database_path.name)):
                logger.debug("Database-related file modified", path=event.src_path)
                self._schedule_refresh()
    
    def on_moved(self, event: FileSystemEvent) -> None:
        """Handle file move events."""
        if event.is_directory:
            return
        
        # Check if the database was moved or renamed
        old_path = Path(event.src_path)
        new_path = Path(event.dest_path)
        
        if (old_path.resolve() == self.database_path.resolve() or
            new_path.resolve() == self.database_path.resolve()):
            logger.info("Database file moved", old_path=event.src_path, new_path=event.dest_path)
            self._schedule_refresh()
    
    def _schedule_refresh(self) -> None:
        """Schedule a debounced refresh."""
        if self._pending_refresh:
            logger.debug("Refresh already pending, skipping")
            return
        
        self._pending_refresh = True
        
        # Cancel existing refresh task
        if self._refresh_task and not self._refresh_task.done():
            self._refresh_task.cancel()
        
        # Try to schedule new refresh task in the main event loop
        try:
            loop = asyncio.get_running_loop()
            self._refresh_task = loop.create_task(self._debounced_refresh())
        except RuntimeError:
            # No event loop running, call callback directly (synchronous fallback)
            logger.debug("No event loop available, calling callback synchronously")
            self._pending_refresh = False
            self.callback()
    
    async def _debounced_refresh(self) -> None:
        """Execute a debounced refresh after waiting."""
        try:
            await asyncio.sleep(self.debounce_seconds)
            
            logger.info("Triggering database refresh after debounce")
            self.callback()
            
        except asyncio.CancelledError:
            logger.debug("Refresh task cancelled")
        except Exception as e:
            logger.error("Error during database refresh", error=str(e))
        finally:
            self._pending_refresh = False


class BearDatabaseMonitor:
    """Monitors the Bear database for changes and triggers refresh callbacks."""
    
    def __init__(self, database_path: Path, debounce_seconds: float = 2.0):
        """Initialize the database monitor.
        
        Args:
            database_path: Path to the Bear database file
            debounce_seconds: Time to wait before triggering callbacks
        """
        self.database_path = database_path
        self.debounce_seconds = debounce_seconds
        
        # Components
        self._observer: Optional[Observer] = None
        self._event_handler: Optional[DatabaseChangeHandler] = None
        
        # Callbacks
        self._callbacks: Set[Callable[[], None]] = set()
        
        # State
        self._running = False
    
    def add_callback(self, callback: Callable[[], None]) -> None:
        """Add a callback to be triggered on database changes.
        
        Args:
            callback: Function to call when database changes
        """
        self._callbacks.add(callback)
        logger.debug("Added database change callback", callback=callback.__name__)
    
    def remove_callback(self, callback: Callable[[], None]) -> None:
        """Remove a callback.
        
        Args:
            callback: Function to remove
        """
        self._callbacks.discard(callback)
        logger.debug("Removed database change callback", callback=callback.__name__)
    
    def start(self) -> None:
        """Start monitoring the database for changes.
        
        Raises:
            FileNotFoundError: If database file doesn't exist
            RuntimeError: If monitor is already running
        """
        if self._running:
            raise RuntimeError("Monitor is already running")
        
        if not self.database_path.exists():
            raise FileNotFoundError(f"Database file not found: {self.database_path}")
        
        logger.info("Starting database monitor", path=str(self.database_path))
        
        # Create event handler
        self._event_handler = DatabaseChangeHandler(
            self.database_path,
            self._trigger_callbacks,
            self.debounce_seconds
        )
        
        # Create and configure observer
        self._observer = Observer()
        
        # Watch the directory containing the database
        watch_path = self.database_path.parent
        self._observer.schedule(
            self._event_handler,
            str(watch_path),
            recursive=False
        )
        
        # Start the observer
        self._observer.start()
        self._running = True
        
        logger.info("Database monitor started", watch_path=str(watch_path))
    
    def stop(self) -> None:
        """Stop monitoring the database."""
        if not self._running:
            logger.debug("Monitor is not running, nothing to stop")
            return
        
        logger.info("Stopping database monitor")
        
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5.0)
            self._observer = None
        
        self._event_handler = None
        self._running = False
        
        logger.info("Database monitor stopped")
    
    def is_running(self) -> bool:
        """Check if the monitor is currently running.
        
        Returns:
            bool: True if monitoring is active
        """
        return self._running
    
    def _trigger_callbacks(self) -> None:
        """Trigger all registered callbacks."""
        logger.debug("Triggering database change callbacks", count=len(self._callbacks))
        
        for callback in self._callbacks:
            try:
                callback()
                logger.debug("Called database change callback", callback=callback.__name__)
            except Exception as e:
                logger.error(
                    "Error in database change callback", 
                    callback=callback.__name__, 
                    error=str(e)
                )
    
    def __enter__(self) -> "BearDatabaseMonitor":
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()