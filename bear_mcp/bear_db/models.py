"""Data models for Bear Notes database entities."""

from datetime import datetime, timedelta
from typing import List, Optional

from pydantic import BaseModel, Field


class BearNote(BaseModel):
    """Represents a note in the Bear database.
    
    Based on the Bear 2.x database schema, notes are stored in the ZSFNOTE table.
    """
    
    # Primary key and identifiers
    z_pk: int = Field(..., description="Primary key in the database")
    zuniqueidentifier: str = Field(..., description="Unique identifier for the note")
    
    # Content fields
    ztitle: Optional[str] = Field(None, description="Note title")
    ztext: Optional[str] = Field(None, description="Full note text content")
    
    # Timestamps (Core Data timestamps - seconds since 2001-01-01)
    zcreationdate: Optional[float] = Field(None, description="Creation timestamp")
    zmodificationdate: Optional[float] = Field(None, description="Last modification timestamp")
    
    # Status and metadata
    ztrashed: bool = Field(default=False, description="Whether the note is in trash")
    zarchived: bool = Field(default=False, description="Whether the note is archived")
    zpinned: bool = Field(default=False, description="Whether the note is pinned")
    
    # Additional metadata
    zlasteditingdevice: Optional[str] = Field(None, description="Device that last edited the note")
    
    @property
    def creation_date(self) -> Optional[datetime]:
        """Convert Core Data timestamp to Python datetime."""
        if self.zcreationdate is None:
            return None
        # Core Data reference date: 2001-01-01 00:00:00 UTC
        reference_date = datetime(2001, 1, 1)
        return reference_date + timedelta(seconds=self.zcreationdate)
    
    @property
    def modification_date(self) -> Optional[datetime]:
        """Convert Core Data timestamp to Python datetime."""
        if self.zmodificationdate is None:
            return None
        reference_date = datetime(2001, 1, 1)
        return reference_date + timedelta(seconds=self.zmodificationdate)
    
    @property
    def is_active(self) -> bool:
        """Check if note is active (not trashed or archived)."""
        return not (self.ztrashed or self.zarchived)
    
    class Config:
        """Pydantic configuration."""
        extra = "allow"  # Allow additional fields from the database


class BearTag(BaseModel):
    """Represents a tag in the Bear database.
    
    Based on the Bear 2.x database schema, tags are stored in the ZSFNOTETAG table.
    """
    
    z_pk: int = Field(..., description="Primary key in the database")
    ztitle: str = Field(..., description="Tag title/name")
    zcreationdate: Optional[float] = Field(None, description="Creation timestamp")
    zmodificationdate: Optional[float] = Field(None, description="Modification timestamp")
    
    @property
    def creation_date(self) -> Optional[datetime]:
        """Convert Core Data timestamp to Python datetime."""
        if self.zcreationdate is None:
            return None
        reference_date = datetime(2001, 1, 1)
        return reference_date + timedelta(seconds=self.zcreationdate)
    
    @property
    def modification_date(self) -> Optional[datetime]:
        """Convert Core Data timestamp to Python datetime."""
        if self.zmodificationdate is None:
            return None
        reference_date = datetime(2001, 1, 1)
        return reference_date + timedelta(seconds=self.zmodificationdate)


class BearNoteTag(BaseModel):
    """Represents the many-to-many relationship between notes and tags.
    
    This is typically stored in a junction table in the Bear database.
    """
    
    note_z_pk: int = Field(..., description="Foreign key to note")
    tag_z_pk: int = Field(..., description="Foreign key to tag")


class NoteMetadata(BaseModel):
    """Simplified note metadata for quick access."""
    
    id: str = Field(..., description="Note unique identifier")
    title: str = Field(..., description="Note title")
    creation_date: Optional[datetime] = Field(None, description="Creation date")
    modification_date: Optional[datetime] = Field(None, description="Modification date")
    tags: List[str] = Field(default_factory=list, description="List of tag names")
    is_pinned: bool = Field(default=False, description="Whether note is pinned")
    is_trashed: bool = Field(default=False, description="Whether note is trashed")
    is_archived: bool = Field(default=False, description="Whether note is archived")
    word_count: int = Field(default=0, description="Approximate word count")


class DatabaseStats(BaseModel):
    """Statistics about the Bear database."""
    
    total_notes: int = Field(default=0, description="Total number of notes")
    active_notes: int = Field(default=0, description="Number of active notes")
    trashed_notes: int = Field(default=0, description="Number of trashed notes")
    archived_notes: int = Field(default=0, description="Number of archived notes")
    total_tags: int = Field(default=0, description="Total number of unique tags")
    last_updated: datetime = Field(default_factory=datetime.now, description="When stats were calculated")