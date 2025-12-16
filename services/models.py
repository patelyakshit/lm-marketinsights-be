"""
Database models for ADK session storage.

Simplified async-compatible models for storing sessions and events.
"""

from ast import List
from sqlalchemy import Column, String, DateTime, JSON, ForeignKey, ForeignKeyConstraint, Index
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase, relationship
from datetime import datetime, timezone


class Base(AsyncAttrs, DeclarativeBase):
    """Base class for all database models."""
    pass


class SessionModel(Base):
    """Store ADK session data."""
    __tablename__ = "adk_sessions"
    
    # Composite primary key (matches ADK pattern)
    app_name = Column(String(256), primary_key=True)
    user_id = Column(String(256), primary_key=True)
    session_id = Column(String(256), primary_key=True)
    
    # Session data
    state = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Relationship to events
    events = relationship("EventModel", back_populates="session", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_session_user', 'app_name', 'user_id'),
        Index('idx_session_updated', 'updated_at'),
    )


class EventModel(Base):
    """Store ADK events."""
    __tablename__ = "adk_events"
    
    id = Column(String(256), primary_key=True)
    
    # Foreign key to session
    app_name = Column(String(256), nullable=False)
    user_id = Column(String(256), nullable=False)
    session_id = Column(String(256), nullable=False)
    
    # Event data
    event_data = Column(JSON, nullable=False)  # Serialized Event object
    timestamp = Column(DateTime(timezone=True), nullable=False)
    
    # Relationship
    session = relationship("SessionModel", back_populates="events")
    
    __table_args__ = (
        ForeignKeyConstraint(
            ['app_name', 'user_id', 'session_id'], 
            ['adk_sessions.app_name', 'adk_sessions.user_id', 'adk_sessions.session_id'],
            ondelete='CASCADE'
        ),
        Index('idx_event_session', 'app_name', 'user_id', 'session_id'),
        Index('idx_event_timestamp', 'timestamp'),
    )