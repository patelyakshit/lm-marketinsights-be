"""
Async database wrapper for ADK session storage.

This service provides async operations with proper connection management,
ensuring connections are always closed after each transaction.
"""

import logging
from typing import Optional
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from google.adk.sessions import Session as ADKSession
from google.adk.events import Event

from .models import Base, SessionModel, EventModel

logger = logging.getLogger(__name__)


class SessionStorageService:
    """
    Async database wrapper for ADK session storage.
    
    Key features:
    - Async operations (non-blocking)
    - Automatic connection cleanup via context managers
    - Compatible interface with ADK DatabaseSessionService
    - Proper connection pooling with explicit management
    """
    
    def __init__(
        self,
        database_url: str,
        pool_size: int = 10,
        max_overflow: int = 10,
        pool_recycle: int = 600,
        pool_pre_ping: bool = True,
        pool_timeout: int = 30,
        connection_timeout: int = 60,
        command_timeout: int = 60,
        echo: bool = False,
    ):
        """Initialize async engine with connection pooling."""
        
        # Convert sync URL to async (postgresql:// -> postgresql+asyncpg://)
        # Handle various PostgreSQL URL formats and replace any existing driver
        if database_url.startswith("postgresql+asyncpg://"):
            # Already using async driver, no conversion needed
            pass
        elif database_url.startswith("postgresql+psycopg2://"):
            # Replace psycopg2 with asyncpg
            database_url = database_url.replace("postgresql+psycopg2://", "postgresql+asyncpg://", 1)
        elif database_url.startswith("postgresql://"):
            # Standard postgresql:// URL, add asyncpg driver
            database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
        elif database_url.startswith("postgres://"):
            # Short form postgres:// URL, convert to async
            database_url = database_url.replace("postgres://", "postgresql+asyncpg://", 1)
        elif "://" in database_url:
            # Check if it's a PostgreSQL URL with some other format
            parts = database_url.split("://", 1)
            if len(parts) == 2:
                scheme, rest = parts
                if scheme in ["postgresql", "postgres"]:
                    database_url = f"postgresql+asyncpg://{rest}"
                elif "+" in scheme:
                    # Has a driver specified, replace it
                    base_scheme = scheme.split("+")[0]
                    if base_scheme in ["postgresql", "postgres"]:
                        database_url = f"postgresql+asyncpg://{rest}"
                    else:
                        raise ValueError(
                            f"Unsupported database URL scheme: {scheme}. "
                            "Only PostgreSQL URLs are supported."
                        )
                else:
                    raise ValueError(
                        f"Unsupported database URL scheme: {scheme}. "
                        "Only PostgreSQL URLs are supported. Use postgresql:// or postgres://"
                    )
        else:
            raise ValueError(
                f"Invalid database URL format: {database_url}. "
                "Expected format: postgresql://user:pass@host:port/dbname"
            )
        
        self.engine = create_async_engine(
            database_url,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_recycle=pool_recycle,
            pool_pre_ping=pool_pre_ping,
            pool_timeout=pool_timeout,
            echo=echo,
            connect_args={
                "server_settings": {
                    "jit": "off",  # Disable JIT for faster simple queries
                },
                "command_timeout": command_timeout,  # Timeout for individual commands/queries
                "timeout": connection_timeout,  # Timeout for establishing connection (including ping)
            },
        )
        
        self.async_session_factory = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,  # Keep objects usable after commit
        )
        
        logger.info(f"SessionStorageService initialized with pool_size={pool_size}")
    
    @asynccontextmanager
    async def get_db_session(self):
        """
        Context manager for database sessions.
        Automatically closes connection after use.
        """
        async with self.async_session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    async def initialize_tables(self):
        """Create tables if they don't exist."""
        from sqlalchemy.exc import IntegrityError
        from sqlalchemy import text
        
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all, checkfirst=True)
            logger.info("Database tables initialized")
        except IntegrityError as e:
            # Handle case where PostgreSQL types/tables already exist
            error_str = str(e).lower()
            if "already exists" in error_str or "duplicate key" in error_str or "pg_type_typname_nsp_index" in error_str:
                logger.warning(f"Database objects may already exist: {e}. Verifying tables...")
                # Verify tables actually exist
                try:
                    async with self.engine.connect() as conn:
                        # Check if tables exist by querying information_schema
                        result = await conn.execute(text("""
                            SELECT table_name 
                            FROM information_schema.tables 
                            WHERE table_schema = 'public' 
                            AND table_name IN ('adk_sessions', 'adk_events')
                        """))
                        existing_tables = [row[0] for row in result]
                        expected_tables = ["adk_sessions", "adk_events"]
                        missing_tables = [t for t in expected_tables if t not in existing_tables]
                        
                        if missing_tables:
                            logger.error(f"Some tables are missing: {missing_tables}. Re-raising original error.")
                            raise
                        logger.info(f"Database tables verified to exist: {existing_tables}")
                except Exception as verify_error:
                    logger.error(f"Error verifying tables: {verify_error}")
                    raise e from verify_error
            else:
                # Re-raise if it's a different IntegrityError
                raise
        except Exception as e:
            # Re-raise any other errors
            logger.error(f"Unexpected error initializing tables: {e}")
            raise
    
    async def get_session(
        self,
        app_name: str,
        user_id: str,
        session_id: str,
    ) -> Optional[ADKSession]:
        """
        Load session from database.
        Returns None if not found.
        """
        async with self.get_db_session() as db:
            stmt = (
                select(SessionModel)
                .options(selectinload(SessionModel.events))
                .where(
                    SessionModel.app_name == app_name,
                    SessionModel.user_id == user_id,
                    SessionModel.session_id == session_id,
                )
            )
            result = await db.execute(stmt)
            session_model = result.scalar_one_or_none()
            
            if not session_model:
                return None
            
            # Convert to ADK Session object
            adk_session = ADKSession(
                id=session_model.session_id,
                app_name=session_model.app_name,
                user_id=session_model.user_id,
                state=session_model.state or {},
                events=[],  # Events loaded separately if needed
                last_update_time=session_model.updated_at.timestamp(),
            )
            
            # Load events
            for event_model in session_model.events:
                try:
                    # Deserialize event from JSON
                    event = Event.model_validate(event_model.event_data)
                    adk_session.events.append(event)
                except Exception as e:
                    logger.warning(f"Failed to deserialize event {event_model.id}: {e}")
                    continue
            
            return adk_session
    
    async def create_session(
        self,
        app_name: str,
        user_id: str,
        session_id: str,
        state: dict = None,
    ) -> ADKSession:
        """
        Create new session in database.
        """
        async with self.get_db_session() as db:
            session_model = SessionModel(
                app_name=app_name,
                user_id=user_id,
                session_id=session_id,
                state=state or {},
            )
            db.add(session_model)
            await db.flush()  # Get timestamps
            
            adk_session = ADKSession(
                id=session_model.session_id,
                app_name=session_model.app_name,
                user_id=session_model.user_id,
                state=session_model.state,
                events=[],
                last_update_time=session_model.updated_at.timestamp(),
            )
            
            return adk_session
    
    async def update_session(
        self,
        session: ADKSession,
    ) -> ADKSession:
        """
        Update session state in database.
        Connection automatically closed after transaction.
        """
        async with self.get_db_session() as db:
            # Get session model
            stmt = select(SessionModel).where(
                SessionModel.app_name == session.app_name,
                SessionModel.user_id == session.user_id,
                SessionModel.session_id == session.id,
            )
            result = await db.execute(stmt)
            session_model = result.scalar_one()
            
            # Update state
            session_model.state = session.state
            
            # Update session timestamp
            session_model.updated_at = datetime.now(timezone.utc)
            
            await db.flush()
            
            # Update in-memory session
            session.last_update_time = session_model.updated_at.timestamp()
        
        return session
    
    async def append_event(
        self,
        session: ADKSession,
        event: Event,
    ) -> Event:
        """
        Append event to session and update state.
        Connection automatically closed after transaction.
        """
        async with self.get_db_session() as db:
            # Update session state
            stmt = select(SessionModel).where(
                SessionModel.app_name == session.app_name,
                SessionModel.user_id == session.user_id,
                SessionModel.session_id == session.id,
            )
            result = await db.execute(stmt)
            session_model = result.scalar_one()
            
            # Update state from event
            if event.actions and event.actions.state_delta:
                session_model.state.update(event.actions.state_delta)
            
            # Serialize event to JSON using Pydantic model_dump
            event_data = event.model_dump(mode="json", exclude_none=True)
            
            # Save event
            event_model = EventModel(
                id=event.id,
                app_name=session.app_name,
                user_id=session.user_id,
                session_id=session.id,
                event_data=event_data,
                timestamp=datetime.fromtimestamp(event.timestamp, tz=timezone.utc),
            )
            db.add(event_model)
            
            # Update session timestamp
            session_model.updated_at = datetime.now(timezone.utc)
            
            await db.flush()
            
            # Update in-memory session
            session.last_update_time = session_model.updated_at.timestamp()
            if event.actions and event.actions.state_delta:
                session.state.update(event.actions.state_delta)
        
        return event
    
    async def close(self):
        """Dispose connection pool."""
        await self.engine.dispose()
        logger.info("SessionStorageService connection pool disposed")

