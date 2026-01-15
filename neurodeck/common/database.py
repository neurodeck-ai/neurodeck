"""SQLite3 database management for agent context persistence."""

import sqlite3
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
from contextlib import contextmanager

from .logging import get_logger

logger = get_logger("database")


class ContextDatabase:
    """
    Manages persistent conversation context using SQLite3.
    
    Each agent's conversation history is stored and survives crashes/restarts.
    Context is isolated by session_id for each orchestrator run.
    """
    
    def __init__(self, db_path: str = "data/context.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.session_id = str(uuid.uuid4())
        self.init_database()
        
        logger.info(f"Context database initialized: {db_path}")
        logger.info(f"Session ID: {self.session_id}")
    
    def init_database(self):
        """Create database tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_context (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_name TEXT NOT NULL,
                    message_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    session_id TEXT NOT NULL,
                    message_id TEXT
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_agent_session 
                ON agent_context(agent_name, session_id)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_timestamp 
                ON agent_context(session_id, timestamp)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_message_id 
                ON agent_context(message_id)
            """)
            
            conn.commit()
    
    @contextmanager
    def get_connection(self):
        """Get database connection with proper error handling."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Enable dict-like access
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def save_message(
        self,
        agent_name: str,
        message_type: str,
        content: str,
        session_id: Optional[str] = None,
        message_id: Optional[str] = None
    ) -> int:
        """
        Save message to agent context.
        
        Args:
            agent_name: Name of the agent
            message_type: Type of message ('user', 'assistant', 'system')
            content: Message content
            session_id: Session ID (defaults to current session)
            message_id: Unique message identifier (for deduplication)
            
        Returns:
            ID of inserted message
        """
        if session_id is None:
            session_id = self.session_id
        
        # Generate message_id if not provided (for backward compatibility)
        if message_id is None:
            message_id = str(uuid.uuid4())
        
        with self.get_connection() as conn:
            cursor = conn.execute(
                """INSERT INTO agent_context 
                   (agent_name, message_type, content, session_id, message_id) 
                   VALUES (?, ?, ?, ?, ?)""",
                (agent_name, message_type, content, session_id, message_id)
            )
            conn.commit()
            
            message_id = cursor.lastrowid
            logger.debug(f"Saved message for {agent_name}: {message_type} (ID: {message_id})")
            return message_id
    
    def load_context(
        self,
        agent_name: str,
        session_id: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Load conversation context for agent.
        
        Args:
            agent_name: Name of the agent
            session_id: Session ID (defaults to current session)
            limit: Maximum number of messages to load (most recent)
            
        Returns:
            List of message dictionaries
        """
        if session_id is None:
            session_id = self.session_id
        
        query = """
            SELECT 
                COALESCE(
                    (SELECT agent_name FROM agent_context ac2 
                     WHERE ac2.message_id = ac1.message_id AND ac2.message_type = 'assistant' LIMIT 1),
                    'user'
                ) as sender,
                message_type, content, timestamp 
            FROM agent_context ac1
            WHERE agent_name = ? AND session_id = ? AND message_id IS NOT NULL
            ORDER BY timestamp ASC
        """
        
        params = [agent_name, session_id]
        
        if limit:
            query = """
                SELECT sender, message_type, content, timestamp FROM (
                    SELECT 
                        COALESCE(
                            (SELECT agent_name FROM agent_context ac2 
                             WHERE ac2.message_id = ac1.message_id AND ac2.message_type = 'assistant' LIMIT 1),
                            'user'
                        ) as sender,
                        message_type, content, timestamp 
                    FROM agent_context ac1
                    WHERE agent_name = ? AND session_id = ? AND message_id IS NOT NULL
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ) 
                ORDER BY timestamp ASC
            """
            params.append(limit)
        
        with self.get_connection() as conn:
            cursor = conn.execute(query, params)
            messages = []
            
            for row in cursor.fetchall():
                messages.append({
                    "sender": row["sender"],
                    "type": row["message_type"],
                    "content": row["content"],
                    "timestamp": row["timestamp"]
                })
            
            logger.info(f"Loaded {len(messages)} messages for {agent_name}")
            return messages
    
    def clear_context(
        self,
        agent_name: str,
        session_id: Optional[str] = None
    ) -> int:
        """
        Clear context for specific agent.
        
        Args:
            agent_name: Name of the agent
            session_id: Session ID (defaults to current session)
            
        Returns:
            Number of messages deleted
        """
        if session_id is None:
            session_id = self.session_id
        
        with self.get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM agent_context WHERE agent_name = ? AND session_id = ?",
                (agent_name, session_id)
            )
            conn.commit()
            
            deleted_count = cursor.rowcount
            logger.info(f"Cleared {deleted_count} messages for {agent_name}")
            return deleted_count
    
    def clear_all_context(self, session_id: Optional[str] = None) -> int:
        """
        Clear all context for session.
        
        Args:
            session_id: Session ID (defaults to current session)
            
        Returns:
            Number of messages deleted
        """
        if session_id is None:
            session_id = self.session_id
        
        with self.get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM agent_context WHERE session_id = ?",
                (session_id,)
            )
            conn.commit()
            
            deleted_count = cursor.rowcount
            logger.info(f"Cleared all context ({deleted_count} messages) for session {session_id}")
            return deleted_count
    
    def get_agent_list(self, session_id: Optional[str] = None) -> List[str]:
        """
        Get list of agents with saved context.
        
        Args:
            session_id: Session ID (defaults to current session)
            
        Returns:
            List of agent names
        """
        if session_id is None:
            session_id = self.session_id
        
        with self.get_connection() as conn:
            cursor = conn.execute(
                """SELECT DISTINCT agent_name 
                   FROM agent_context 
                   WHERE session_id = ? 
                   ORDER BY agent_name""",
                (session_id,)
            )
            
            agents = [row["agent_name"] for row in cursor.fetchall()]
            return agents
    
    def wipe_agent_all_sessions(self, agent_name: str) -> int:
        """
        Clear ALL context for specific agent across ALL sessions.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Number of messages deleted
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM agent_context WHERE agent_name = ?",
                (agent_name,)
            )
            conn.commit()
            
            deleted_count = cursor.rowcount
            logger.info(f"WIPED {deleted_count} messages for {agent_name} across all sessions")
            return deleted_count
    
    def wipe_all_context_all_sessions(self) -> int:
        """
        UTILITY METHOD: Clear ALL context for ALL agents across ALL sessions.
        
        Note: This method is not used by the /wipe command (which delegates to 
        CLEAR_CONTEXT system). Kept for database administration and debugging.
        
        Returns:
            Number of messages deleted
        """
        with self.get_connection() as conn:
            cursor = conn.execute("DELETE FROM agent_context")
            conn.commit()
            
            deleted_count = cursor.rowcount
            logger.info(f"WIPED ALL context ({deleted_count} messages) from database")
            return deleted_count
    
    def get_agent_list_all_sessions(self) -> List[str]:
        """
        UTILITY METHOD: Get list of all agents with saved context across all sessions.
        
        Note: This method is not used by the /wipe command (which uses config-based
        validation). Kept for database introspection and debugging.
        
        Returns:
            List of agent names
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT DISTINCT agent_name FROM agent_context ORDER BY agent_name"
            )
            return [row["agent_name"] for row in cursor.fetchall()]
    
    def load_recent_chat_history(
        self, 
        limit: int = 20, 
        session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Load recent chat history across all agents and users.
        
        Args:
            limit: Maximum number of recent messages to load
            session_id: Session ID (defaults to current session)
            
        Returns:
            List of message dictionaries with sender, content, timestamp
        """
        if session_id is None:
            session_id = self.session_id
        
        query = """
            SELECT agent_name, message_type, content, timestamp 
            FROM agent_context 
            WHERE session_id = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        """
        
        with self.get_connection() as conn:
            cursor = conn.execute(query, [session_id, limit])
            rows = cursor.fetchall()
        
        # Convert to chat message format and reverse to chronological order
        messages = []
        for row in reversed(rows):  # Reverse to get chronological order
            # Determine sender based on message type and agent
            if row['message_type'] == 'user':
                sender = 'user'
            else:  # assistant
                sender = row['agent_name']
            
            messages.append({
                'sender': sender,
                'content': row['content'],
                'timestamp': row['timestamp']
            })
        
        logger.debug(f"Loaded {len(messages)} recent chat messages for session {session_id[:8]}...")
        return messages

    def load_recent_chat_history_all_sessions(
        self, 
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Load recent chat history across all sessions and agents.
        Only includes original messages (not copies stored by receivers).
        
        Args:
            limit: Maximum number of recent messages to load
            
        Returns:
            List of message dictionaries with sender, content, timestamp
        """
        # Use message_id to get unique messages (no more duplicates!)
        query = """
            SELECT DISTINCT 
                COALESCE(
                    (SELECT agent_name FROM agent_context ac2 
                     WHERE ac2.message_id = ac1.message_id AND ac2.message_type = 'assistant' LIMIT 1),
                    'user'
                ) as sender,
                message_type, 
                content, 
                timestamp,
                message_id
            FROM agent_context ac1
            WHERE message_id IS NOT NULL
            ORDER BY timestamp DESC 
            LIMIT ?
        """
        
        with self.get_connection() as conn:
            cursor = conn.execute(query, [limit])
            rows = cursor.fetchall()
        
        # Convert to chat message format and reverse to chronological order
        messages = []
        for row in reversed(rows):  # Reverse to get chronological order
            messages.append({
                'sender': row['sender'],  # Already determined by query
                'content': row['content'],
                'timestamp': row['timestamp']
            })
        
        logger.debug(f"Loaded {len(messages)} recent chat messages from all sessions")
        return messages

    def load_context_all_sessions(
        self,
        agent_name: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Load conversation context for agent across ALL sessions.
        
        Args:
            agent_name: Name of the agent
            limit: Maximum number of messages to load (most recent)
            
        Returns:
            List of message dictionaries
        """
        query = """
            SELECT 
                COALESCE(
                    (SELECT agent_name FROM agent_context ac2 
                     WHERE ac2.message_id = ac1.message_id AND ac2.message_type = 'assistant' LIMIT 1),
                    'user'
                ) as sender,
                message_type, content, timestamp 
            FROM agent_context ac1
            WHERE agent_name = ? AND message_id IS NOT NULL
            ORDER BY timestamp ASC
        """
        
        params = [agent_name]
        
        if limit:
            query = """
                SELECT sender, message_type, content, timestamp FROM (
                    SELECT 
                        COALESCE(
                            (SELECT agent_name FROM agent_context ac2 
                             WHERE ac2.message_id = ac1.message_id AND ac2.message_type = 'assistant' LIMIT 1),
                            'user'
                        ) as sender,
                        message_type, content, timestamp 
                    FROM agent_context ac1
                    WHERE agent_name = ? AND message_id IS NOT NULL
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ) 
                ORDER BY timestamp ASC
            """
            params.append(limit)
        
        with self.get_connection() as conn:
            cursor = conn.execute(query, params)
            messages = []
            
            for row in cursor.fetchall():
                messages.append({
                    "sender": row["sender"],
                    "type": row["message_type"],
                    "content": row["content"],
                    "timestamp": row["timestamp"]
                })
            
            logger.info(f"Loaded {len(messages)} messages for {agent_name} from all sessions")
            return messages

    def get_context_stats(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics about stored context.
        
        Args:
            session_id: Session ID (defaults to current session)
            
        Returns:
            Dictionary with context statistics
        """
        if session_id is None:
            session_id = self.session_id
        
        with self.get_connection() as conn:
            # Total messages
            cursor = conn.execute(
                "SELECT COUNT(*) as total FROM agent_context WHERE session_id = ?",
                (session_id,)
            )
            total_messages = cursor.fetchone()["total"]
            
            # Messages per agent
            cursor = conn.execute(
                """SELECT agent_name, COUNT(*) as count 
                   FROM agent_context 
                   WHERE session_id = ? 
                   GROUP BY agent_name 
                   ORDER BY count DESC""",
                (session_id,)
            )
            agent_counts = {row["agent_name"]: row["count"] for row in cursor.fetchall()}
            
            # Date range
            cursor = conn.execute(
                """SELECT MIN(timestamp) as first, MAX(timestamp) as last 
                   FROM agent_context 
                   WHERE session_id = ?""",
                (session_id,)
            )
            date_range = cursor.fetchone()
            
            return {
                "session_id": session_id,
                "total_messages": total_messages,
                "agent_counts": agent_counts,
                "first_message": date_range["first"],
                "last_message": date_range["last"]
            }
    
    def cleanup_old_sessions(self, days_to_keep: int = 7) -> int:
        """
        Clean up old session data.
        
        Args:
            days_to_keep: Number of days of history to keep
            
        Returns:
            Number of messages deleted
        """
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        with self.get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM agent_context WHERE timestamp < ?",
                (cutoff_date.isoformat(),)
            )
            conn.commit()
            
            deleted_count = cursor.rowcount
            logger.info(f"Cleaned up {deleted_count} old messages (older than {days_to_keep} days)")
            return deleted_count


def test_database():
    """Test database functionality."""
    db = ContextDatabase("test_context.db")
    
    # Test saving messages
    db.save_message("claude", "user", "Hello Claude!")
    db.save_message("claude", "assistant", "Hello! How can I help you?")
    db.save_message("grok", "user", "Hello Grok!")
    db.save_message("grok", "assistant", "Hey there! What's up?")
    
    # Test loading context
    claude_context = db.load_context("claude")
    print(f"Claude context: {len(claude_context)} messages")
    
    grok_context = db.load_context("grok")
    print(f"Grok context: {len(grok_context)} messages")
    
    # Test stats
    stats = db.get_context_stats()
    print(f"Context stats: {stats}")
    
    # Test clearing context
    cleared = db.clear_context("claude")
    print(f"Cleared {cleared} messages for Claude")
    
    # Clean up test file
    import os
    os.unlink("test_context.db")
    print("✅ Database tests completed")


if __name__ == "__main__":
    test_database()