"""
Secure filesystem operations tool for NeuroDeck.

Features:
- Path traversal protection
- File size limits
- Operation logging
- Async file operations
"""

import os
import json
import time
import fcntl
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import aiofiles
from ..common.logging import get_logger

logger = get_logger(__name__)

class FileLockRegistry:
    """
    Registry for tracking file locks and retry attempts.
    
    Features:
    - Track which agent holds each file lock
    - Monitor retry frequency per agent/file
    - Calculate smart delays to prevent retry storms
    """
    
    def __init__(self):
        self._lock_holders: Dict[str, str] = {}  # file_path -> agent_name
        self._lock_timestamps: Dict[str, float] = {}  # file_path -> lock_time
        self._retry_counts: Dict[Tuple[str, str], int] = {}  # (agent, file_path) -> retry_count
        self._last_attempt: Dict[Tuple[str, str], float] = {}  # (agent, file_path) -> timestamp
    
    def register_lock(self, file_path: str, agent_name: str):
        """Register that agent_name holds lock on file_path."""
        self._lock_holders[file_path] = agent_name
        self._lock_timestamps[file_path] = time.time()
        logger.debug(f"Lock registered: {agent_name} holds {file_path}")
    
    def release_lock(self, file_path: str):
        """Release lock registration."""
        if file_path in self._lock_holders:
            agent_name = self._lock_holders[file_path]
            del self._lock_holders[file_path]
            del self._lock_timestamps[file_path]
            logger.debug(f"Lock released: {agent_name} released {file_path}")
    
    def get_lock_holder(self, file_path: str) -> Optional[str]:
        """Get name of agent holding lock, or None if unlocked."""
        return self._lock_holders.get(file_path)
    
    def get_lock_duration(self, file_path: str) -> Optional[float]:
        """Get how long lock has been held in seconds."""
        if file_path in self._lock_timestamps:
            return time.time() - self._lock_timestamps[file_path]
        return None
    
    def get_retry_count(self, agent_name: str, file_path: str) -> int:
        """Get current retry count for agent/file combination."""
        key = (agent_name, file_path)
        return self._retry_counts.get(key, 0)
    
    def should_delay_response(self, agent_name: str, file_path: str, 
                            base_delay: float = 1.0, max_delay: float = 30.0) -> float:
        """Calculate delay before responding to agent based on retry history."""
        key = (agent_name, file_path)
        
        # Get current retry count
        retry_count = self._retry_counts.get(key, 0)
        
        # Calculate exponential delay: base_delay * (2 ^ retry_count)
        delay = min(base_delay * (2 ** retry_count), max_delay)
        
        # Increment retry count for next time
        self._retry_counts[key] = retry_count + 1
        self._last_attempt[key] = time.time()
        
        logger.info(f"Smart delay for {agent_name} on {file_path}: {delay:.1f}s (attempt #{retry_count + 1})")
        return delay
    
    def reset_retry_count(self, agent_name: str, file_path: str):
        """Reset retry count when agent successfully acquires lock."""
        key = (agent_name, file_path)
        if key in self._retry_counts:
            old_count = self._retry_counts[key]
            del self._retry_counts[key]
            del self._last_attempt[key]
            logger.debug(f"Reset retry count for {agent_name} on {file_path} (was {old_count})")
    
    def cleanup_stale_locks(self, max_age: float = 300.0):
        """Clean up locks held longer than max_age seconds."""
        current_time = time.time()
        stale_paths = []
        
        for file_path, lock_time in self._lock_timestamps.items():
            if current_time - lock_time > max_age:
                stale_paths.append(file_path)
        
        for file_path in stale_paths:
            agent_name = self._lock_holders.get(file_path, "unknown")
            logger.warning(f"Cleaning up stale lock on {file_path} held by {agent_name}")
            self.release_lock(file_path)

class FilesystemTool:
    """
    Secure filesystem operations tool.
    
    Features:
    - Path traversal protection
    - File size limits
    - Operation logging
    - Async file operations
    """
    
    def __init__(self, allowed_paths: List[str], max_file_size: int):
        """
        Initialize filesystem tool.
        
        Args:
            allowed_paths: List of allowed directory paths
            max_file_size: Maximum file size in bytes
        """
        self.allowed_paths = [Path(p).resolve() for p in allowed_paths]
        self.max_file_size = max_file_size
        self._lock_registry = FileLockRegistry()
        logger.info(f"Filesystem tool initialized with paths: {allowed_paths}")
    
    def _validate_path(self, path: str) -> Path:
        """Validate that path is within allowed directories."""
        target_path = Path(path).resolve()
        
        # Check if path is within any allowed directory
        for allowed_path in self.allowed_paths:
            try:
                target_path.relative_to(allowed_path)
                return target_path
            except ValueError:
                continue
        
        raise PermissionError(f"Access denied: {path} is outside allowed directories")
    
    async def execute(self, action: str, path: str, content: Optional[str] = None,
                     agent_name: Optional[str] = None, use_locking: bool = True,
                     base_delay: float = 1.0, max_delay: float = 30.0, **kwargs) -> Any:
        """
        Execute filesystem operation.

        Args:
            action: Operation to perform (read, write, append, list, delete, lock_status)
            path: File or directory path
            content: Content for write/append operations
            agent_name: Name of agent performing operation (for lock tracking)

        Returns:
            Operation result
        """
        # Validate path
        target_path = self._validate_path(path)

        logger.info(f"Executing filesystem {action} on {target_path} (agent: {agent_name or 'unknown'})")

        if action == "read":
            return await self._read_file(target_path)
        elif action == "write":
            return await self._write_file_with_locking(target_path, content, agent_name,
                                                     base_delay, max_delay, use_locking)
        elif action == "append":
            return await self._append_file_with_locking(target_path, content, agent_name,
                                                       base_delay, max_delay, use_locking)
        elif action == "list":
            return await self._list_directory(target_path)
        elif action == "delete":
            return await self._delete_file_with_locking(target_path, agent_name, base_delay, max_delay)
        elif action == "lock_status":
            return await self._get_lock_status(target_path, agent_name)
        else:
            raise ValueError(f"Unknown filesystem action: {action}")
    
    async def _read_file(self, path: Path) -> str:
        """Read file with size validation."""
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        if not path.is_file():
            raise ValueError(f"Not a file: {path}")
        
        # Check file size
        file_size = path.stat().st_size
        if file_size > self.max_file_size:
            raise ValueError(f"File too large: {file_size} bytes (max: {self.max_file_size})")
        
        # Read file asynchronously
        async with aiofiles.open(path, 'r', encoding='utf-8') as f:
            content = await f.read()
        
        logger.info(f"Read {file_size} bytes from {path}")
        return content
    
    async def _write_file_with_locking(self, path: Path, content: Optional[str], 
                                     agent_name: Optional[str], 
                                     base_delay: float = 1.0, max_delay: float = 30.0,
                                     use_locking: bool = True) -> str:
        """Write file with OS-level locking and smart retry delays."""
        if content is None:
            raise ValueError("Content is required for write operation")
        
        # If locking is disabled, use simple write
        if not use_locking:
            return await self._write_file_simple(path, content, agent_name)
        
        if agent_name is None:
            agent_name = "unknown"
        
        # Check content size
        content_size = len(content.encode('utf-8'))
        if content_size > self.max_file_size:
            raise ValueError(f"Content too large: {content_size} bytes (max: {self.max_file_size})")
        
        # Create parent directories if needed
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            async with aiofiles.open(path, 'w', encoding='utf-8') as f:
                fd = f.fileno()
                
                # Try to acquire exclusive lock (non-blocking, single attempt)
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                
                # Success! Register lock holder and reset retry count
                self._lock_registry.register_lock(str(path), agent_name)
                self._lock_registry.reset_retry_count(agent_name, str(path))
                
                try:
                    await f.write(content)
                    logger.info(f"Agent '{agent_name}' wrote {content_size} bytes to {path}")
                    return f"Successfully wrote {content_size} bytes to {path}"
                finally:
                    self._lock_registry.release_lock(str(path))
                    
        except BlockingIOError:
            # File is locked - calculate delay before responding
            delay = self._lock_registry.should_delay_response(
                agent_name, str(path), base_delay, max_delay
            )
            
            # Sleep before responding to slow down retry attempts
            await asyncio.sleep(delay)
            
            # Get lock holder info for error message
            lock_holder = self._lock_registry.get_lock_holder(str(path))
            retry_count = self._lock_registry.get_retry_count(agent_name, str(path))
            
            error_msg = self._format_lock_error(path, agent_name, lock_holder, retry_count, delay)
            raise ValueError(error_msg)
    
    def _format_lock_error(self, path: Path, agent_name: str, lock_holder: Optional[str], 
                          retry_count: int, suggested_wait: float) -> str:
        """Format comprehensive lock error message."""
        
        base_msg = f"File {path.name} is currently locked"
        
        if lock_holder:
            base_msg += f" by agent '{lock_holder}'"
        else:
            base_msg += f" by another process"
        
        base_msg += f". This is attempt #{retry_count}."
        
        suggestions = []
        
        if retry_count == 1:
            suggestions.append("You can retry this operation immediately")
        elif retry_count <= 3:
            suggestions.append(f"Consider waiting {suggested_wait:.0f} seconds before retrying")
        else:
            suggestions.append("Consider working on a different file for now")
            
        if lock_holder and lock_holder != agent_name:
            suggestions.append(f"or coordinate with agent '{lock_holder}'")
        
        return base_msg + " " + ", ".join(suggestions) + "."
    
    async def _list_directory(self, path: Path) -> List[Dict[str, Any]]:
        """List directory contents."""
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {path}")
        
        if not path.is_dir():
            raise ValueError(f"Not a directory: {path}")
        
        items = []
        for item in sorted(path.iterdir()):
            try:
                stat = item.stat()
                items.append({
                    "name": item.name,
                    "type": "directory" if item.is_dir() else "file",
                    "size": stat.st_size if item.is_file() else None,
                    "modified": stat.st_mtime
                })
            except (OSError, PermissionError):
                # Skip items we can't stat
                items.append({
                    "name": item.name,
                    "type": "unknown",
                    "error": "Permission denied"
                })
        
        logger.info(f"Listed {len(items)} items in {path}")
        return items
    
    async def _delete_file(self, path: Path) -> str:
        """Delete file or empty directory."""
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")
        
        if path.is_dir():
            # Only delete empty directories for safety
            try:
                path.rmdir()
                logger.info(f"Deleted empty directory: {path}")
                return f"Deleted empty directory: {path}"
            except OSError:
                raise ValueError(f"Directory not empty: {path}")
        else:
            path.unlink()
            logger.info(f"Deleted file: {path}")
            return f"Deleted file: {path}"
    
    async def _delete_file_with_locking(self, path: Path, agent_name: Optional[str],
                                      base_delay: float = 1.0, max_delay: float = 30.0) -> str:
        """Delete file with locking considerations."""
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")
        
        if agent_name is None:
            agent_name = "unknown"
        
        if path.is_dir():
            # Directories don't need file locking
            try:
                path.rmdir()
                logger.info(f"Agent '{agent_name}' deleted empty directory: {path}")
                return f"Deleted empty directory: {path}"
            except OSError:
                raise ValueError(f"Directory not empty: {path}")
        else:
            # Check if file is locked before attempting delete
            lock_holder = self._lock_registry.get_lock_holder(str(path))
            if lock_holder and lock_holder != agent_name:
                retry_count = self._lock_registry.get_retry_count(agent_name, str(path))
                delay = self._lock_registry.should_delay_response(
                    agent_name, str(path), base_delay, max_delay
                )
                await asyncio.sleep(delay)
                
                error_msg = self._format_lock_error(path, agent_name, lock_holder, retry_count, delay)
                raise ValueError(f"Cannot delete: {error_msg}")
            
            # Safe to delete
            path.unlink()
            self._lock_registry.release_lock(str(path))  # Clean up any lock registry entry
            logger.info(f"Agent '{agent_name}' deleted file: {path}")
            return f"Deleted file: {path}"
    
    async def _get_lock_status(self, path: Path, agent_name: Optional[str]) -> Dict[str, Any]:
        """Get detailed lock status for debugging."""
        if agent_name is None:
            agent_name = "unknown"
            
        lock_holder = self._lock_registry.get_lock_holder(str(path))
        retry_count = self._lock_registry.get_retry_count(agent_name, str(path))
        lock_duration = self._lock_registry.get_lock_duration(str(path))
        
        status = {
            "file": str(path),
            "locked": lock_holder is not None,
            "locked_by": lock_holder,
            "your_retry_count": retry_count,
            "lock_duration_seconds": lock_duration,
            "suggested_action": self._suggest_action(lock_holder, retry_count, lock_duration, agent_name)
        }
        
        logger.info(f"Lock status for {path}: {status}")
        return status
    
    def _suggest_action(self, lock_holder: Optional[str], retry_count: int, 
                       lock_duration: Optional[float], agent_name: str) -> str:
        """Suggest what agent should do based on lock status."""
        if not lock_holder:
            return "File is unlocked, safe to write"
        elif lock_holder == agent_name:
            return "You currently hold the lock on this file"
        elif lock_duration and lock_duration > 60:
            return "Lock has been held for over a minute, may be stale"
        elif retry_count > 5:
            return "Many retries attempted, consider working on different file"
        else:
            return f"File is locked by '{lock_holder}', wait or try different file"
    
    async def _write_file_simple(self, path: Path, content: str, agent_name: Optional[str]) -> str:
        """Write file without locking (fallback for when locking is disabled)."""
        if agent_name is None:
            agent_name = "unknown"

        # Check content size
        content_size = len(content.encode('utf-8'))
        if content_size > self.max_file_size:
            raise ValueError(f"Content too large: {content_size} bytes (max: {self.max_file_size})")

        # Create parent directories if needed
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write file asynchronously (no locking)
        async with aiofiles.open(path, 'w', encoding='utf-8') as f:
            await f.write(content)

        logger.info(f"Agent '{agent_name}' wrote {content_size} bytes to {path} (no locking)")
        return f"Successfully wrote {content_size} bytes to {path}"

    async def _append_file_with_locking(self, path: Path, content: Optional[str],
                                        agent_name: Optional[str],
                                        base_delay: float = 1.0, max_delay: float = 30.0,
                                        use_locking: bool = True) -> str:
        """Append to file with OS-level locking and smart retry delays."""
        if content is None:
            raise ValueError("Content is required for append operation")

        if agent_name is None:
            agent_name = "unknown"

        # Check content size
        content_size = len(content.encode('utf-8'))

        # For append, check combined size if file exists
        existing_size = path.stat().st_size if path.exists() else 0
        if existing_size + content_size > self.max_file_size:
            raise ValueError(f"Append would exceed max file size: {existing_size} + {content_size} > {self.max_file_size}")

        # Create parent directories if needed
        path.parent.mkdir(parents=True, exist_ok=True)

        # If locking is disabled, use simple append
        if not use_locking:
            async with aiofiles.open(path, 'a', encoding='utf-8') as f:
                await f.write(content)
            logger.info(f"Agent '{agent_name}' appended {content_size} bytes to {path} (no locking)")
            return f"Successfully appended {content_size} bytes to {path}"

        try:
            async with aiofiles.open(path, 'a', encoding='utf-8') as f:
                fd = f.fileno()

                # Try to acquire exclusive lock (non-blocking, single attempt)
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

                # Success! Register lock holder and reset retry count
                self._lock_registry.register_lock(str(path), agent_name)
                self._lock_registry.reset_retry_count(agent_name, str(path))

                try:
                    await f.write(content)
                    logger.info(f"Agent '{agent_name}' appended {content_size} bytes to {path}")
                    return f"Successfully appended {content_size} bytes to {path}"
                finally:
                    self._lock_registry.release_lock(str(path))

        except BlockingIOError:
            # File is locked - calculate delay before responding
            delay = self._lock_registry.should_delay_response(
                agent_name, str(path), base_delay, max_delay
            )

            # Sleep before responding to slow down retry attempts
            await asyncio.sleep(delay)

            # Get lock holder info for error message
            lock_holder = self._lock_registry.get_lock_holder(str(path))
            retry_count = self._lock_registry.get_retry_count(agent_name, str(path))

            error_msg = self._format_lock_error(path, agent_name, lock_holder, retry_count, delay)
            raise ValueError(error_msg)