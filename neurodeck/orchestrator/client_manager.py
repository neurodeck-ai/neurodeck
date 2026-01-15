"""
Client connection management and authentication for NeuroDeck orchestrator.

Handles client registration, authentication state, and connection lifecycle.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from dataclasses import dataclass

from ..common.protocol import MessageProtocol, MessageType, auth_result_message, error_message
from ..common.security import TokenManager
from ..common.logging import get_logger

logger = get_logger("orchestrator.client_manager")


@dataclass
class ClientConnection:
    """Represents a connected and authenticated client."""
    client_id: str
    client_type: str  # 'ui' or 'agent'
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    connected_at: datetime
    authenticated_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.authenticated_at is None:
            self.authenticated_at = datetime.now()
    
    async def send_message(self, message: dict) -> bool:
        """
        Send a message to this client.
        
        Returns:
            True if message sent successfully, False if connection failed
        """
        try:
            encoded = MessageProtocol.encode_message(message)
            self.writer.write(encoded)
            await self.writer.drain()
            return True
        except Exception as e:
            logger.error(f"Failed to send message to {self.client_id}: {e}")
            return False
    
    async def close(self) -> None:
        """Close the client connection gracefully."""
        try:
            self.writer.close()
            await self.writer.wait_closed()
        except Exception as e:
            logger.error(f"Error closing connection for {self.client_id}: {e}")
    
    def get_address(self) -> str:
        """Get the client's network address."""
        try:
            return str(self.writer.get_extra_info('peername'))
        except Exception:
            return "unknown"


class ClientManager:
    """
    Manages client connections, authentication, and message routing.
    
    Responsibilities:
    - Track authenticated client connections
    - Validate authentication tokens
    - Manage client types (UI vs agent)
    - Handle connection cleanup
    """
    
    def __init__(self, auth_token: str):
        self.auth_token = auth_token
        self.clients: Dict[str, ClientConnection] = {}
        self.pending_auth: Dict[asyncio.StreamWriter, datetime] = {}
        
    async def authenticate_client(
        self, 
        message: dict, 
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter
    ) -> Optional[str]:
        """
        Authenticate a client connection.
        
        Args:
            message: Authentication message from client
            reader: Client's stream reader
            writer: Client's stream writer
            
        Returns:
            client_id if authentication successful, None otherwise
        """
        # Validate message format
        if message.get("type") != MessageType.AUTH.value:
            error_msg = error_message("Authentication required")
            await self._send_auth_response(writer, error_msg)
            return None
        
        token = message.get("token")
        client_id = message.get("client_id")
        
        if not token or not client_id:
            auth_result = auth_result_message(False, "Missing token or client_id")
            await self._send_auth_response(writer, auth_result)
            return None
        
        # Validate token
        if not TokenManager.validate_token(token, self.auth_token):
            auth_result = auth_result_message(False, "Invalid authentication token")
            await self._send_auth_response(writer, auth_result)
            logger.warning(f"Authentication failed for client_id: {client_id}")
            return None
        
        # Determine client type from client_id
        client_type = self._determine_client_type(client_id)
        
        # Check for existing connection with same client_id
        if client_id in self.clients:
            logger.info(f"Replacing existing connection for client {client_id}")
            await self.disconnect_client(client_id, "Duplicate connection")
        
        # Create client connection
        connection = ClientConnection(
            client_id=client_id,
            client_type=client_type,
            reader=reader,
            writer=writer,
            connected_at=datetime.now()
        )
        
        # Register client
        self.clients[client_id] = connection
        
        # Remove from pending auth if present
        self.pending_auth.pop(writer, None)
        
        # Send success response
        auth_result = auth_result_message(True, f"Authenticated as {client_type}")
        await self._send_auth_response(writer, auth_result)
        
        logger.info(f"Client authenticated: {client_id} ({client_type}) from {connection.get_address()}")
        return client_id
    
    def _determine_client_type(self, client_id: str) -> str:
        """Determine client type from client_id pattern."""
        if client_id.startswith("ui-") or client_id == "console":
            return "ui"
        elif client_id.startswith("agent-"):
            return "agent"
        else:
            return "unknown"
    
    async def _send_auth_response(self, writer: asyncio.StreamWriter, message: dict) -> None:
        """Send authentication response message."""
        try:
            encoded = MessageProtocol.encode_message(message)
            writer.write(encoded)
            await writer.drain()
        except Exception as e:
            logger.error(f"Failed to send auth response: {e}")
    
    def add_pending_auth(self, writer: asyncio.StreamWriter) -> None:
        """Add a connection to pending authentication list."""
        self.pending_auth[writer] = datetime.now()
    
    def remove_pending_auth(self, writer: asyncio.StreamWriter) -> None:
        """Remove a connection from pending authentication list."""
        self.pending_auth.pop(writer, None)
    
    async def disconnect_client(self, client_id: str, reason: str) -> None:
        """
        Disconnect a client and clean up resources.
        
        Args:
            client_id: ID of client to disconnect
            reason: Reason for disconnection (for logging)
        """
        if client_id not in self.clients:
            logger.warning(f"Attempted to disconnect unknown client: {client_id}")
            return
        
        connection = self.clients[client_id]
        logger.info(f"Disconnecting client {client_id} ({connection.client_type}): {reason}")
        
        # Remove from registry
        del self.clients[client_id]
        
        # Close connection
        await connection.close()
    
    def get_client(self, client_id: str) -> Optional[ClientConnection]:
        """Get a client connection by ID."""
        return self.clients.get(client_id)
    
    def get_all_clients(self) -> List[ClientConnection]:
        """Get all authenticated client connections."""
        return list(self.clients.values())
    
    def get_clients_by_type(self, client_type: str) -> List[ClientConnection]:
        """Get all clients of a specific type."""
        return [conn for conn in self.clients.values() if conn.client_type == client_type]
    
    def get_ui_clients(self) -> List[ClientConnection]:
        """Get all UI client connections."""
        return self.get_clients_by_type("ui")
    
    def get_agent_clients(self) -> List[ClientConnection]:
        """Get all agent client connections."""
        return self.get_clients_by_type("agent")
    
    async def broadcast_to_all(self, message: dict) -> int:
        """
        Broadcast message to all authenticated clients.
        
        Returns:
            Number of clients that successfully received the message
        """
        successful_sends = 0
        failed_clients = []
        
        for client_id, connection in list(self.clients.items()):
            if await connection.send_message(message):
                successful_sends += 1
            else:
                failed_clients.append(client_id)
        
        # Clean up failed connections
        for client_id in failed_clients:
            await self.disconnect_client(client_id, "Send failed")
        
        logger.debug(f"Broadcast to {successful_sends}/{len(self.clients)} clients")
        return successful_sends
    
    async def send_to_ui_clients(self, message: dict) -> int:
        """
        Send message to all UI clients only.
        
        Returns:
            Number of UI clients that successfully received the message
        """
        successful_sends = 0
        failed_clients = []
        
        ui_clients = self.get_ui_clients()
        for connection in ui_clients:
            if await connection.send_message(message):
                successful_sends += 1
            else:
                failed_clients.append(connection.client_id)
        
        # Clean up failed connections
        for client_id in failed_clients:
            await self.disconnect_client(client_id, "Send failed")
        
        logger.debug(f"Sent to {successful_sends}/{len(ui_clients)} UI clients")
        return successful_sends
    
    async def send_to_client(self, client_id: str, message: dict) -> bool:
        """
        Send message to a specific client.
        
        Returns:
            True if message sent successfully, False if failed
        """
        connection = self.get_client(client_id)
        if not connection:
            logger.warning(f"Attempted to send to unknown client: {client_id}")
            return False
        
        success = await connection.send_message(message)
        if not success:
            await self.disconnect_client(client_id, "Send failed")
        
        return success
    
    def get_connection_stats(self) -> dict:
        """Get statistics about current connections."""
        ui_count = len(self.get_ui_clients())
        agent_count = len(self.get_agent_clients())
        unknown_count = len(self.get_clients_by_type("unknown"))
        pending_count = len(self.pending_auth)
        
        return {
            "total_clients": len(self.clients),
            "ui_clients": ui_count,
            "agent_clients": agent_count,
            "unknown_clients": unknown_count,
            "pending_auth": pending_count,
            "client_ids": list(self.clients.keys())
        }
    
    async def cleanup_stale_connections(self, max_age_minutes: int = 60) -> int:
        """
        Clean up stale connections that have been inactive.
        
        Args:
            max_age_minutes: Maximum age of connections to keep
            
        Returns:
            Number of connections cleaned up
        """
        cutoff_time = datetime.now() - timedelta(minutes=max_age_minutes)
        stale_clients = []
        
        for client_id, connection in self.clients.items():
            if connection.connected_at < cutoff_time:
                stale_clients.append(client_id)
        
        for client_id in stale_clients:
            await self.disconnect_client(client_id, "Stale connection cleanup")
        
        if stale_clients:
            logger.info(f"Cleaned up {len(stale_clients)} stale connections")
        
        return len(stale_clients)
    
    async def shutdown(self) -> None:
        """Shutdown all client connections gracefully."""
        logger.info(f"Shutting down {len(self.clients)} client connections")
        
        # Disconnect all clients
        for client_id in list(self.clients.keys()):
            await self.disconnect_client(client_id, "Server shutdown")
        
        # Clear pending auth
        self.pending_auth.clear()
        
        logger.info("All client connections closed")