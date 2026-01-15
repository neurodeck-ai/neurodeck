"""
NeuroDeck console client - handles connection to orchestrator.
"""

import asyncio
import json
import ssl
from pathlib import Path
from typing import Optional, Dict, Any, Callable
import os

from ..common.logging import get_logger
from ..common.protocol import MessageProtocol, auth_message, chat_message

logger = get_logger(__name__)


class NeuroDeckClient:
    """Client for connecting to NeuroDeck orchestrator."""
    
    def __init__(self, host: str = "localhost", port: int = 9999, log_callback: Optional[Callable[[str, str, str], None]] = None):
        self.host = host
        self.port = port
        self.reader: Optional[asyncio.StreamReader] = None
        self.writer: Optional[asyncio.StreamWriter] = None
        self.connected = False
        self.authenticated = False
        self.message_handler: Optional[Callable] = None
        self.log_callback = log_callback
    
    def _log(self, sender: str, message: str, msg_type: str = "system-info"):
        """
        Log a message either to TUI debug panel or fallback to logger.
        
        Args:
            sender: Source of the log message
            message: Log message content  
            msg_type: Message type for styling (system-info, system-error, system-warning, system-success)
        """
        if self.log_callback:
            try:
                self.log_callback(sender, message, msg_type)
            except Exception:
                # If callback fails, fallback to logger to avoid losing critical info
                logger.warning(f"Client log callback failed: {sender}: {message}")
        # For critical errors, always log to file regardless of callback
        elif msg_type == "system-error":
            logger.error(f"{sender}: {message}")
        
    async def connect(self) -> bool:
        """Connect to orchestrator with TLS."""
        try:
            self._log("Client", f"Connecting to orchestrator at {self.host}:{self.port}", "system-info")
            
            # Create SSL context (allow self-signed certs for development)
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            self.reader, self.writer = await asyncio.open_connection(
                self.host, self.port, ssl=ssl_context,
                limit=10 * 1024 * 1024  # 10MB buffer for large tool results
            )
            
            self.connected = True
            self._log("Client", "Connected to orchestrator", "system-success")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False
    
    async def authenticate(self) -> bool:
        """Authenticate with orchestrator."""
        if not self.connected:
            return False
            
        try:
            # Get token from environment
            token = os.getenv("NEURODECK_TOKEN")
            if not token:
                logger.error("NEURODECK_TOKEN not set in environment")
                return False
            
            # Send authentication
            auth_msg = auth_message(token, "ui-neurodeck-console")
            await self.send_message(auth_msg)
            
            # Wait for auth response
            response = await self.receive_message()
            if response and response.get("type") == "auth_result" and response.get("success"):
                self.authenticated = True
                self._log("Client", "Authenticated successfully", "system-success")
                return True
            else:
                logger.error(f"Authentication failed: {response}")
                return False
                
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return False
    
    async def send_message(self, message: Dict[str, Any]):
        """Send message to orchestrator."""
        if not self.writer:
            raise RuntimeError("Not connected")
            
        try:
            encoded = MessageProtocol.encode_message(message)
            self.writer.write(encoded)
            await self.writer.drain()
            
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            raise
    
    async def receive_message(self) -> Optional[Dict[str, Any]]:
        """Receive message from orchestrator."""
        if not self.reader:
            return None
            
        try:
            line = await self.reader.readline()
            if not line:
                return None
                
            return MessageProtocol.decode_message(line)
            
        except Exception as e:
            logger.error(f"Failed to receive message: {e}")
            return None
    
    async def send_chat(self, content: str):
        """Send chat message."""
        if not self.authenticated:
            raise RuntimeError("Not authenticated")
            
        chat_msg = chat_message("user", content)
        await self.send_message(chat_msg)
    
    def set_message_handler(self, handler: Callable):
        """Set handler for incoming messages."""
        self.message_handler = handler
    
    async def message_loop(self):
        """Main message receiving loop."""
        while self.connected:
            try:
                message = await self.receive_message()
                if message is None:
                    # Connection closed
                    break
                    
                if self.message_handler:
                    await self.message_handler(message)
                    
            except Exception as e:
                logger.error(f"Error in message loop: {e}")
                break
        
        self.connected = False
        self._log("Client", "Message loop ended", "system-info")
    
    async def disconnect(self):
        """Disconnect from orchestrator."""
        if self.writer:
            self.writer.close()
            await self.writer.wait_closed()
        
        self.connected = False
        self.authenticated = False
        self._log("Client", "Disconnected from orchestrator", "system-info")