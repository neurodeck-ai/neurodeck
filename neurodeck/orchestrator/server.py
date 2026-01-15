"""
TCP socket server with TLS for NeuroDeck orchestrator daemon.

Handles concurrent connections from console UI and agent processes.
Provides authentication and message routing in a broadcast model.
"""

import asyncio
import ssl
import json
from datetime import datetime
from typing import Dict, Set, Optional, Tuple, Any
from pathlib import Path

from ..common.protocol import (
    MessageProtocol, MessageType, ApprovalLevel, LogLevel,
    auth_message, auth_result_message, error_message, system_log_message,
    chat_history_response_message, tool_result_message, 
    tool_approval_request_message, clear_context_message, yolo_status_message
)
from ..common.security import CertificateManager, TokenManager
from ..common.logging import get_logger
from ..common.config import OrchestratorConfig, ToolConfig, FilesystemToolConfig, ConfigManager
from ..common.database import ContextDatabase
from .tool_executor import ToolExecutor

logger = get_logger("orchestrator.server")


class ClientConnection:
    """Represents a connected client (UI or agent process)."""
    
    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        client_id: str,
        address: Tuple[str, int]
    ):
        self.reader = reader
        self.writer = writer
        self.client_id = client_id
        self.address = address
        self.authenticated = False
        self.client_type = None  # "ui" or "agent"
        self.agent_name = None   # For agent connections
        
    async def send_message(self, message_dict: Dict) -> bool:
        """
        Send message to client.
        
        Returns:
            True if successful, False if connection failed
        """
        try:
            encoded = MessageProtocol.encode_message(message_dict)
            self.writer.write(encoded)
            await self.writer.drain()
            return True
        except Exception as e:
            logger.error(f"Failed to send message to {self.client_id}: {e}")
            return False
    
    async def receive_message(self) -> Optional[Dict]:
        """
        Receive message from client.
        
        Returns:
            Message dict or None if connection closed/failed
        """
        try:
            line = await self.reader.readline()
            if not line:
                return None
            
            return MessageProtocol.decode_message(line)
        except Exception as e:
            logger.error(f"Failed to receive message from {self.client_id}: {e}")
            return None
    
    def close(self):
        """Close the connection."""
        try:
            self.writer.close()
        except Exception as e:
            logger.error(f"Error closing connection {self.client_id}: {e}")


class OrchestratorServer:
    """
    Main orchestrator server that handles TLS connections and message routing.
    """
    
    def __init__(self, config: OrchestratorConfig, tool_configs: Dict[str, ToolConfig] = None, 
                 agent_tool_overrides: Dict[str, Dict[str, ToolConfig]] = None,
                 mcp_configs: Dict[str, Any] = None):
        self.config = config
        self.tool_configs = tool_configs or {}
        self.agent_tool_overrides = agent_tool_overrides or {}
        self.mcp_configs = mcp_configs or {}
        self.clients: Dict[str, ClientConnection] = {}
        self.running = False
        self.server = None
        self.process_manager = None  # Will be set by orchestrator daemon
        
        # Initialize database for chat history
        self.context_db = ContextDatabase()
        
        # Initialize tool executor with configurations
        self.tool_executor = ToolExecutor(
            config.__dict__, 
            self.tool_configs, 
            self.agent_tool_overrides,
            self.mcp_configs
        )
        self.pending_tool_approvals: Dict[str, Dict[str, Any]] = {}
        
        # YOLO mode - auto-approve all tool requests (always starts disabled)
        self.yolo_mode = False
        
        # Get default approval timeout from filesystem config or use 30s
        filesystem_config = self.tool_configs.get("filesystem")
        self.approval_timeout = filesystem_config.approval_timeout if filesystem_config else 30.0
        
        # Set orchestrator server reference for chat_info tool
        self.tool_executor.set_orchestrator_server(self)
        
        # Initialize TLS
        self.cert_manager = CertificateManager()
        self.cert_manager.ensure_certificates_exist(config.host)
        
        logger.info(f"Orchestrator server initialized - {config.host}:{config.port}")
    
    def set_process_manager(self, process_manager):
        """Set the process manager for agent lifecycle management."""
        self.process_manager = process_manager
    
    async def authenticate_client(self, connection: ClientConnection) -> bool:
        """
        Authenticate a client connection.
        
        Returns:
            True if authentication successful
        """
        logger.info(f"Authenticating client {connection.client_id} from {connection.address}")
        
        try:
            # Wait for auth message with timeout
            auth_msg = await asyncio.wait_for(
                connection.receive_message(),
                timeout=30.0  # 30 second timeout for authentication
            )
            if not auth_msg or auth_msg.get("type") != MessageType.AUTH.value:
                await connection.send_message(
                    auth_result_message(False, "Authentication required")
                )
                return False
            
            # Validate token
            provided_token = auth_msg.get("token", "")
            if not TokenManager.validate_token(provided_token, self.config.auth_token):
                await connection.send_message(
                    auth_result_message(False, "Invalid authentication token")
                )
                logger.warning(f"Invalid token from {connection.address}")
                return False
            
            # Check client type
            client_id = auth_msg.get("client_id", "")
            if client_id.startswith("ui-"):
                connection.client_type = "ui"
            elif client_id.startswith("agent-"):
                connection.client_type = "agent"
                connection.agent_name = client_id.split("-", 1)[1]
            else:
                await connection.send_message(
                    auth_result_message(False, "Invalid client ID format")
                )
                return False
            
            connection.authenticated = True
            await connection.send_message(
                auth_result_message(True, f"Authenticated as {connection.client_type}")
            )
            
            logger.info(f"Client {client_id} authenticated as {connection.client_type}")
            return True
            
        except asyncio.TimeoutError:
            await connection.send_message(
                auth_result_message(False, "Authentication timeout")
            )
            logger.warning(f"Authentication timeout for {connection.address}")
            return False
        except Exception as e:
            await connection.send_message(
                auth_result_message(False, f"Authentication error: {str(e)}")
            )
            logger.error(f"Authentication error for {connection.address}: {e}")
            return False
    
    async def broadcast_message(
        self,
        message_dict: Dict,
        exclude_client: Optional[str] = None,
        target_type: Optional[str] = None
    ):
        """
        Broadcast message to all connected clients.
        
        Args:
            message_dict: Message to broadcast
            exclude_client: Client ID to exclude from broadcast
            target_type: Only send to specific client type ("ui" or "agent")
        """
        message_type = message_dict.get("type", "unknown")
        logger.debug(f"📡 BROADCASTING {message_type} message")
        logger.debug(f"   └─ Exclude: {exclude_client}, Target: {target_type}")
        
        # Count eligible clients
        eligible_clients = []
        skipped_clients = []
        disconnected_clients = []
        
        for client_id, connection in list(self.clients.items()):
            # Skip excluded client (use list() to avoid dict size change during iteration)
            if exclude_client and client_id == exclude_client:
                skipped_clients.append(f"{client_id} (excluded)")
                continue
            
            # Skip if target type specified and doesn't match
            if target_type and connection.client_type != target_type:
                skipped_clients.append(f"{client_id} (wrong type: {connection.client_type})")
                continue
            
            # Skip unauthenticated clients
            if not connection.authenticated:
                skipped_clients.append(f"{client_id} (not authenticated)")
                continue
            
            eligible_clients.append(f"{client_id} ({connection.client_type})")
            
            # Send message
            logger.debug(f"   └─ 📤 Sending to {client_id} ({connection.client_type})")
            success = await connection.send_message(message_dict)
            if not success:
                disconnected_clients.append(client_id)
                logger.debug(f"   └─ ❌ Failed to send to {client_id}")
            else:
                logger.debug(f"   └─ ✅ Sent successfully to {client_id}")
        
        logger.debug(f"📡 BROADCAST SUMMARY:")
        logger.debug(f"   └─ Eligible: {len(eligible_clients)} - {eligible_clients}")
        logger.debug(f"   └─ Skipped: {len(skipped_clients)} - {skipped_clients}")
        logger.debug(f"   └─ Failed: {len(disconnected_clients)} - {disconnected_clients}")
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            await self.remove_client(client_id)
    
    async def broadcast_system_log(self, log_message: Dict[str, Any]):
        """
        Broadcast system log message to UI clients only.
        
        This method is used by the process manager to send system logs.
        """
        await self.broadcast_message(log_message, target_type="ui")
    
    async def handle_client_message(
        self,
        connection: ClientConnection,
        message_dict: Dict
    ):
        """Handle incoming message from client."""
        message_type = message_dict.get("type")
        
        if message_type == MessageType.CHAT.value:
            sender = message_dict.get("sender", "unknown")
            content = message_dict.get("content", "")
            logger.debug(f"CHAT MESSAGE received from {connection.client_id} ({connection.client_type})")
            logger.debug(f"   └─ Sender: {sender}, Content: '{content[:50]}{'...' if len(content) > 50 else ''}'")
            
            # Debug: Check if this might be a misrouted system log
            if "waiting for message" in content.lower():
                logger.warning(f"SERVER_DEBUG: Suspicious CHAT message that looks like system log: '{content}'")
            
            # Skip broadcasting if agent responds with NO_RESPONSE
            if content.strip() == "NO_RESPONSE":
                logger.debug(f"   └─ Skipping broadcast: {sender} sent NO_RESPONSE")
                return
            
            logger.debug(f"   └─ Will broadcast to {len(self.clients)} total clients, excluding {connection.client_id}")
            
            # Broadcast chat message to all clients
            await self.broadcast_message(
                message_dict,
                exclude_client=connection.client_id
            )
            
        elif message_type == MessageType.COMMAND.value:
            # Handle system commands (only from UI clients)
            if connection.client_type == "ui":
                await self.handle_command(connection, message_dict)
            else:
                await connection.send_message(
                    error_message("Commands only allowed from UI clients")
                )
                
        elif message_type == MessageType.SYSTEM_LOG.value:
            # Forward system logs to UI clients only
            log_message = message_dict.get("message", "")
            sender = message_dict.get("sender", "unknown")
            logger.debug(f"SERVER_DEBUG: Routing SYSTEM_LOG from {sender}: '{log_message[:50]}...'")
            await self.broadcast_message(
                message_dict,
                exclude_client=connection.client_id,
                target_type="ui"
            )
        
        elif message_type == MessageType.CHAT_HISTORY_REQUEST.value:
            # Handle chat history request (only from UI clients)
            if connection.client_type == "ui":
                await self.handle_chat_history_request(connection, message_dict)
            else:
                await connection.send_message(
                    error_message("Chat history requests only allowed from UI clients")
                )
        
        elif message_type == MessageType.TOOL_REQUEST.value:
            # Handle tool execution request (only from agent clients)
            if connection.client_type == "agent":
                await self.handle_tool_request(connection, message_dict)
            else:
                await connection.send_message(
                    error_message("Tool requests only allowed from agent clients")
                )
        
        elif message_type == MessageType.TOOL_APPROVAL.value:
            # Handle tool approval response (only from UI clients)
            if connection.client_type == "ui":
                await self.handle_tool_approval(connection, message_dict)
            else:
                await connection.send_message(
                    error_message("Tool approvals only allowed from UI clients")
                )
        
        else:
            logger.warning(f"Unknown message type: {message_type}")
    
    async def handle_command(self, connection: ClientConnection, message_dict: Dict):
        """Handle system commands from UI clients."""
        cmd = message_dict.get("cmd")
        
        if cmd == "list_agents":
            # Return list of configured agents with their status
            if self.process_manager:
                agent_status = self.process_manager.get_agent_status()
                agent_list = [
                    {
                        "name": name,
                        "status": status["status"],
                        "connected": status["connected"],
                        "pid": status["pid"],
                        "restart_count": status["restart_count"]
                    }
                    for name, status in agent_status.items()
                ]
            else:
                # Fallback to connected agents only
                agent_list = [
                    {"name": conn.agent_name, "client_id": client_id, "status": "connected"}
                    for client_id, conn in list(self.clients.items())
                    if conn.client_type == "agent" and conn.authenticated
                ]
            
            response = {
                "type": MessageType.COMMAND_RESULT.value,
                "cmd": cmd,
                "success": True,
                "data": agent_list
            }
            await connection.send_message(response)
            
        elif cmd == "agent_status":
            # Return detailed status of all agents
            if self.process_manager:
                status = self.process_manager.get_agent_status()
                response = {
                    "type": MessageType.COMMAND_RESULT.value,
                    "cmd": cmd,
                    "success": True,
                    "data": status
                }
            else:
                response = {
                    "type": MessageType.COMMAND_RESULT.value,
                    "cmd": cmd,
                    "success": False,
                    "error": "Process manager not available"
                }
            await connection.send_message(response)
            
        elif cmd == "restart_agent":
            # Restart a specific agent
            agent_name = message_dict.get("agent")
            if not agent_name:
                response = {
                    "type": MessageType.COMMAND_RESULT.value,
                    "cmd": cmd,
                    "success": False,
                    "error": "Missing agent name"
                }
            elif self.process_manager:
                try:
                    await self.process_manager.restart_agent(agent_name)
                    response = {
                        "type": MessageType.COMMAND_RESULT.value,
                        "cmd": cmd,
                        "success": True,
                        "data": f"Agent {agent_name} restart initiated"
                    }
                except Exception as e:
                    response = {
                        "type": MessageType.COMMAND_RESULT.value,
                        "cmd": cmd,
                        "success": False,
                        "error": str(e)
                    }
            else:
                response = {
                    "type": MessageType.COMMAND_RESULT.value,
                    "cmd": cmd,
                    "success": False,
                    "error": "Process manager not available"
                }
            await connection.send_message(response)
            
        elif cmd == "stop_agent":
            # Stop a specific agent
            agent_name = message_dict.get("agent")
            if not agent_name:
                response = {
                    "type": MessageType.COMMAND_RESULT.value,
                    "cmd": cmd,
                    "success": False,
                    "error": "Missing agent name"
                }
            elif self.process_manager:
                try:
                    await self.process_manager.stop_agent(agent_name)
                    response = {
                        "type": MessageType.COMMAND_RESULT.value,
                        "cmd": cmd,
                        "success": True,
                        "data": f"Agent {agent_name} stopped"
                    }
                except Exception as e:
                    response = {
                        "type": MessageType.COMMAND_RESULT.value,
                        "cmd": cmd,
                        "success": False,
                        "error": str(e)
                    }
            else:
                response = {
                    "type": MessageType.COMMAND_RESULT.value,
                    "cmd": cmd,
                    "success": False,
                    "error": "Process manager not available"
                }
            await connection.send_message(response)
            
        elif cmd == "ping":
            # Simple ping response
            response = {
                "type": MessageType.COMMAND_RESULT.value,
                "cmd": cmd,
                "success": True,
                "data": "pong"
            }
            await connection.send_message(response)
            
        elif cmd == "wipe":
            # Handle database context wipe command
            await self.handle_wipe_command(connection, message_dict)
            
        elif cmd == "yolo":
            # Handle YOLO mode toggle
            await self.handle_yolo_command(connection, message_dict)
            
        else:
            response = {
                "type": MessageType.COMMAND_RESULT.value,
                "cmd": cmd,
                "success": False,
                "error": f"Unknown command: {cmd}"
            }
            await connection.send_message(response)
    
    async def handle_chat_history_request(self, connection: ClientConnection, message_dict: Dict):
        """Handle chat history request from UI client."""
        try:
            limit = message_dict.get("limit", 20)
            logger.debug(f"Chat history request from {connection.client_id} (limit: {limit})")
            
            # Load recent chat history from database across all sessions
            messages = self.context_db.load_recent_chat_history_all_sessions(limit=limit)
            
            # Send response back to requesting client
            response = chat_history_response_message(messages)
            await connection.send_message(response)
            
            logger.debug(f"Sent {len(messages)} historical messages to {connection.client_id}")
            
        except Exception as e:
            logger.error(f"Error handling chat history request: {e}")
            await connection.send_message(
                error_message(f"Failed to load chat history: {str(e)}")
            )
    
    async def handle_wipe_command(self, connection: ClientConnection, message_dict: Dict):
        """Handle database wipe command by delegating to CLEAR_CONTEXT system."""
        try:
            agent_name = message_dict.get("agent")
            cmd = message_dict.get("cmd", "wipe")
            
            if agent_name:
                # Validate agent exists (using configured agents, not just database agents)
                config_manager = ConfigManager("config/agents.ini")
                available_agents = list(config_manager.load_agent_configs().keys())
                if agent_name not in available_agents:
                    error_msg = f"Agent '{agent_name}' not found. Available: {', '.join(available_agents)}"
                    response = {
                        "type": MessageType.COMMAND_RESULT.value,
                        "cmd": cmd,
                        "success": False,
                        "error": error_msg
                    }
                    await connection.send_message(response)
                    return
                
                # Delegate to CLEAR_CONTEXT system for specific agent
                clear_msg = clear_context_message(agent_name)
                await self.broadcast_message(clear_msg, target_type="agent")
                success_msg = f"Wipe initiated for '{agent_name}'"
                logger.debug(f"Sent CLEAR_CONTEXT message to agent '{agent_name}'")
                
            else:
                # Delegate to CLEAR_CONTEXT system for all agents
                clear_msg = clear_context_message("all")
                await self.broadcast_message(clear_msg, target_type="agent")
                success_msg = "Wipe initiated for ALL agents"
                logger.debug("Sent CLEAR_CONTEXT message to all agents")
            
            # Send success response (agents will handle actual clearing and logging)
            response = {
                "type": MessageType.COMMAND_RESULT.value,
                "cmd": cmd,
                "success": True,
                "message": success_msg
            }
            await connection.send_message(response)
            
        except Exception as e:
            logger.error(f"Wipe command failed: {e}")
            error_response = {
                "type": MessageType.COMMAND_RESULT.value,
                "cmd": message_dict.get("cmd", "wipe"),
                "success": False,
                "error": f"Wipe failed: {str(e)}"
            }
            await connection.send_message(error_response)
    
    async def handle_yolo_command(self, connection: ClientConnection, message_dict: Dict):
        """Handle YOLO mode toggle command with confirmation."""
        cmd = message_dict.get("cmd", "yolo")
        confirm = message_dict.get("confirm", False)
        
        try:
            if not self.yolo_mode and not confirm:
                # Enabling YOLO mode - require explicit confirmation
                response = {
                    "type": MessageType.COMMAND_RESULT.value,
                    "cmd": cmd,
                    "success": False,
                    "data": "⚠️ YOLO mode will auto-approve ALL tool requests without human oversight. Type '/yolo confirm' to proceed.",
                    "requires_confirmation": True
                }
                await connection.send_message(response)
                return
            
            # Either disabling or confirmed enabling
            self.yolo_mode = not self.yolo_mode
            status = "enabled" if self.yolo_mode else "disabled"
            
            # Notify UI clients only
            yolo_msg = yolo_status_message(self.yolo_mode, f"YOLO mode {status}")
            await self.broadcast_message(yolo_msg, target_type="ui")
            
            # Response to command
            response = {
                "type": MessageType.COMMAND_RESULT.value,
                "cmd": cmd,
                "success": True,
                "data": f"🔥 YOLO mode {status}" if self.yolo_mode else f"YOLO mode {status}"
            }
            await connection.send_message(response)
            
            logger.warning(f"YOLO mode {status} by {connection.client_id}")
            
        except Exception as e:
            logger.error(f"YOLO command failed: {e}")
            error_response = {
                "type": MessageType.COMMAND_RESULT.value,
                "cmd": cmd,
                "success": False,
                "error": f"YOLO command failed: {str(e)}"
            }
            await connection.send_message(error_response)
    
    async def handle_tool_request(self, connection: ClientConnection, message_dict: Dict):
        """Handle tool execution request from agent client."""
        tool = message_dict.get("tool")
        command = message_dict.get("command")
        args = message_dict.get("args", {})
        request_id = message_dict.get("request_id")
        agent_name = connection.agent_name
        
        logger.info(f"Tool request from {agent_name}: {tool}.{command}")
        
        # Send detailed tool request info to UI debug panel
        await self._log_tool_request(agent_name, tool, command, args, request_id)
        
        try:
            # Check if tool requires approval
            if self._requires_approval(tool, command, args, connection.agent_name):
                # Send approval request to UI clients
                await self._request_approval(
                    agent_name, tool, command, args, request_id
                )
                
                # Store pending request
                self.pending_tool_approvals[request_id] = {
                    "agent_id": connection.client_id,
                    "tool": tool,
                    "command": command,
                    "args": args,
                    "timestamp": datetime.now()
                }
                
                # Log approval requirement to debug panel
                await self._log_tool_approval_required(agent_name, tool, command, request_id)
                
                # Start timeout timer
                asyncio.create_task(
                    self._approval_timeout(request_id)
                )
            else:
                # Auto-approve safe operations
                await self._log_tool_auto_approved(agent_name, tool, command, request_id)
                await self._execute_approved_tool(
                    connection.client_id, agent_name, tool, command, args, request_id
                )
                
        except Exception as e:
            logger.error(f"Error handling tool request: {e}")
            # Send error back to agent
            error_msg = tool_result_message(
                request_id=request_id,
                success=False,
                error=str(e)
            )
            await connection.send_message(error_msg)
    
    async def handle_tool_approval(self, connection: ClientConnection, message_dict: Dict):
        """Handle tool approval response from UI client."""
        agent = message_dict.get("agent")
        approval = message_dict.get("approval")
        request_id = message_dict.get("request_id")
        
        logger.info(f"Tool approval from {connection.client_id}: {approval} for request {request_id}")
        
        if request_id not in self.pending_tool_approvals:
            logger.warning(f"No pending approval for request {request_id}")
            await connection.send_message(
                error_message(f"No pending approval found for request {request_id}")
            )
            return
        
        pending = self.pending_tool_approvals.pop(request_id)
        
        try:
            if approval == ApprovalLevel.DENY.value:
                # Log denial to debug panel
                await self._log_tool_denied(pending.get("agent_id", "unknown"), pending["tool"], pending["command"], request_id)
                
                # Send denial to agent
                error_msg = tool_result_message(
                    request_id=request_id,
                    success=False,
                    error="Tool execution denied by user"
                )
                # Find agent connection
                agent_connection = None
                for client_id, conn in list(self.clients.items()):
                    if client_id == pending["agent_id"]:
                        agent_connection = conn
                        break
                
                if agent_connection:
                    await agent_connection.send_message(error_msg)
            else:
                # Log approval to debug panel
                await self._log_tool_approved(pending.get("agent_id", "unknown"), pending["tool"], pending["command"], request_id)
                
                # Extract agent name from agent_id and execute approved tool
                agent_name_extracted = self._extract_agent_name(pending["agent_id"])
                await self._execute_approved_tool(
                    pending["agent_id"],
                    agent_name_extracted,
                    pending["tool"],
                    pending["command"],
                    pending["args"],
                    request_id
                )
                
        except Exception as e:
            logger.error(f"Error processing tool approval: {e}")
            await connection.send_message(
                error_message(f"Failed to process approval: {str(e)}")
            )
    
    async def handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter
    ):
        """Handle a new client connection."""
        address = writer.get_extra_info('peername')
        client_id = f"client-{address[0]}-{address[1]}"
        
        connection = ClientConnection(reader, writer, client_id, address)
        
        try:
            # Authenticate client
            if not await self.authenticate_client(connection):
                return
            
            # Add to client list
            self.clients[client_id] = connection
            logger.info(f"Client {client_id} connected ({connection.client_type})")
            
            # Notify process manager of agent connection
            if connection.client_type == "agent" and self.process_manager:
                self.process_manager.mark_agent_connected(connection.agent_name)
            
            # Handle messages
            while self.running:
                message_dict = await connection.receive_message()
                if message_dict is None:
                    break  # Connection closed
                
                await self.handle_client_message(connection, message_dict)
                
        except Exception as e:
            logger.error(f"Error handling client {client_id}: {e}")
        finally:
            await self.remove_client(client_id)
    
    async def remove_client(self, client_id: str):
        """Remove client and close connection."""
        if client_id in self.clients:
            connection = self.clients[client_id]
            
            # Notify process manager of agent disconnection
            if connection.client_type == "agent" and self.process_manager:
                self.process_manager.mark_agent_disconnected(connection.agent_name)
            
            connection.close()
            del self.clients[client_id]
            logger.info(f"Client {client_id} disconnected")
    
    async def start(self):
        """Start the orchestrator server."""
        if self.running:
            raise RuntimeError("Server is already running")
        
        logger.info(f"Starting orchestrator server on {self.config.host}:{self.config.port}")
        
        # Create SSL context
        ssl_context = self.cert_manager.create_ssl_context(is_server=True)
        
        # Start server
        self.server = await asyncio.start_server(
            self.handle_client,
            host=self.config.host,
            port=self.config.port,
            ssl=ssl_context,
            reuse_address=True,
            limit=10 * 1024 * 1024  # 10MB buffer for large tool results
        )
        
        self.running = True
        logger.info(f"Orchestrator server listening on {self.config.host}:{self.config.port}")
        
        # Run server
        async with self.server:
            await self.server.serve_forever()
    
    async def stop(self):
        """Stop the orchestrator server."""
        if not self.running:
            return
        
        logger.info("Stopping orchestrator server...")
        self.running = False
        
        # Close all client connections
        for client_id in list(self.clients.keys()):
            await self.remove_client(client_id)
        
        # Close server
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        
        logger.info("Orchestrator server stopped")
    
    def _get_operation_identifier(self, tool: str, command: str, args: Dict[str, Any]) -> str:
        """
        Extract operation identifier for consistent approval logic across tool types.
        
        RATIONALE: Built-in tools use 'action' parameter in args, MCP tools use 'command' parameter.
        This abstraction allows unified approval logic while respecting each tool's interface.
        
        Examples:
        - filesystem tool: args={'action': 'read', 'path': '...'} -> returns 'read'
        - MCP postgres tool: command='select', args={'query': '...'} -> returns 'select'
        
        Args:
            tool: Tool name (e.g., 'filesystem', 'postgres_tool')
            command: Command parameter from tool call
            args: Arguments dictionary from tool call
            
        Returns:
            Operation identifier string for approval checking
        """
        if tool in ['filesystem', 'chat_info', 'tts_config', 'bash']:
            # Built-in tools: operation is in args['action']
            operation = args.get('action', 'unknown')
            logger.debug(f"Built-in tool {tool}: extracted operation '{operation}' from args.action")
            return operation
        else:
            # MCP tools: operation is the command parameter
            logger.debug(f"MCP tool {tool}: using command '{command}' as operation")
            return command
    
    def _get_tool_default_approval(self, tool: str) -> bool:
        """
        Define default approval behavior per tool type for backward compatibility.
        
        RATIONALE: Each tool type has different security implications and historical behavior:
        - filesystem: historically auto-approved unless explicitly in require_approval_operations
        - chat_info/tts_config: safe operations, historically auto-approved
        - bash: dangerous shell access, should require approval by default for security
        - MCP tools: unknown capabilities from external sources, require approval by default
        
        This maintains backward compatibility while providing secure defaults for new tools.
        
        Args:
            tool: Tool name
            
        Returns:
            True if approval required by default, False if auto-approved by default
        """
        # Built-in tools with historically permissive defaults (maintains current behavior)
        if tool in ['filesystem', 'chat_info', 'tts_config']:
            logger.debug(f"Tool {tool}: using permissive default (auto-approve)")
            return False  # Auto-approve by default
        
        # Security-sensitive tools require approval by default
        elif tool == 'bash':
            logger.debug(f"Tool {tool}: using secure default (require approval)")
            return True   # Require approval by default (secure default)
        
        # MCP tools and unknown tools - be conservative for security
        else:
            logger.debug(f"Tool {tool}: unknown tool type, using secure default (require approval)")
            return True   # Require approval by default (secure default)

    def _requires_approval(self, tool: str, command: str, args: Dict[str, Any], agent_id: str = None) -> bool:
        """
        Unified approval logic for ALL tools.
        
        PRECEDENCE ORDER (highest to lowest):
        1. YOLO mode - auto-approve everything (security bypass for testing)
        2. Paranoid mode - require approval for everything (security override)
        3. auto_approve_operations - explicit allow list (user preference)
        4. require_approval_operations - explicit deny list (security policy)
        5. Tool default behavior - maintain backward compatibility
        
        RATIONALE: This order ensures security overrides work while honoring user config.
        The unified logic makes auto_approve_operations work for built-in tools, fixing
        the inconsistency where they were previously ignored.
        
        Args:
            tool: Tool name (e.g., 'filesystem', 'bash', 'postgres_tool')
            command: Command from tool call (used for MCP tools)
            args: Arguments dict from tool call (contains 'action' for built-in tools)
            agent_id: Agent identifier for per-agent config overrides
            
        Returns:
            True if approval required, False if auto-approved
        """
        # 1. YOLO mode override - bypass all security for testing/development
        if self.yolo_mode:
            operation = self._get_operation_identifier(tool, command, args)
            logger.debug(f"YOLO mode: auto-approving {tool}.{operation} for agent {agent_id}")
            return False
            
        # 2. Get tool configuration with agent-specific overrides applied
        tool_config = self._get_tool_config(tool, agent_id)
        
        # 3. Paranoid mode override - security-first approach requires approval for everything
        if tool_config.paranoid_mode:
            operation = self._get_operation_identifier(tool, command, args)
            logger.info(f"Paranoid mode: requiring approval for {tool}.{operation} for agent {agent_id}")
            return True
        
        # 4. Get normalized operation identifier for consistent logic across tool types
        operation = self._get_operation_identifier(tool, command, args)
        
        # 5. Check auto-approve first - user explicitly trusts this operation
        if operation in tool_config.auto_approve_operations:
            logger.debug(f"Auto-approved: {tool}.{operation} for agent {agent_id} (in auto_approve_operations)")
            return False
            
        # 6. Check require-approval - user/admin explicitly distrusts this operation
        if operation in tool_config.require_approval_operations:
            logger.info(f"Approval required: {tool}.{operation} for agent {agent_id} (in require_approval_operations)")
            return True
            
        # 7. Fall back to tool-specific default - maintain backward compatibility
        default_requires_approval = self._get_tool_default_approval(tool)
        logger.debug(f"Using default for {tool}.{operation} for agent {agent_id}: {'approval required' if default_requires_approval else 'auto-approved'}")
        return default_requires_approval
    
    def _get_tool_config(self, tool_name: str, agent_id: str = None) -> ToolConfig:
        """Get tool configuration with optional agent overrides."""
        # Start with global tool config or create default
        if tool_name in self.tool_configs:
            base_config = self.tool_configs[tool_name]
        elif tool_name in self.mcp_configs:
            base_config = self.mcp_configs[tool_name]
        else:
            # Create default configuration
            if tool_name == "filesystem":
                base_config = FilesystemToolConfig(
                    name=tool_name,
                    allowed_paths=["/tmp/neurodeck", "/tmp"],
                    auto_approve_operations=["read", "list"],
                    require_approval_operations=["write", "delete"]
                )
            elif tool_name == "chat_info":
                base_config = ToolConfig(
                    name=tool_name,
                    auto_approve_operations=["list_participants", "get_chat_status", "get_my_info", "get_participant_info"],
                    require_approval_operations=[],  # All chat_info operations are safe
                    execution_timeout=10.0
                )
            elif tool_name == "tts_config":
                base_config = ToolConfig(
                    name=tool_name,
                    auto_approve_operations=["get_voice_settings", "set_voice_settings", "list_available_voices", "reset_voice_settings", "get_help", "help"],
                    require_approval_operations=[],  # All tts_config operations are safe
                    execution_timeout=5.0
                )
            else:
                base_config = ToolConfig(
                    name=tool_name,
                    require_approval_operations=["*"]
                )
        
        # Apply agent-specific overrides if they exist
        if (agent_id and agent_id in self.agent_tool_overrides and 
            tool_name in self.agent_tool_overrides[agent_id]):
            override_config = self.agent_tool_overrides[agent_id][tool_name]
            return self._merge_tool_configs(base_config, override_config)
        
        return base_config
    
    def _merge_tool_configs(self, base: ToolConfig, override: ToolConfig) -> ToolConfig:
        """Merge tool configurations (override takes precedence)."""
        if isinstance(base, FilesystemToolConfig) and isinstance(override, FilesystemToolConfig):
            return FilesystemToolConfig(
                name=base.name,
                enabled=override.enabled if override.enabled is not None else base.enabled,
                execution_timeout=override.execution_timeout if override.execution_timeout is not None else base.execution_timeout,
                approval_timeout=override.approval_timeout if override.approval_timeout is not None else base.approval_timeout,
                auto_approve_operations=override.auto_approve_operations if override.auto_approve_operations is not None else base.auto_approve_operations,
                require_approval_operations=override.require_approval_operations if override.require_approval_operations is not None else base.require_approval_operations,
                paranoid_mode=override.paranoid_mode if override.paranoid_mode is not None else base.paranoid_mode,
                allowed_paths=override.allowed_paths if override.allowed_paths is not None else base.allowed_paths,
                max_file_size=override.max_file_size if override.max_file_size is not None else base.max_file_size
            )
        else:
            return ToolConfig(
                name=base.name,
                enabled=override.enabled if override.enabled is not None else base.enabled,
                execution_timeout=override.execution_timeout if override.execution_timeout is not None else base.execution_timeout,
                approval_timeout=override.approval_timeout if override.approval_timeout is not None else base.approval_timeout,
                auto_approve_operations=override.auto_approve_operations if override.auto_approve_operations is not None else base.auto_approve_operations,
                require_approval_operations=override.require_approval_operations if override.require_approval_operations is not None else base.require_approval_operations,
                paranoid_mode=override.paranoid_mode if override.paranoid_mode is not None else base.paranoid_mode
            )
    
    async def _request_approval(self, agent: str, tool: str, 
                               command: str, args: Dict[str, Any], 
                               request_id: str):
        """Send approval request to UI clients."""
        # Create human-readable description
        description = self._format_tool_description(tool, command, args)
        
        approval_request = tool_approval_request_message(
            agent=agent,
            tool=tool,
            command=command,
            args=args,
            description=description,
            request_id=request_id
        )
        
        # Send to all connected UI clients
        await self.broadcast_message(approval_request, target_type="ui")
    
    def _format_tool_description(self, tool: str, command: str, 
                                args: Dict[str, Any]) -> str:
        """Format human-readable tool operation description."""
        if tool == "filesystem":
            action = args.get("action")
            path = args.get("path")
            
            if action == "read":
                return f"Read file: {path}"
            elif action == "write":
                content_len = len(args.get("content", ""))
                return f"Write {content_len} bytes to: {path}"
            elif action == "list":
                return f"List directory: {path}"
            elif action == "delete":
                return f"Delete: {path}"
        
        return f"{tool}.{command} with args: {args}"
    
    async def _execute_approved_tool(self, agent_id: str, agent_name: str, tool: str,
                                    command: str, args: Dict[str, Any],
                                    request_id: str):
        """Execute tool after approval."""
        try:
            # Log tool execution start
            await self._log_tool_executing(agent_name, tool, command, request_id)
            
            # Execute tool
            result = await self.tool_executor.execute_tool(
                agent_id, agent_name, tool, command, args
            )
            
            # Log successful execution
            await self._log_tool_success(agent_name, tool, command, request_id, result)
            
            # Send success result
            result_msg = tool_result_message(
                request_id=request_id,
                success=True,
                result=result
            )
            
            # Find agent connection and send result
            for client_id, conn in list(self.clients.items()):
                if client_id == agent_id:
                    await conn.send_message(result_msg)
                    break
            
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            
            # Log execution failure
            await self._log_tool_error(agent_name, tool, command, request_id, str(e))
            
            # Send error result
            error_msg = tool_result_message(
                request_id=request_id,
                success=False,
                error=str(e)
            )
            
            # Find agent connection and send error
            for client_id, conn in list(self.clients.items()):
                if client_id == agent_id:
                    await conn.send_message(error_msg)
                    break
    
    async def _approval_timeout(self, request_id: str):
        """Handle approval timeout."""
        await asyncio.sleep(self.approval_timeout)
        
        if request_id in self.pending_tool_approvals:
            pending = self.pending_tool_approvals.pop(request_id)
            
            # Send timeout error to agent
            error_msg = tool_result_message(
                request_id=request_id,
                success=False,
                error="Tool approval timed out"
            )
            
            # Find agent connection and send timeout error
            for client_id, conn in list(self.clients.items()):
                if client_id == pending["agent_id"]:
                    await conn.send_message(error_msg)
                    break
            
            logger.warning(f"Tool approval timed out: {request_id}")
    
    # Tool logging helpers for enhanced debug visibility
    async def _log_tool_request(self, agent_name: str, tool: str, command: str, args: Dict[str, Any], request_id: str):
        """Log tool request to UI debug panel."""
        action = args.get("action", command)
        # Create concise but informative log message
        args_summary = self._format_args_summary(tool, action, args)
        message = f"🔧 {agent_name} requests {tool}.{action} {args_summary}"
        
        log_msg = system_log_message(
            sender="tool-system", 
            level=LogLevel.INFO, 
            message=message
        )
        await self.broadcast_system_log(log_msg)
    
    async def _log_tool_approval_required(self, agent_name: str, tool: str, command: str, request_id: str):
        """Log that tool requires approval."""
        message = f"⏳ {agent_name}'s {tool}.{command} requires approval [{request_id[:8]}]"
        
        log_msg = system_log_message(
            sender="tool-system", 
            level=LogLevel.WARNING, 
            message=message
        )
        await self.broadcast_system_log(log_msg)
    
    async def _log_tool_auto_approved(self, agent_name: str, tool: str, command: str, request_id: str):
        """Log that tool was auto-approved."""
        message = f"✅ {agent_name}'s {tool}.{command} auto-approved [{request_id[:8]}]"
        
        log_msg = system_log_message(
            sender="tool-system", 
            level=LogLevel.INFO, 
            message=message
        )
        await self.broadcast_system_log(log_msg)
    
    async def _log_tool_approved(self, agent_id: str, tool: str, command: str, request_id: str):
        """Log that tool was manually approved."""
        agent_name = self._extract_agent_name(agent_id)
        message = f"✅ {agent_name}'s {tool}.{command} approved by user [{request_id[:8]}]"
        
        log_msg = system_log_message(
            sender="tool-system", 
            level=LogLevel.INFO, 
            message=message
        )
        await self.broadcast_system_log(log_msg)
    
    async def _log_tool_denied(self, agent_id: str, tool: str, command: str, request_id: str):
        """Log that tool was denied."""
        agent_name = self._extract_agent_name(agent_id)
        message = f"❌ {agent_name}'s {tool}.{command} denied by user [{request_id[:8]}]"
        
        log_msg = system_log_message(
            sender="tool-system", 
            level=LogLevel.WARNING, 
            message=message
        )
        await self.broadcast_system_log(log_msg)
    
    async def _log_tool_executing(self, agent_name: str, tool: str, command: str, request_id: str):
        """Log that tool execution is starting."""
        message = f"⚡ Executing {agent_name}'s {tool}.{command} [{request_id[:8]}]"
        
        log_msg = system_log_message(
            sender="tool-system", 
            level=LogLevel.INFO, 
            message=message
        )
        await self.broadcast_system_log(log_msg)
    
    async def _log_tool_success(self, agent_name: str, tool: str, command: str, request_id: str, result: Any):
        """Log successful tool execution."""
        # Summarize result for logging
        result_summary = self._format_result_summary(tool, command, result)
        message = f"✅ {agent_name}'s {tool}.{command} completed {result_summary} [{request_id[:8]}]"
        
        log_msg = system_log_message(
            sender="tool-system", 
            level=LogLevel.INFO, 
            message=message
        )
        await self.broadcast_system_log(log_msg)
    
    async def _log_tool_error(self, agent_name: str, tool: str, command: str, request_id: str, error: str):
        """Log tool execution error."""
        # Truncate long error messages
        error_short = error[:100] + "..." if len(error) > 100 else error
        message = f"❌ {agent_name}'s {tool}.{command} failed: {error_short} [{request_id[:8]}]"
        
        log_msg = system_log_message(
            sender="tool-system", 
            level=LogLevel.ERROR, 
            message=message
        )
        await self.broadcast_system_log(log_msg)
    
    def _extract_agent_name(self, agent_id: str) -> str:
        """Extract agent name from agent_id (format: agent-name)."""
        if "-" in agent_id:
            return agent_id.split("-", 1)[1]
        return agent_id
    
    def _format_args_summary(self, tool: str, action: str, args: Dict[str, Any]) -> str:
        """Format tool arguments for concise logging."""
        if tool == "filesystem":
            path = args.get("path", "")
            if action == "read":
                return f"({path})"
            elif action == "write":
                content_len = len(args.get("content", ""))
                return f"({content_len} bytes to {path})"
            elif action == "list":
                return f"({path})"
            elif action == "delete":
                return f"({path})"
        elif tool == "tts_config":
            if action in ["get_help", "list_available_voices"]:
                return ""
            elif action == "get_voice_settings":
                return f"(for {args.get('agent_name', 'agent')})"
            elif action == "set_voice_settings":
                params = []
                if "length_scale" in args:
                    params.append(f"speed={args['length_scale']}")
                if "inference_noise_scale" in args:
                    params.append(f"expr={args['inference_noise_scale']}")
                if "voice_id" in args:
                    params.append(f"voice={args['voice_id']}")
                return f"({', '.join(params)})" if params else ""
        elif tool == "chat_info":
            if action == "get_participant_info":
                return f"({args.get('participant_name', 'unknown')})"
            return ""
        
        # Default: show key args
        key_args = []
        for key in ["action", "query", "path", "message"]:
            if key in args and args[key]:
                value = str(args[key])
                if len(value) > 30:
                    value = value[:27] + "..."
                key_args.append(f"{key}={value}")
        return f"({', '.join(key_args)})" if key_args else ""
    
    def _format_result_summary(self, tool: str, command: str, result: Any) -> str:
        """Format tool result for concise logging."""
        if isinstance(result, dict):
            if tool == "filesystem":
                if "content" in result:
                    content_len = len(result["content"])
                    return f"({content_len} bytes)"
                elif "files" in result:
                    file_count = len(result["files"])
                    return f"({file_count} files)"
                elif "success" in result and result["success"]:
                    return "(success)"
            elif tool == "tts_config":
                if "current_settings" in result:
                    return "(settings retrieved)"
                elif "success" in result and result["success"]:
                    return "(updated)"
            elif tool == "chat_info":
                if "participants" in result:
                    count = result.get("total_count", 0)
                    return f"({count} participants)"
                elif "agent_name" in result:
                    return f"(info for {result['agent_name']})"
        
        # Default: success indicator
        return "(completed)"


async def test_server():
    """Test the orchestrator server."""
    from ..common.config import OrchestratorConfig
    
    # Test config
    config = OrchestratorConfig(
        host="localhost",
        port=9999,
        tls_cert="config/certs/server.crt",
        tls_key="config/certs/server.key",
        auth_token="test-token-123",
        log_level="INFO"
    )
    
    server = OrchestratorServer(config)
    
    try:
        await server.start()
    except KeyboardInterrupt:
        await server.stop()


if __name__ == "__main__":
    asyncio.run(test_server())