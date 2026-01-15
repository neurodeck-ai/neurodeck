"""
Base agent class for NeuroDeck AI agents.

All AI agents inherit from this class and implement provider-specific methods.
Handles socket communication, context persistence, and tool execution.
"""

import asyncio
import ssl
import sys
import os
import argparse
import signal
import random
import traceback
import uuid
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import re

from dotenv import load_dotenv

from ..common.protocol import (
    MessageProtocol, MessageType, LogLevel,
    auth_message, chat_message, system_log_message,
    tool_request_message, tool_approval_request_message, tool_result_message
)
from ..common.security import CertificateManager
from ..common.database import ContextDatabase
from ..common.config import ConfigManager, AgentConfig
from ..common.logging import get_logger


class BaseAgent(ABC):
    """
    Base class for all AI agent processes.
    
    Each agent runs as a separate process and connects to the orchestrator
    via TLS socket using the same JSON protocol as the TUI client.
    """
    
    def __init__(
        self,
        name: str,
        host: str,
        port: int,
        token: str,
        config_path: str = "config/agents.ini"
    ):
        # Load environment variables from .env file
        env_path = Path("config/.env")
        if env_path.exists():
            load_dotenv(env_path)
            print(f"Agent {name} loaded .env from {env_path}")
        else:
            print(f"Agent {name} - .env file not found at {env_path}")
        
        self.name = name
        self.host = host
        self.port = port
        self.token = token
        self.config_path = config_path
        
        # Connection state
        self.reader: Optional[asyncio.StreamReader] = None
        self.writer: Optional[asyncio.StreamWriter] = None
        self.connected = False
        self.authenticated = False
        self.running = False
        
        # Configuration
        self.config_manager = ConfigManager(config_path)
        self.agent_config = self._load_agent_config()
        self.orchestrator_config = self.config_manager.load_orchestrator_config()
        self.mcp_tools = self.config_manager.load_mcp_tool_configs()
        
        # Context management
        self.context_db = ContextDatabase()
        self.conversation_context: List[Dict[str, Any]] = []
        self.max_context_messages = self.agent_config.max_context_messages
        
        # Response delay management
        self.response_delay_max = self.agent_config.response_delay_max
        self.response_timer: Optional[asyncio.Task] = None
        self.pending_message: Optional[Dict[str, Any]] = None
        
        # Heartbeat management
        self.heartbeat_interval = self.agent_config.heartbeat_interval
        self.messages_processed = 0
        self.last_response_time: Optional[str] = None
        
        # Tool management
        self.available_tools = set(self.agent_config.tools)
        self.built_in_tools = self.config_manager.get_built_in_tools()
        self.mcp_clients = {}  # tool_name -> MCP client
        self.pending_tool_results: Dict[str, asyncio.Future] = {}
        
        # Load tool configurations locally for dynamic tool definitions
        self.tool_configs = self.config_manager.load_tool_configs()
        self.agent_tool_overrides = self.config_manager.load_agent_tool_overrides()
        
        # Logging
        self.logger = get_logger(f"agent.{name}")
        
        # API client (to be set by subclasses)
        self.api_client = None
        
        # Note: Cannot use send_system_log here as we're not connected yet
        self.logger.info(f"Agent {name} initialized")
    
    def _load_agent_config(self) -> AgentConfig:
        """Load configuration for this agent."""
        agents = self.config_manager.load_agent_configs()
        if self.name not in agents:
            raise ValueError(f"Agent {self.name} not found in configuration")
        return agents[self.name]
    
    async def start(self):
        """Start the agent process."""
        if self.running:
            return
        
        self.logger.info(f"Starting agent {self.name}...")
        self.running = True
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        # Set up debug signal handler for stack traces
        signal.signal(signal.SIGUSR1, self._debug_signal_handler)
        
        try:
            # Connect to orchestrator first
            await self.connect_to_orchestrator()
            
            # Authenticate
            await self.authenticate()
            
            # Initialize API client (can now send system logs)
            await self.initialize_api_client()
            
            # Load conversation context from database
            await self.load_context()
            
            # Initialize tools
            await self.initialize_tools()
            
            # Start message handling loop
            await self.message_loop()
            
        except Exception as e:
            self.logger.error(f"Agent {self.name} startup failed: {e}")
            # Cannot use send_system_log here - not in async context and likely not connected
            raise
        finally:
            await self.shutdown()
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Agent {self.name} received signal {signum}, shutting down...")
        self.running = False
    
    def _debug_signal_handler(self, signum, frame):
        """Handle debug signal (SIGUSR1) to dump stack trace."""
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Get stack trace for all threads
        stack_info = []
        stack_info.append(f"=== STACK TRACE FOR AGENT {self.name} at {timestamp} ===")
        stack_info.append(f"PID: {os.getpid()}")
        stack_info.append(f"Agent status: running={self.running}, connected={self.connected}, authenticated={self.authenticated}")
        stack_info.append(f"Messages processed: {self.messages_processed}")
        stack_info.append(f"Response timer active: {self.response_timer is not None and not self.response_timer.done() if self.response_timer else False}")
        stack_info.append("")
        
        # Current thread stack trace
        stack_info.append("MAIN THREAD STACK TRACE:")
        for line in traceback.format_stack(frame):
            stack_info.append(line.rstrip())
        
        stack_info.append("=== END STACK TRACE ===")
        
        # Write to both logger and stderr for immediate visibility
        stack_trace_text = "\n".join(stack_info)
        self.logger.error(stack_trace_text)
        print(stack_trace_text, file=sys.stderr, flush=True)
        
        # Also try to send to orchestrator if connected
        if self.connected and self.authenticated:
            try:
                # Create a simple async task to send the debug info
                import asyncio
                
                def send_debug_info():
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            # Schedule the coroutine to run
                            asyncio.create_task(self.send_system_log(LogLevel.ERROR, f"STACK TRACE:\n{stack_trace_text}"))
                    except Exception as e:
                        print(f"Failed to send debug info to orchestrator: {e}", file=sys.stderr)
                
                send_debug_info()
            except Exception as e:
                print(f"Failed to send stack trace to orchestrator: {e}", file=sys.stderr)
    
    async def connect_to_orchestrator(self):
        """Connect to the orchestrator via TLS socket."""
        self.logger.info(f"Connecting to orchestrator at {self.host}:{self.port}")
        
        # Create SSL context for client
        cert_manager = CertificateManager()
        ssl_context = cert_manager.create_ssl_context(is_server=False)
        
        try:
            self.reader, self.writer = await asyncio.open_connection(
                self.host, self.port, ssl=ssl_context,
                limit=10 * 1024 * 1024  # 10MB buffer for large tool results
            )
            self.connected = True
            self.logger.info("Connected to orchestrator")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to orchestrator: {e}")
            raise
    
    async def authenticate(self):
        """Authenticate with the orchestrator."""
        client_id = f"agent-{self.name}"
        auth_msg = auth_message(self.token, client_id)
        
        await self.send_message(auth_msg)
        
        # Wait for authentication result
        response = await self.receive_message()
        if not response or response.get("type") != MessageType.AUTH_RESULT.value:
            raise RuntimeError("Invalid authentication response")
        
        if not response.get("success"):
            error_msg = response.get("message", "Authentication failed")
            raise RuntimeError(f"Authentication failed: {error_msg}")
        
        self.authenticated = True
        self.logger.info("Authenticated with orchestrator")
        self.logger.info(f"Agent {self.name} connected and authenticated")
    
    async def send_message(self, message_dict: Dict[str, Any]):
        """Send message to orchestrator."""
        if not self.writer:
            error_msg = f"Agent {self.name} not connected to orchestrator - cannot send message"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        message_type = message_dict.get("type", "unknown")
        # Note: Cannot use send_system_log here as this IS the send mechanism
        
        # Debug: Log what message types we're sending to trace routing issues
        if message_type in ["system_log", "chat"]:
            content_preview = str(message_dict.get("content", message_dict.get("message", "")))[:50]
            self.logger.debug(f"SEND_MESSAGE_DEBUG: Sending {message_type} - '{content_preview}...'")
        
        try:
            encoded = MessageProtocol.encode_message(message_dict)
            self.logger.debug(f"Agent {self.name} encoded message: {encoded[:200]}{'...' if len(encoded) > 200 else ''}")
            
            self.writer.write(encoded)
            await self.writer.drain()
            self.logger.debug(f"Agent {self.name} successfully sent {message_type} message")
            
        except Exception as e:
            self.logger.error(f"Agent {self.name} failed to send {message_type} message: {e}")
            import traceback
            self.logger.error(f"Agent {self.name} send_message traceback: {traceback.format_exc()}")
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
            self.logger.error(f"Failed to receive message: {e}")
            # Mark connection as broken so message loop will exit
            self.connected = False
            return None
    
    async def send_chat_response(self, content: str):
        """Send chat response to orchestrator (will be broadcast to all clients)."""
        self.logger.debug(f"Agent {self.name} preparing to send chat response")
        self.logger.debug(f"Content: '{content}'")
        
        try:
            # Extract thinking content from <think> tags and send to debug log
            thinking_matches = re.findall(r'<think>(.*?)</think>', content, re.DOTALL)
            for thinking in thinking_matches:
                thinking_clean = thinking.strip()
                if thinking_clean:
                    await self.send_system_log(LogLevel.DEBUG, f"🧠 {self.name} THINKING: {thinking_clean}")
            
            # Remove thinking tags from the content to send to chat
            clean_content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            
            # Only send if there's content left after removing thinking tags
            if clean_content:
                chat_msg = chat_message(self.name, clean_content)
                self.logger.debug(f"Agent {self.name} created chat message: {chat_msg}")
                
                await self.send_message(chat_msg)
                self.logger.debug(f"Agent {self.name} successfully sent chat response to orchestrator")
                
                # Save to context database with message_id from the message we're sending
                message_id = chat_msg.get("message_id")
                # FIXED: Save original content (with thinking blocks) to preserve reasoning context
                self.context_db.save_message(self.name, "assistant", content, message_id=message_id)
                self.logger.debug(f"Agent {self.name} saved response to context database")
            else:
                # Even if no clean content, save the thinking-only response to context for reasoning continuity
                self.context_db.save_message(self.name, "assistant", content)
                self.logger.debug(f"Agent {self.name} response contained only thinking tags, not sending to chat but saved to context")
                
                # Trigger follow-up call to continue reasoning after thinking-only response
                await self._continue_thinking_response()
                
        except Exception as e:
            self.logger.error(f"Agent {self.name} failed to send chat response: {e}")
            import traceback
            self.logger.error(f"Agent {self.name} send_chat_response traceback: {traceback.format_exc()}")
            raise
    
    async def _continue_thinking_response(self):
        """Continue reasoning after a thinking-only response."""
        try:
            self.logger.info(f"Agent {self.name} attempting to continue reasoning after thinking-only response")
            
            # Get recent context for follow-up
            context_messages = self.context_db.get_recent_context(self.name, limit=self.agent_config.max_context_messages)
            
            # Generate follow-up response
            follow_up_response = await self.generate_response(context_messages)
            
            if follow_up_response:
                self.logger.info(f"Agent {self.name} generated follow-up response ({len(follow_up_response)} chars)")
                # This will go through send_chat_response again, preserving the cycle
                await self.send_chat_response(follow_up_response)
            else:
                self.logger.debug(f"Agent {self.name} follow-up attempt produced no response")
                
        except Exception as e:
            self.logger.error(f"Agent {self.name} failed to continue thinking response: {e}")
            # Don't re-raise to avoid breaking the main flow
    
    async def send_system_log(self, level: LogLevel, message: str):
        """Send system log message (only visible to TUI, not broadcast to agents)."""
        # Always log locally for unit tests and debugging
        if level == LogLevel.DEBUG:
            self.logger.debug(message)
        elif level == LogLevel.INFO:
            self.logger.info(message)
        elif level == LogLevel.WARNING:
            self.logger.warning(message)
        elif level == LogLevel.ERROR:
            self.logger.error(message)
        
        if not self.connected or not self.authenticated:
            # If not connected, print to stderr so we don't lose debug info
            import sys
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] AGENT-{self.name} {level.value.upper()}: {message}", file=sys.stderr)
            return
            
        try:
            log_msg = system_log_message(self.name, level, message)
            await self.send_message(log_msg)
        except Exception as e:
            # Fallback to stderr if sending fails
            import sys
            print(f"AGENT-{self.name} {level.value.upper()}: {message} (send failed: {e})", file=sys.stderr)
    
    async def send_debug_heartbeat(self):
        """Send periodic heartbeat via debug log."""
        from datetime import datetime
        current_time = datetime.now().strftime("%H:%M:%S")
        
        heartbeat_msg = (
            f"HEARTBEAT - Messages: {self.messages_processed}, "
            f"Context: {len(self.conversation_context)}, "
            f"Last response: {self.last_response_time or 'Never'}"
        )
        
        # Send to orchestrator via system log so it's visible in debug logs
        await self.send_system_log(LogLevel.DEBUG, heartbeat_msg)
    
    async def load_context(self):
        """Load conversation context from database across all sessions."""
        messages = self.context_db.load_context_all_sessions(self.name, self.max_context_messages)
        self.conversation_context = []
        
        for msg in messages:
            sender = msg["sender"]
            content = msg["content"]
            
            if msg["type"] == "user":
                role = "user"
                # Add attribution for actual user messages
                content = f"@{self.orchestrator_config.username} said: {content}"
            elif msg["type"] == "assistant":
                role = "user"  # Present agent messages as user messages with attribution
                # Add agent attribution using the actual sender
                content = f"@{sender} said: {content}"
            else:
                role = "system"
                # Don't modify system messages
            
            self.conversation_context.append({
                "role": role,
                "content": content
            })
        
        self.logger.info(f"Loaded {len(self.conversation_context)} messages from context")
    
    def should_respond_to_message(self, message: Dict[str, Any]) -> bool:
        """
        Determine if agent should respond to a chat message.
        
        Agents respond to:
        1. Messages from "user" - always respond (unless directed at another agent)
        2. Messages from other agents - only if tagged with @agent_name or @everyone
        3. @everyone messages - all agents respond
        4. Multi-agent tagging is supported (e.g., "@human hello @kimi @grok")
        """
        content = message.get("content", "").lower()
        sender = message.get("sender", "").lower()
        
        # Don't respond to our own messages
        if sender == self.name.lower():
            return False
        
        agent_name_lower = self.name.lower()
        
        # Always check for @mention pattern or @everyone/@all anywhere in the message
        if f"@{agent_name_lower}" in content or "@everyone" in content or "@all" in content:
            return True
        
        # For user messages, respond if not directed at a specific other agent
        if sender == "user":
            # Check if message starts with @someone_else (but not us or @everyone)
            if content.startswith("@"):
                words = content.split()
                if words:
                    mentioned_name = words[0][1:]  # Remove the @ symbol
                    # Don't respond if message starts with @someone_else (unless we're also tagged)
                    # @everyone and @all are universal mentions that all agents respond to
                    if mentioned_name != agent_name_lower and mentioned_name not in ("everyone", "all"):
                        return False
            # User message not directed elsewhere, respond
            return True
        
        # For agent messages, only respond if we found a tag (already checked above)
        return False
    
    
    async def message_loop(self):
        """Main message handling loop with heartbeat timeout."""
        self.logger.info("Starting message loop")
        await self.send_system_log(LogLevel.DEBUG, f"Agent {self.name} entering message loop")
        
        while self.running and self.connected:
            try:
                await self.send_system_log(LogLevel.DEBUG, f"Agent {self.name} waiting for message (timeout: {self.heartbeat_interval}s)...")
                
                # Use timeout to force periodic heartbeat with shorter socket timeout
                try:
                    message = await asyncio.wait_for(
                        self.receive_message(), 
                        timeout=10.0  # Shorter timeout to detect socket hangs faster
                    )
                    await self.send_system_log(LogLevel.DEBUG, f"Agent {self.name} received message from wait_for")
                except asyncio.TimeoutError:
                    # Check if we should send heartbeat or just continue waiting
                    import time
                    current_time = time.time()
                    if not hasattr(self, '_last_heartbeat_time'):
                        self._last_heartbeat_time = current_time
                    
                    if current_time - self._last_heartbeat_time >= self.heartbeat_interval:
                        await self.send_system_log(LogLevel.DEBUG, f"Agent {self.name} heartbeat timeout triggered")
                        await self.send_debug_heartbeat()
                        self._last_heartbeat_time = current_time
                    continue  # Back to waiting for messages
                
                if message is None:
                    # Connection closed
                    await self.send_system_log(LogLevel.DEBUG, f"Agent {self.name} received None message - connection closed")
                    break
                
                message_type = message.get("type", "unknown")
                await self.send_system_log(LogLevel.DEBUG, f"Agent {self.name} received {message_type} message")
                
                # Increment message counter
                self.messages_processed += 1
                
                await self.handle_message(message)
                
            except Exception as e:
                self.logger.error(f"Error in message loop: {e}")
                await self.send_system_log(LogLevel.ERROR, f"Message handling error: {e}")
                # Continue loop - don't crash on single message errors
    
    async def handle_message(self, message: Dict[str, Any]):
        """Handle incoming message from orchestrator."""
        message_type = message.get("type")
        
        await self.send_system_log(LogLevel.DEBUG, f"Agent {self.name} handling {message_type} message")
        
        if message_type == MessageType.CHAT.value:
            await self.send_system_log(LogLevel.DEBUG, f"Agent {self.name} processing CHAT message")
            await self.handle_chat_message(message)
        elif message_type == MessageType.CLEAR_CONTEXT.value:
            await self.send_system_log(LogLevel.DEBUG, f"Agent {self.name} processing CLEAR_CONTEXT message")
            await self.handle_clear_context(message)
        elif message_type == MessageType.TOOL_APPROVAL_RESULT.value:
            await self.send_system_log(LogLevel.DEBUG, f"Agent {self.name} processing TOOL_APPROVAL_RESULT message")
            await self.handle_tool_approval_result(message)
        elif message_type == MessageType.TOOL_RESULT.value:
            await self.send_system_log(LogLevel.DEBUG, f"Agent {self.name} processing TOOL_RESULT message")
            await self.handle_tool_result(message)
        else:
            await self.send_system_log(LogLevel.DEBUG, f"Agent {self.name} ignoring message type: {message_type}")
    
    async def handle_chat_message(self, message: Dict[str, Any]):
        """Handle incoming chat message."""
        sender = message.get("sender", "")
        content = message.get("content", "")
        
        await self.send_system_log(LogLevel.DEBUG, f"Agent {self.name} handling chat message from {sender}")
        
        # Save messages to context (preserving whether they're from user or other agents)
        if sender != self.name:
            message_id = message.get("message_id")  # Get message_id from protocol
            # Save as "user" only if sender is actually "user", otherwise save as "assistant" (from other agents)
            message_type = "user" if sender == "user" else "assistant"
            self.context_db.save_message(self.name, message_type, content, message_id=message_id)
            
            # Add to conversation context with attribution and timestamp
            from datetime import datetime
            timestamp = message.get("timestamp", datetime.now().isoformat())
            # Format timestamp for readability (date and time)
            try:
                if "T" in str(timestamp):
                    dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    time_str = timestamp
            except:
                time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if sender == "user":
                attributed_content = f"[{time_str}] @{self.orchestrator_config.username} said: {content}"
            else:
                attributed_content = f"[{time_str}] @{sender} said: {content}"
            
            self.conversation_context.append({
                "role": "user",
                "content": attributed_content
            })
            
            # Trim context if it exceeds limit
            self._trim_context()
            await self.send_system_log(LogLevel.DEBUG, f"Added to context (now {len(self.conversation_context)} messages)")
            
            # Check if we should respond to this message
            await self.send_system_log(LogLevel.DEBUG, f"Agent {self.name} checking should_respond_to_message...")
            if not self.should_respond_to_message(message):
                await self.send_system_log(LogLevel.DEBUG, f"Agent {self.name} will not respond - no mention/tag found")
                return
            await self.send_system_log(LogLevel.DEBUG, f"Agent {self.name} will respond - mention/tag found")
            
            # Cancel any existing timer with timeout
            if self.response_timer and not self.response_timer.done():
                await self.send_system_log(LogLevel.DEBUG, f"Agent {self.name} cancelling previous response timer...")
                try:
                    self.response_timer.cancel()
                    # Wait a bit for the cancellation to take effect
                    try:
                        await asyncio.wait_for(self.response_timer, timeout=0.1)
                    except (asyncio.CancelledError, asyncio.TimeoutError):
                        pass  # Expected - timer was cancelled or timed out
                    await self.send_system_log(LogLevel.DEBUG, f"Agent {self.name} successfully cancelled previous response timer")
                except Exception as e:
                    await self.send_system_log(LogLevel.ERROR, f"Agent {self.name} error cancelling timer: {e}")
            
            # Set up delayed response with error handling
            try:
                delay = random.randint(1, self.response_delay_max)
                await self.send_system_log(LogLevel.DEBUG, f"Agent {self.name} setting response timer for {delay} seconds")
                self.pending_message = message
                self.response_timer = asyncio.create_task(self._delayed_response(delay))
                await self.send_system_log(LogLevel.DEBUG, f"Agent {self.name} response timer created successfully")
            except Exception as e:
                await self.send_system_log(LogLevel.ERROR, f"Agent {self.name} error creating response timer: {e}")
                # Fallback: respond immediately if timer creation fails
                await self.send_system_log(LogLevel.DEBUG, f"Agent {self.name} falling back to immediate response...")
                response = await self.generate_response(self.conversation_context)
                if response:
                    await self.send_chat_response(response)
    
    async def _delayed_response(self, delay: int):
        """Execute delayed response after timer expires."""
        try:
            await asyncio.sleep(delay)
            await self.send_system_log(LogLevel.DEBUG, f"Agent {self.name} response timer expired, generating response...")
            
            # Generate response using AI provider with current context
            await self.send_system_log(LogLevel.DEBUG, f"Agent {self.name} calling generate_response with {len(self.conversation_context)} context messages")
            response = await self.generate_response(self.conversation_context)
            
            if response:
                await self.send_system_log(LogLevel.DEBUG, f"Agent {self.name} generated response ({len(response)} chars)")
                # Send response
                await self.send_system_log(LogLevel.DEBUG, f"Agent {self.name} sending chat response...")
                await self.send_chat_response(response)
                await self.send_system_log(LogLevel.DEBUG, f"Agent {self.name} chat response sent successfully")

                # Update last response time for heartbeat tracking
                from datetime import datetime
                self.last_response_time = datetime.now().strftime("%H:%M:%S")

                # Add to conversation context
                self.conversation_context.append({
                    "role": "assistant",
                    "content": response
                })

                # Trim context after adding response
                self._trim_context()
            else:
                await self.send_system_log(LogLevel.DEBUG, f"Agent {self.name} generated empty/None response")

                # Check if error feedback was injected (indicates we should retry)
                if self.conversation_context and self.conversation_context[-1].get("content", "").startswith("[SYSTEM ERROR"):
                    await self.send_system_log(LogLevel.INFO, f"Agent {self.name} retrying after error feedback...")
                    await asyncio.sleep(1)  # Brief pause before retry
                    retry_response = await self.generate_response(self.conversation_context)
                    if retry_response:
                        await self.send_system_log(LogLevel.DEBUG, f"Agent {self.name} retry succeeded ({len(retry_response)} chars)")
                        await self.send_chat_response(retry_response)
                        from datetime import datetime
                        self.last_response_time = datetime.now().strftime("%H:%M:%S")
                        self.conversation_context.append({
                            "role": "assistant",
                            "content": retry_response
                        })
                        self._trim_context()
                    else:
                        await self.send_system_log(LogLevel.WARNING, f"Agent {self.name} retry also failed")
                
        except asyncio.CancelledError:
            await self.send_system_log(LogLevel.DEBUG, f"Agent {self.name} response timer was cancelled")
            raise
        except Exception as e:
            await self.send_system_log(LogLevel.ERROR, f"Agent {self.name} error in delayed response: {e}")
            import traceback
            self.logger.error(f"Agent {self.name} traceback: {traceback.format_exc()}")
        finally:
            self.pending_message = None
    
    async def handle_clear_context(self, message: Dict[str, Any]):
        """Handle context clear command."""
        agent = message.get("agent")
        if agent == self.name or agent == "all":
            # Use all-sessions scope instead of current session only for complete wipe
            cleared = self.context_db.wipe_agent_all_sessions(self.name)
            self.conversation_context = []
            await self.send_system_log(LogLevel.INFO, f"Context cleared ({cleared} messages across all sessions)")
    
    async def handle_tool_result(self, message: Dict[str, Any]):
        """Handle tool execution result from orchestrator."""
        request_id = message.get("request_id")
        if request_id in self.pending_tool_results:
            self.pending_tool_results[request_id].set_result(message)

    async def handle_tool_approval_result(self, message: Dict[str, Any]):
        """Handle tool approval result from orchestrator."""
        # TODO: Implement tool approval handling
        pass

    def truncate_tool_result(self, result: Any, max_chars: int = 50000) -> str:
        """
        Truncate tool result if it exceeds max_chars, with clear indication of truncation.

        Args:
            result: The tool result (any type, will be converted to string)
            max_chars: Maximum characters to keep (default 50k to leave room for context)

        Returns:
            String result, possibly truncated with truncation notice
        """
        result_str = str(result)
        original_len = len(result_str)

        if original_len <= max_chars:
            return result_str

        # Calculate truncation stats
        truncated_len = max_chars - 500  # Leave room for the truncation notice
        removed_chars = original_len - truncated_len
        removed_pct = (removed_chars / original_len) * 100

        # Truncate and add notice
        truncated_result = result_str[:truncated_len]

        truncation_notice = (
            f"\n\n[TRUNCATED - Output too large for context window]\n"
            f"  Original size: {original_len:,} characters\n"
            f"  Shown: {truncated_len:,} characters\n"
            f"  Removed: {removed_chars:,} characters ({removed_pct:.1f}%)\n"
            f"  Tip: Use more targeted queries or process data in smaller chunks."
        )

        self.logger.warning(f"Tool result truncated: {original_len:,} -> {truncated_len:,} chars ({removed_pct:.1f}% removed)")

        return truncated_result + truncation_notice

    async def handle_tool_request(self, tool_name: str, tool_args: dict) -> Any:
        """Handle tool execution request from AI provider."""
        # Validate tool availability
        if tool_name not in self.available_tools:
            raise ValueError(f"Tool not available: {tool_name}")
        
        # Check if built-in or MCP tool
        if tool_name in self.built_in_tools:
            return await self.execute_built_in_tool(tool_name, tool_args)
        elif tool_name in self.mcp_clients:
            return await self.execute_mcp_tool(tool_name, tool_args)
        else:
            raise ValueError(f"Unknown tool type: {tool_name}")
    
    async def execute_built_in_tool(self, tool_name: str, args: dict) -> Any:
        """Execute built-in tool through orchestrator."""
        request_id = str(uuid.uuid4())

        # Get timeout from tool config or use default
        timeout = 30.0
        if tool_name in self.tool_configs:
            tool_config = self.tool_configs[tool_name]
            # Use approval_timeout + execution_timeout as total wait time
            timeout = tool_config.approval_timeout + tool_config.execution_timeout

        # Log tool request to context so agent remembers it made the request
        args_summary = str(args)[:200] + "..." if len(str(args)) > 200 else str(args)
        self.context_db.save_message(
            self.name, "tool_request",
            f"[Tool Request] {tool_name}: {args_summary}"
        )

        # Send tool request to orchestrator
        tool_req = tool_request_message(
            agent=self.name,
            tool=tool_name,
            command="execute",
            args=args,
            request_id=request_id
        )
        await self.send_message(tool_req)

        # Wait for tool result with clear timeout handling
        try:
            result = await self.wait_for_tool_result(request_id, timeout=timeout)
        except asyncio.TimeoutError:
            # Log timeout to context so agent knows what happened
            timeout_msg = (
                f"[Tool Timeout] {tool_name} request timed out after {timeout}s "
                f"waiting for user approval. You can retry if needed."
            )
            self.context_db.save_message(self.name, "tool_result", timeout_msg)
            # Also add to conversation context so agent sees it in next API call
            self.conversation_context.append({
                "role": "user",
                "content": f"[SYSTEM] {timeout_msg}"
            })
            raise Exception(timeout_msg)

        # Log result to context
        if result["success"]:
            result_summary = str(result.get("result", ""))[:500]
            self.context_db.save_message(
                self.name, "tool_result",
                f"[Tool Success] {tool_name}: {result_summary}"
            )
        else:
            self.context_db.save_message(
                self.name, "tool_result",
                f"[Tool Failed] {tool_name}: {result.get('error', 'Unknown error')}"
            )
            raise Exception(f"Tool execution failed: {result.get('error', 'Unknown error')}")

        return result["result"]
    
    async def wait_for_tool_result(self, request_id: str, timeout: float) -> Dict[str, Any]:
        """Wait for tool execution result from orchestrator."""
        self.pending_tool_results[request_id] = asyncio.Future()
        
        try:
            result = await asyncio.wait_for(
                self.pending_tool_results[request_id],
                timeout=timeout
            )
            return result
        finally:
            self.pending_tool_results.pop(request_id, None)

    async def execute_mcp_tool(self, tool_name: str, args: dict) -> Any:
        """Execute MCP tool."""
        if tool_name not in self.mcp_clients:
            raise ValueError(f"MCP tool not initialized: {tool_name}")

        mcp_client = self.mcp_clients[tool_name]

        # Get timeout from tool config or use default
        timeout = 60.0  # MCP tools get longer default
        if tool_name in self.tool_configs:
            tool_config = self.tool_configs[tool_name]
            timeout = tool_config.approval_timeout + tool_config.execution_timeout

        # Log tool request to context
        args_summary = str(args)[:200] + "..." if len(str(args)) > 200 else str(args)
        self.context_db.save_message(
            self.name, "tool_request",
            f"[MCP Tool Request] {tool_name}: {args_summary}"
        )

        # Send to orchestrator for approval first
        request_id = str(uuid.uuid4())

        tool_req = tool_request_message(
            agent=self.name,
            tool=tool_name,
            command="execute",
            args=args,
            request_id=request_id
        )

        await self.send_message(tool_req)

        # Wait for approval with clear timeout handling
        try:
            result = await self.wait_for_tool_result(request_id, timeout=timeout)
        except asyncio.TimeoutError:
            timeout_msg = (
                f"[MCP Tool Timeout] {tool_name} request timed out after {timeout}s "
                f"waiting for user approval. You can retry if needed."
            )
            self.context_db.save_message(self.name, "tool_result", timeout_msg)
            # Also add to conversation context so agent sees it in next API call
            self.conversation_context.append({
                "role": "user",
                "content": f"[SYSTEM] {timeout_msg}"
            })
            raise Exception(timeout_msg)

        if not result["success"]:
            self.context_db.save_message(
                self.name, "tool_result",
                f"[MCP Tool Denied] {tool_name}: {result.get('error')}"
            )
            raise Exception(f"Tool execution denied: {result.get('error')}")

        # If approved, execute via MCP
        try:
            mcp_result = await mcp_client.execute_tool(tool_name, args)
            result_summary = str(mcp_result)[:500]
            self.context_db.save_message(
                self.name, "tool_result",
                f"[MCP Tool Success] {tool_name}: {result_summary}"
            )
            return mcp_result
        except Exception as e:
            self.context_db.save_message(
                self.name, "tool_result",
                f"[MCP Tool Error] {tool_name}: {str(e)}"
            )
            raise
    
    def get_agent_tool_config(self, tool_name: str) -> Dict[str, Any]:
        """Get tool configuration with agent-specific overrides applied."""
        # Get base config
        base_config = self.tool_configs.get(tool_name, {})
        
        # Apply agent overrides if they exist
        if (self.name in self.agent_tool_overrides and 
            tool_name in self.agent_tool_overrides[self.name]):
            override_config = self.agent_tool_overrides[self.name][tool_name]
            # Merge configurations (override takes precedence)
            return self._merge_tool_configs(base_config, override_config)
        
        # Convert config object to dictionary if needed
        return self._config_to_dict(base_config)
    
    def _merge_tool_configs(self, base_config: Any, override_config: Any) -> Dict[str, Any]:
        """Merge tool configurations with override taking precedence."""
        # Convert config objects to dictionaries for merging
        base_dict = {}
        override_dict = {}
        
        # Extract attributes from base config
        if hasattr(base_config, '__dict__'):
            for key, value in base_config.__dict__.items():
                if not key.startswith('_'):
                    base_dict[key] = value
        elif isinstance(base_config, dict):
            base_dict = base_config.copy()
        
        # Extract attributes from override config
        if hasattr(override_config, '__dict__'):
            for key, value in override_config.__dict__.items():
                if not key.startswith('_') and value is not None:
                    override_dict[key] = value
        elif isinstance(override_config, dict):
            override_dict = {k: v for k, v in override_config.items() if v is not None}
        
        # Merge dictionaries
        merged = base_dict.copy()
        merged.update(override_dict)
        return merged
    
    def _config_to_dict(self, config: Any) -> Dict[str, Any]:
        """Convert config object to dictionary."""
        if isinstance(config, dict):
            return config.copy()
        elif hasattr(config, '__dict__'):
            return {k: v for k, v in config.__dict__.items() if not k.startswith('_')}
        else:
            return {}

    async def initialize_tools(self):
        """Initialize available tools for this agent."""
        self.logger.info(f"Initializing tools: {self.available_tools}")
        
        # Initialize MCP clients for external tools
        for tool_name in self.available_tools:
            if tool_name not in self.built_in_tools and tool_name in self.mcp_tools:
                try:
                    # TODO: Initialize MCP client
                    self.logger.info(f"MCP tool {tool_name} initialized")
                except Exception as e:
                    self.logger.error(f"Failed to initialize MCP tool {tool_name}: {e}")
                    await self.send_system_log(LogLevel.ERROR, f"MCP tool {tool_name} initialization failed: {e}")
    
    def _trim_context(self):
        """Trim conversation context to stay within limits."""
        if len(self.conversation_context) > self.max_context_messages:
            # Keep the most recent messages
            messages_to_remove = len(self.conversation_context) - self.max_context_messages
            self.conversation_context = self.conversation_context[messages_to_remove:]
            self.logger.debug(f"Agent {self.name} trimmed context: removed {messages_to_remove} old messages")

    def _inject_error_feedback(self, error_type: str, error_msg: str):
        """
        Inject error feedback into conversation context so agent can learn and adapt.

        This allows agents to be agentic - they can see what went wrong and take
        corrective action (e.g., summarize context if context_length_exceeded).
        """
        # For context length errors, aggressively trim FIRST before adding feedback
        is_context_error = "context_length" in error_type.lower() or "context_length" in error_msg.lower() or "token" in error_msg.lower()

        if is_context_error:
            # Keep only the last few messages to make room
            original_len = len(self.conversation_context)
            if original_len > 5:
                self.conversation_context = self.conversation_context[-5:]
                self.logger.warning(f"Context length error - trimmed from {original_len} to {len(self.conversation_context)} messages")

        feedback = f"[SYSTEM ERROR - {error_type}] {error_msg}"

        # For context length errors, add helpful guidance
        if is_context_error:
            feedback += "\n\nYour context exceeded the model's limit and was automatically trimmed. Your previous conversation history has been reduced. Please continue with the current task or ask to /clear context if needed."

        self.conversation_context.append({
            "role": "user",  # Use 'user' role so the model sees it as input to respond to
            "content": feedback
        })
        self.logger.info(f"Injected error feedback into context: {error_type}")
    
    async def shutdown(self):
        """Graceful shutdown of agent."""
        self.logger.info("Shutting down agent...")
        self.running = False
        
        # Cancel any pending response timer
        if self.response_timer and not self.response_timer.done():
            self.response_timer.cancel()
            try:
                await self.response_timer
            except asyncio.CancelledError:
                pass
        
        # Close orchestrator connection
        if self.writer:
            try:
                self.writer.close()
                await self.writer.wait_closed()
            except Exception as e:
                self.logger.error(f"Error closing connection: {e}")
        
        # Shutdown MCP clients
        for tool_name, client in self.mcp_clients.items():
            try:
                if hasattr(client, 'stop'):
                    await client.stop()
            except Exception as e:
                self.logger.error(f"Error shutting down MCP client {tool_name}: {e}")
        
        self.logger.info("Agent shutdown complete")
    
    # Abstract methods to be implemented by subclasses
    
    @abstractmethod
    async def initialize_api_client(self):
        """Initialize the API client for this provider."""
        pass
    
    @abstractmethod
    async def generate_response(self, conversation: List[Dict[str, Any]]) -> Optional[str]:
        """
        Generate response using the AI provider.
        
        Args:
            conversation: List of messages in OpenAI format
            
        Returns:
            Generated response text or None if no response
        """
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Get the name of this AI provider."""
        pass


def create_agent_main():
    """Create main function for agent process."""
    
    def main():
        """Main entry point for agent process."""
        parser = argparse.ArgumentParser(description="NeuroDeck AI Agent")
        parser.add_argument("--name", required=True, help="Agent name")
        parser.add_argument("--host", required=True, help="Orchestrator host")
        parser.add_argument("--port", type=int, required=True, help="Orchestrator port")
        parser.add_argument("--token", required=True, help="Authentication token")
        parser.add_argument("--config", default="config/agents.ini", help="Configuration file")
        
        args = parser.parse_args()
        
        # This will be overridden by specific agent implementations
        raise NotImplementedError("This should be overridden by specific agent classes")
    
    return main


# Test agent for development
class TestAgent(BaseAgent):
    """Test agent implementation for development."""
    
    async def initialize_api_client(self):
        """Initialize test API client."""
        self.logger.info("Test agent - no real API client")
    
    async def generate_response(self, conversation: List[Dict[str, Any]]) -> Optional[str]:
        """Generate test response."""
        last_message = conversation[-1]["content"] if conversation else ""
        return f"Test agent {self.name} received: {last_message[:50]}..."
    
    def get_provider_name(self) -> str:
        """Get provider name."""
        return "test"


async def test_agent():
    """Test the base agent functionality."""
    agent = TestAgent(
        name="test-agent",
        host="localhost", 
        port=9999,
        token="test-token-123"
    )
    
    try:
        await agent.start()
    except KeyboardInterrupt:
        await agent.shutdown()


if __name__ == "__main__":
    asyncio.run(test_agent())