"""
NeuroDeck Textual-based TUI - main interface.
"""

import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List

from textual.app import App, ComposeResult
from textual.containers import Vertical, Horizontal
from textual.widgets import Input, TextArea, RichLog, Static, Footer
from textual.message import Message
from textual.reactive import reactive
from textual.events import Key

from .client import NeuroDeckClient
from ..common.logging import get_logger
from ..common.protocol import (
    chat_history_request_message, tool_approval_message, 
    MessageType, ApprovalLevel
)
from ..common.config import ConfigManager

logger = get_logger(__name__)


class NeuroDeckTUI(App):
    """NeuroDeck Terminal User Interface using Textual."""
    
    CSS = """
    #header {
        dock: top;
        height: 3;
        background: $primary;
        color: $text;
        content-align: center middle;
    }
    
    #main_container {
        height: 1fr;
    }
    
    #chat_container {
        width: 2fr;
    }
    
    #chat_log {
        border: solid $primary;
        height: 2fr;
        margin: 1;
        padding: 1;
    }
    
    #debug_log {
        border: solid $secondary;
        height: 1fr;
        margin: 1;
        padding: 1;
    }
    
    #sidebar {
        dock: right;
        width: 25;
        border: solid $secondary;
        margin: 1;
        padding: 1;
    }
    
    /* DOS-STYLE INPUT HEADER STYLING:
     * Based on Textual's Collapsible and TabbedContent header patterns.
     * Uses disabled text color and dim style to clearly indicate it's instructional,
     * not content. Center alignment provides clean, professional appearance.
     * dock: top ensures it stays at the top of the container
     */
    .input-header {
        dock: top;
        height: 1;
        text-align: center;
        color: $text-disabled;
        text-style: dim;
        padding: 0 1;
        background: $surface;
    }
    
    /* INPUT CONTAINER STYLING:
     * Height 8 = 1 for header + ~6 for TextArea content + spacing
     * No padding needed - docked header and TextArea handle their own spacing
     * Border matches other panels (chat_log, debug_log, sidebar)
     */
    #input_container {
        height: 8;
        margin: 1;
        border: solid $secondary;
        padding: 0;
    }
    
    /* TEXTAREA STYLING:
     * No border (container provides it), fills remaining space after header
     * Margin provides internal spacing from container borders
     * This prevents double-border effect and maintains clean appearance
     */
    #message_input {
        width: 1fr;
        height: 1fr;
        margin: 1;
        border: none;
    }
    
    .agent-claudius { color: $primary; }
    .agent-grok { color: $error; }
    .agent-chatgpt { color: $success; }
    .agent-kimi { color: $warning; }
    .agent-user { color: $text; }
    .system-log { color: $secondary; }
    .debug-log { color: $secondary; }
    """
    
    TITLE = "NeuroDeck - Multi-Agent AI Console"
    
    def __init__(self):
        super().__init__()
        self.client: Optional[NeuroDeckClient] = None
        self.pending_approvals: List[Dict[str, Any]] = []  # Track multiple pending approvals
        self.tts_service: Optional[TTSService] = None
        
        # Load orchestrator configuration
        try:
            config_manager = ConfigManager("config/agents.ini")
            self.orchestrator_config = config_manager.load_orchestrator_config()
            # Load agent configs and initialize status
            self.agent_configs = config_manager.load_agent_configs()
            self.agents_status = {name: "●" for name in self.agent_configs.keys()}
            
            # Store TTS configuration for lazy loading (don't create TTS service yet)
            self.agent_voices = {name: config.tts_voice for name, config in self.agent_configs.items() if config.tts_voice}
                
        except Exception as e:
            logger.error(f"Failed to load agent configs: {e}")
            # Fallback to defaults if config loading fails
            from ..common.config import OrchestratorConfig
            self.orchestrator_config = OrchestratorConfig(
                host="localhost", port=9999, tls_cert="", tls_key="", 
                auth_token="", log_level="INFO", username="human", chat_history_limit=20,
                tts_enabled=True
            )
            self.agent_configs = {}
            self.agent_voices = {}
            self.agents_status = {
                "claudius": "●",
                "grok": "●", 
                "chatgpt": "●",
                "kimi": "●"
            }
        # Track auto-scroll preference (simpler approach)
        self.auto_scroll_enabled = True
        
        # Status polling
        self.status_poll_task: Optional[asyncio.Task] = None
        
    
    def compose(self) -> ComposeResult:
        """Compose the UI layout."""
        yield Static("🤖 NeuroDeck - Multi-Agent AI Console", id="header")
        
        with Horizontal(id="main_container"):
            with Vertical(id="chat_container"):
                yield Static("💬 Chat", classes="panel-title")
                yield RichLog(id="chat_log", highlight=True, markup=True, wrap=True, auto_scroll=True)
                yield Static("🔧 Debug Logs", classes="panel-title", id="debug_header")
                yield RichLog(id="debug_log", highlight=True, markup=True, wrap=True)
                
                # DOS-STYLE INPUT PATTERN:
                # This follows Textual's sanctioned approach for titled/labeled inputs,
                # as seen in Collapsible, TabbedContent, and other built-in widgets.
                # 
                # PATTERN: Vertical container with Static header + content widget
                # - Static widget provides permanent instructional text (never disappears)
                # - Vertical layout stacks header above input cleanly
                # - No complex state management or event handling needed
                # - Follows same pattern as Textual's Collapsible widget composition
                #
                # WHY NOT PLACEHOLDER TEXT:
                # - TextArea doesn't support placeholder parameter (only Input does)
                # - Attempting to manage placeholder via content + events causes:
                #   * Character eating (events fire after text is already modified)
                #   * Complex state tracking and reactive cycles
                #   * Interference with TextArea's internal text handling
                #
                # WHY THIS APPROACH IS BETTER:
                # - Zero interference with TextArea functionality
                # - Always visible instructions (better UX than disappearing placeholder)
                # - Professional DOS/terminal aesthetic
                # - Matches existing UI patterns (Chat/Debug panels have headers)
                # - Simple, maintainable code with no edge cases
                #
                # TEXTUAL BEST PRACTICE:
                # Use Static widgets for labels/headers, let content widgets handle content
                with Vertical(id="input_container"):
                    yield Static("Type message, Ctrl+Enter to send, Ctrl+C to clear", classes="input-header")
                    yield TextArea(id="message_input")
            
            with Vertical(id="sidebar"):
                yield Static("🤖 Agents", classes="sidebar-title")
                yield Static(id="agents_list")
                yield Static("\n🔊 TTS Status:", classes="sidebar-title")
                yield Static(id="tts_status")
                yield Static("\n💡 Commands:", classes="sidebar-title") 
                yield Static("@agent message\n/help\n/clear\n/wipe [agent]\n/yolo\n/tts\n/tts clear\n/tts params\n/debug\n/autoscroll\n/quit", id="commands_help")
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Initialize the app when mounted."""
        self.update_agents_display()
        self.update_tts_display()
        
        # Focus on input
        self.query_one("#message_input").focus()
        
        # Start connection to orchestrator (TTS will be initialized after connection)
        asyncio.create_task(self.connect_to_orchestrator())
    
    
    async def initialize_tts_service(self):
        """Conditionally create and initialize TTS service based on configuration."""
        try:
            # Check if TTS is enabled in configuration  
            if not self.orchestrator_config.tts_enabled:
                self.log_debug("system", "TTS disabled in configuration", "system-info")
                self.update_tts_display()  # Update display to show disabled state
                return
            
            # Check if any agents have TTS voices configured
            if not self.agent_voices:
                self.log_debug("system", "No TTS voices configured - TTS disabled", "system-info")
                self.update_tts_display()  # Update display to show no voices
                return
            
            # Import TTS service here to avoid import delays during UI startup
            from .tts_service import TTSService
            
            # Create TTS service (now that UI is ready)
            self.tts_service = TTSService(self.agent_voices, self.agent_configs, log_callback=self.log_debug)
            
            # Show loading message and start TTS initialization
            self.log_message("system", "⏳ Loading TTS models...", "info")
            self.update_tts_display()  # Show loading state
            
            # Start TTS service and status updater concurrently
            await asyncio.gather(
                self.start_tts_service(),
                self.tts_status_updater(),
                return_exceptions=True
            )
            
        except Exception as e:
            self.log_debug("system", f"Failed to initialize TTS service: {e}", "system-error")
            self.update_tts_display()  # Update display to show error state
    
    
    async def delayed_tts_initialization(self):
        """Initialize TTS service after a brief delay to ensure UI is ready."""
        try:
            # Brief yield to let UI render first
            await asyncio.sleep(0.1)

            # Now initialize TTS service
            await self.initialize_tts_service()
            
        except Exception as e:
            self.log_debug("system", f"Failed to create delayed TTS task: {e}", "system-error")
    
    
    async def connect_to_orchestrator(self):
        """Connect to the orchestrator daemon."""
        self.client = NeuroDeckClient(log_callback=self.log_debug)
        
        # Set message handler
        self.client.set_message_handler(self.handle_message)
        
        # Show connecting status
        self.log_debug("system", "Connecting to orchestrator...", "info")
        
        try:
            # Connect and authenticate
            if not await self.client.connect():
                self.log_debug("system", "❌ Failed to connect to orchestrator", "error")
                return
            
            if not await self.client.authenticate():
                self.log_debug("system", "❌ Authentication failed", "error")
                return
            
            self.log_debug("system", "Connected to NeuroDeck orchestrator", "success")
            
            # Request recent chat history
            await self.request_chat_history()
            
            # Start status polling
            self.status_poll_task = asyncio.create_task(self.status_poll_loop())
            
            # Initialize TTS service as the very last thing after everything is fully ready
            asyncio.create_task(self.delayed_tts_initialization())
            
            # Start message loop
            await self.client.message_loop()
            
        except Exception as e:
            self.log_debug("system", f"❌ Connection error: {e}", "error")
    
    async def handle_message(self, message: Dict[str, Any]):
        """Handle incoming messages from orchestrator."""
        msg_type = message.get("type", "unknown")
        
        if msg_type == "chat":
            sender = message.get("sender", "unknown")
            content = message.get("content", "")
            timestamp = message.get("timestamp", "")
            
            self.log_message(sender, content, "chat", timestamp)
            
            # Queue TTS for AI agent messages (only if TTS is ready)
            if (self.tts_service and 
                self.tts_service.is_enabled() and 
                self.tts_service.is_ready() and
                sender != "user" and 
                sender in self.tts_service.agent_voices):
                try:
                    asyncio.create_task(self.tts_service.queue_message(sender, content))
                except Exception as e:
                    self.log_debug("system", f"TTS queue error: {e}", "system-warning")
            
        elif msg_type == "system_log":
            sender = message.get("sender", "system")
            level = message.get("level", "info")
            log_message = message.get("message", "")
            
            # Send system logs to debug window instead of chat
            self.log_debug(f"{sender}", log_message, f"system-{level}")
            
        elif msg_type == "chat_history_response":
            messages = message.get("messages", [])
            self.load_chat_history(messages)
            
        elif msg_type == MessageType.TOOL_APPROVAL_REQUEST.value:
            await self.handle_tool_approval_request(message)
            
        elif msg_type == MessageType.YOLO_STATUS.value:
            # Handle YOLO mode status broadcast from orchestrator
            enabled = message.get("enabled", False)
            status_msg = message.get("message", "")
            icon = "🔥" if enabled else "🟢"
            self.log_message("system", f"{icon} {status_msg}", "info")
            
        elif msg_type == "auth_result":
            success = message.get("success", False)
            msg = message.get("message", "")
            if success:
                self.log_message("system", f"✅ {msg}", "success")
            else:
                self.log_message("system", f"❌ {msg}", "error")
                
        elif msg_type == "command_result":
            # Handle command results (like list_agents for status polling)
            cmd = message.get("cmd")
            success = message.get("success", False)
            
            if cmd == "list_agents" and success:
                # Update agent status based on response
                agent_list = message.get("data", [])
                self.update_agent_status_from_list(agent_list)
            elif cmd == "wipe":
                # Handle wipe command results with debug logging
                await self.handle_wipe_result(message)
            elif cmd == "yolo":
                # Handle YOLO command results
                await self.handle_yolo_result(message)
    
    def log_message(self, sender: str, content: str, msg_type: str = "chat", timestamp: str = ""):
        """Add message to chat log with proper styling."""
        chat_log = self.query_one("#chat_log", RichLog)
        
        # Format timestamp
        if not timestamp:
            timestamp = datetime.now().strftime("%H:%M:%S")
        else:
            # Parse ISO timestamp to short format
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                timestamp = dt.strftime("%H:%M:%S")
            except:
                timestamp = timestamp[:8]  # Fallback
        
        # Style based on message type and sender
        if msg_type == "chat":
            if sender == "user":
                style_class = "agent-user"
                prefix = "👤"
                # Use configured username instead of "user"
                display_name = self.orchestrator_config.username
            else:
                style_class = f"agent-{sender.lower()}"
                prefix = "🤖"
                display_name = sender
            
            chat_log.write(f"[dim]{timestamp}[/] [{style_class}]{prefix} {display_name}[/]: {content}", scroll_end=self.auto_scroll_enabled)
            # Full-width light horizontal divider (minus 2 chars for padding)
            divider_width = max(1, (chat_log.size.width - 2) if chat_log.size.width > 2 else 78)
            chat_log.write("[dim]" + "─" * divider_width + "[/dim]", scroll_end=self.auto_scroll_enabled)
            
        elif msg_type.startswith("system"):
            level = msg_type.split("-")[-1] if "-" in msg_type else "info"
            
            if level == "error":
                prefix = "❌"
                style = "bold red"
            elif level == "warning":
                prefix = "⚠️"
                style = "bold yellow"
            elif level == "success":
                prefix = "✅"
                style = "bold green"
            else:
                prefix = "ℹ️"
                style = "dim blue"
            
            chat_log.write(f"[dim]{timestamp}[/] [{style}]{prefix} {sender}[/]: {content}", scroll_end=self.auto_scroll_enabled)
            # Full-width light horizontal divider (minus 2 chars for padding)
            divider_width = max(1, (chat_log.size.width - 2) if chat_log.size.width > 2 else 78)
            chat_log.write("[dim]" + "─" * divider_width + "[/dim]", scroll_end=self.auto_scroll_enabled)
        
        else:
            # Default formatting
            chat_log.write(f"[dim]{timestamp}[/] {sender}: {content}", scroll_end=self.auto_scroll_enabled)
            # Full-width light horizontal divider (minus 2 chars for padding)
            divider_width = max(1, (chat_log.size.width - 2) if chat_log.size.width > 2 else 78)
            chat_log.write("[dim]" + "─" * divider_width + "[/dim]", scroll_end=self.auto_scroll_enabled)
    
    def log_debug(self, sender: str, content: str, msg_type: str = "debug", timestamp: str = ""):
        """Add debug message to debug log with proper styling."""
        debug_log = self.query_one("#debug_log", RichLog)
        
        # Format timestamp
        if not timestamp:
            timestamp = datetime.now().strftime("%H:%M:%S")
        else:
            # Parse ISO timestamp to short format
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                timestamp = dt.strftime("%H:%M:%S")
            except:
                timestamp = timestamp[:8]  # Fallback
        
        # Style based on message type
        if msg_type.startswith("system"):
            level = msg_type.split("-")[-1] if "-" in msg_type else "info"
            
            if level == "error":
                prefix = "ERR"
                style = "bold red"
            elif level == "warning":
                prefix = "WARN"
                style = "bold yellow"
            elif level == "success":
                prefix = "OK"
                style = "bold green"
            else:
                prefix = "DBG"
                style = "dim blue"
            
            debug_log.write(f"[dim]{timestamp}[/] [{style}]{prefix} {sender}[/]: {content}", scroll_end=self.auto_scroll_enabled)
        else:
            # Default debug formatting
            debug_log.write(f"[dim]{timestamp} {sender}: {content}[/]", scroll_end=self.auto_scroll_enabled)
    
    async def status_poll_loop(self):
        """Periodically poll for agent status updates."""
        while self.client and self.client.authenticated:
            try:
                await asyncio.sleep(10.0)  # Poll every 10 seconds
                
                if self.client and self.client.authenticated:
                    # Send list_agents command to get current status
                    command_msg = {
                        "type": "command",
                        "cmd": "list_agents"
                    }
                    await self.client.send_message(command_msg)
                    
            except Exception as e:
                self.log_debug("system", f"Status polling error: {e}", "error")
                await asyncio.sleep(10.0)  # Continue polling even on error
    
    def update_agent_status_from_list(self, agent_list):
        """Update agent status based on list_agents command result."""
        # Create a set of connected agents
        connected_agents = set()
        for agent_info in agent_list:
            agent_name = agent_info.get("name", "")
            connected = agent_info.get("connected", False)
            if connected:
                connected_agents.add(agent_name)
        
        # Update status for all known agents
        for agent_name in self.agents_status:
            if agent_name in connected_agents:
                self.agents_status[agent_name] = "●"  # Connected
            else:
                self.agents_status[agent_name] = "○"  # Disconnected
        
        # Refresh the display
        self.update_agents_display()
    
    def update_agents_display(self):
        """Update the agents status display."""
        agents_list = self.query_one("#agents_list", Static)
        
        status_text = ""
        for agent, status in self.agents_status.items():
            color = "green" if status == "●" else "red"
            status_text += f"[{color}]{status}[/] {agent.title()}\n"
        
        agents_list.update(status_text)
    
    async def start_tts_service(self):
        """Start TTS service asynchronously."""
        if not self.tts_service:
            return
            
        try:
            # Show loading message immediately
            self.update_tts_display()
            self.log_debug("system", "Starting TTS service (loading VCTK model...)", "info")
            
            success = await self.tts_service.start_service()
            if success:
                voice_count = len(self.agent_voices)
                self.log_debug("system", f"✅ TTS ready ({voice_count} voices loaded)", "success")
                self.log_message("system", "🎙️ TTS models loaded successfully", "info")
            else:
                self.log_debug("system", "❌ Failed to start TTS service", "error")
            self.update_tts_display()
        except Exception as e:
            self.log_debug("system", f"❌ Error starting TTS service: {e}", "error")
            self.update_tts_display()
    
    def update_tts_display(self):
        """Update the TTS status display."""
        tts_status = self.query_one("#tts_status", Static)
        
        # Check if TTS is disabled in configuration
        if not self.orchestrator_config.tts_enabled:
            tts_status.update("[dim]TTS Disabled[/]")
            return
        
        # Check if no voices are configured
        if not self.agent_voices:
            tts_status.update("[dim]No TTS Voices[/]")
            return
        
        # TTS service not created yet (still loading)
        if not self.tts_service:
            tts_status.update("[yellow]⏳ Starting TTS...[/]")
            return
        
        # Check if service is initializing
        if not self.tts_service.is_ready():
            if hasattr(self.tts_service, 'tts_model') and self.tts_service.tts_model is None:
                tts_status.update("[yellow]⏳ Loading Models...[/]")
                return
            else:
                tts_status.update("[red]❌ TTS Failed[/]")
                return
        
        if self.tts_service.is_enabled():
            icon = "🔊"
            status = "Enabled"
            color = "green"
        else:
            icon = "🔇"
            status = "Disabled"
            color = "red"
        
        queue_size = self.tts_service.get_queue_size()
        if queue_size > 0:
            status_text = f"[{color}]{icon} {status}[/]\n[dim]Queue: {queue_size}[/]"
        else:
            status_text = f"[{color}]{icon} {status}[/]"
        
        tts_status.update(status_text)
    
    async def tts_status_updater(self):
        """Periodically update TTS status display while loading."""
        if not self.tts_service:
            return
            
        max_wait_time = 300  # Maximum 300 seconds (5 minutes) wait time for model downloads
        elapsed_time = 0
        update_interval = 2  # Update every 2 seconds
        
        while not self.tts_service.is_ready() and elapsed_time < max_wait_time:
            await asyncio.sleep(update_interval)
            elapsed_time += update_interval
            self.update_tts_display()
            
        # Final update when ready or timed out
        self.update_tts_display()
        
        if elapsed_time >= max_wait_time and not self.tts_service.is_ready():
            self.log_debug("system", "⚠️ TTS initialization timed out", "system-warning")
    
    async def on_key(self, event: Key) -> None:
        """Handle key events for TextArea submission and clearing."""
        textarea = self.query_one("#message_input", TextArea)
        
        # Check if we're focused on the message input
        if textarea.has_focus:
            if event.key == "ctrl+enter":
                # Send message
                await self.handle_message_submission()
            elif event.key == "ctrl+c":
                # Clear text area (Ctrl+C when focused on input)
                textarea.clear()
                event.prevent_default()  # Prevent default Ctrl+C behavior
    
    async def handle_message_submission(self) -> None:
        """Handle message input submission.
        
        CLEAN MESSAGE HANDLING PATTERN:
        After removing complex placeholder logic, this is now simple and reliable:
        - Check connection status
        - Get trimmed text content 
        - Skip empty messages
        - Clear input after successful extraction
        - No placeholder state management needed
        
        The TextArea starts empty and stays clean - header provides all user guidance.
        """
        if not self.client or not self.client.authenticated:
            self.log_message("system", "❌ Not connected to orchestrator", "error")
            return
        
        textarea = self.query_one("#message_input", TextArea)
        message = textarea.text.strip()
        if not message:
            return
        
        # Clear input immediately after extracting message
        # No need to restore placeholder - header provides permanent guidance
        textarea.clear()
        
        # Handle commands
        if message.startswith("/"):
            await self.handle_command(message)
            return
        
        try:
            # Send chat message
            await self.client.send_chat(message)
            
            # Log user message locally (will also get echo from server)
            self.log_message("user", message, "chat")
            
        except Exception as e:
            self.log_message("system", f"❌ Failed to send message: {e}", "error")
    
    async def handle_command(self, command: str):
        """Handle UI commands."""
        if command == "/help":
            help_text = """
🤖 NeuroDeck Commands:
  @agent message    - Direct message to specific agent
  Ctrl+Enter       - Send multi-line message
  /help            - Show this help
  /clear           - Clear chat history
  /wipe [agent]    - Wipe context (all agents or specific agent)
  /agents          - Show agent status
  /tts             - Toggle text-to-speech
  /tts clear       - Clear TTS queue
  /tts params      - Show TTS parameters
  /debug           - Toggle debug panel
  /autoscroll      - Toggle auto-scroll behavior
  /approve         - Approve pending tool request
  /deny            - Deny pending tool request
  /quit or /exit   - Quit NeuroDeck
            """
            self.log_message("system", help_text, "info")
            
        elif command == "/clear":
            chat_log = self.query_one("#chat_log", RichLog)
            chat_log.clear()
            self.log_message("system", "Chat history cleared", "info")
            
        elif command == "/agents":
            status_info = "Agent Status:\n"
            for agent, status in self.agents_status.items():
                state = "Connected" if status == "●" else "Disconnected"
                status_info += f"  {agent.title()}: {state}\n"
            
            self.log_message("system", status_info, "info")
            
        elif command == "/debug":
            debug_log = self.query_one("#debug_log", RichLog)
            debug_header = self.query_one("#debug_header", Static)
            if debug_log.display:
                debug_log.display = False
                debug_header.display = False
                self.log_message("system", "Debug panel hidden", "info")
            else:
                debug_log.display = True
                debug_header.display = True
                self.log_message("system", "Debug panel shown", "info")
                
        elif command == "/autoscroll":
            self.auto_scroll_enabled = not self.auto_scroll_enabled
            status = "enabled" if self.auto_scroll_enabled else "disabled"
            self.log_message("system", f"Auto-scroll {status}", "info")
        
        elif command.startswith("/tts"):
            # Check if TTS is disabled in configuration
            if not self.orchestrator_config.tts_enabled:
                self.log_message("system", "❌ TTS disabled in configuration", "error")
                return
                
            # Check if no voices configured
            if not self.agent_voices:
                self.log_message("system", "❌ TTS not available - no voices configured", "error")
                return
                
            # Check if TTS service exists
            if not self.tts_service:
                self.log_message("system", "⏳ TTS service not yet initialized... please wait", "warning")
                return
            
            # Check if TTS service is ready
            if not self.tts_service.is_ready():
                if hasattr(self.tts_service, 'tts_model') and self.tts_service.tts_model is None:
                    self.log_message("system", "⏳ TTS service is still loading VCTK model... please wait", "warning")
                else:
                    self.log_message("system", "❌ TTS service failed to initialize", "error")
                return
            
            # Parse TTS subcommands
            parts = command.split()
            if len(parts) == 1:  # Just "/tts" - toggle on/off
                if self.tts_service.is_enabled():
                    self.tts_service.disable_tts()
                    self.log_message("system", "🔇 TTS disabled", "info")
                else:
                    self.tts_service.enable_tts()
                    self.log_message("system", "🔊 TTS enabled", "info")
            
            elif len(parts) == 2 and parts[1] == "clear":  # "/tts clear"
                queue_size = self.tts_service.get_queue_size()
                if queue_size == 0:
                    self.log_message("system", "📭 TTS queue is already empty", "info")
                else:
                    cleared_count = await self.tts_service.clear_queue()
                    self.log_message("system", f"🗑️ Cleared {cleared_count} messages from TTS queue", "info")
            
            elif len(parts) == 2 and parts[1] == "params":  # "/tts params"
                # Show TTS parameters for all configured agents
                self.log_message("system", "🔧 TTS Parameters:", "info")
                for agent_name, agent_config in self.agent_configs.items():
                    if hasattr(agent_config, 'tts_voice') and agent_config.tts_voice:
                        voice = agent_config.tts_voice
                        length = getattr(agent_config, 'tts_length_scale', 1.0)
                        inference = getattr(agent_config, 'tts_inference_noise', 0.667)
                        duration = getattr(agent_config, 'tts_duration_noise', 0.8)
                        self.log_message("system", f"  {agent_name}: voice={voice}, length={length:.1f}, noise={inference:.2f}, duration={duration:.2f}", "info")
            
            else:
                self.log_message("system", "❓ Usage: /tts (toggle), /tts clear (empty queue), or /tts params", "error")
                return
            
            self.update_tts_display()
                
        elif command == "/approve":
            if self.pending_approvals:
                # Approve all pending requests
                count = len(self.pending_approvals)
                agents = set()
                for pending in self.pending_approvals:
                    approval_msg = tool_approval_message(
                        agent=pending["agent"],
                        approval=ApprovalLevel.ALL,
                        request_id=pending["request_id"]
                    )
                    await self.client.send_message(approval_msg)
                    agents.add(pending["agent"])

                agents_str = ", ".join(sorted(agents))
                self.log_message("system", f"✅ Approved {count} tool request(s) from: {agents_str}", "success")
                self.pending_approvals = []
            else:
                self.log_message("system", "No pending tool approvals", "error")

        elif command == "/deny":
            if self.pending_approvals:
                # Deny all pending requests
                count = len(self.pending_approvals)
                agents = set()
                for pending in self.pending_approvals:
                    denial_msg = tool_approval_message(
                        agent=pending["agent"],
                        approval=ApprovalLevel.DENY,
                        request_id=pending["request_id"]
                    )
                    await self.client.send_message(denial_msg)
                    agents.add(pending["agent"])

                agents_str = ", ".join(sorted(agents))
                self.log_message("system", f"❌ Denied {count} tool request(s) from: {agents_str}", "error")
                self.pending_approvals = []
            else:
                self.log_message("system", "No pending tool approvals", "error")
                
        elif command.startswith("/wipe"):
            # Handle database context wipe command
            await self.send_wipe_command(command)
            
        elif command.startswith("/yolo"):
            # Handle YOLO mode toggle command
            await self.send_yolo_command(command)
            
        elif command in ["/quit", "/exit", "/q"]:
            self.log_message("system", "👋 Goodbye! Disconnecting from NeuroDeck...", "info")
            # Give a moment for the message to display
            await asyncio.sleep(0.5)  
            # Properly exit the app
            self.exit()
            
        else:
            self.log_message("system", f"❌ Unknown command: {command}", "error")

    async def send_wipe_command(self, command: str):
        """Send wipe command to orchestrator with debug logging."""
        if not self.client or not self.client.authenticated:
            self.log_debug("system", "❌ Not connected to orchestrator for wipe command", "error")
            return
        
        try:
            # Parse command: "/wipe" or "/wipe agent_name"
            parts = command.strip().split()
            if len(parts) == 1:
                # Wipe all agents
                agent_name = None
                self.log_debug("system", "🗑️ Initiating wipe of ALL agent context...", "info")
            else:
                # Wipe specific agent
                agent_name = parts[1]
                self.log_debug("system", f"🗑️ Initiating wipe of '{agent_name}' context...", "info")
            
            # Send wipe command to server
            wipe_msg = {
                "type": "command",
                "cmd": "wipe"
            }
            if agent_name:
                wipe_msg["agent"] = agent_name
            
            await self.client.send_message(wipe_msg)
            self.log_debug("system", "📤 Wipe command sent to orchestrator", "info")
            
        except Exception as e:
            self.log_debug("system", f"❌ Failed to send wipe command: {e}", "error")

    async def send_yolo_command(self, command: str):
        """Send YOLO mode toggle command to orchestrator."""
        if not self.client or not self.client.authenticated:
            self.log_message("system", "❌ Not connected to orchestrator", "error")
            return
        
        try:
            # Parse command: "/yolo" or "/yolo confirm"
            parts = command.strip().split()
            confirm = len(parts) > 1 and parts[1] == "confirm"
            
            # Send YOLO command to server
            yolo_msg = {
                "type": "command",
                "cmd": "yolo",
                "confirm": confirm
            }
            
            await self.client.send_message(yolo_msg)
            self.log_debug("system", f"📤 YOLO command sent: confirm={confirm}", "info")
            
        except Exception as e:
            self.log_message("system", f"❌ Failed to send YOLO command: {e}", "error")

    async def handle_wipe_result(self, message: Dict[str, Any]):
        """Handle wipe command result with debug logging."""
        success = message.get("success", False)
        
        if success:
            # Success case
            result_message = message.get("message", "Wipe completed")
            deleted_count = message.get("deleted_count", 0)
            self.log_debug("system", f"✅ {result_message} ({deleted_count} messages deleted)", "success")
        else:
            # Error case
            error_message = message.get("error", "Wipe failed")
            self.log_debug("system", f"❌ Wipe failed: {error_message}", "error")

    async def handle_yolo_result(self, message: Dict[str, Any]):
        """Handle YOLO command result."""
        success = message.get("success", False)
        data = message.get("data", "")
        
        if success:
            # Success case - YOLO mode toggled
            self.log_message("system", f"🔥 {data}", "info")
        else:
            # Error case or confirmation required
            error_message = message.get("error", data)
            requires_confirmation = message.get("requires_confirmation", False)
            
            if requires_confirmation:
                self.log_message("system", f"⚠️ {error_message}", "warning")
            else:
                self.log_message("system", f"❌ YOLO command failed: {error_message}", "error")

    async def handle_tool_approval_request(self, message: Dict[str, Any]):
        """Handle tool approval request from orchestrator."""
        agent = message.get("agent")
        tool = message.get("tool")
        description = message.get("description")
        request_id = message.get("request_id")
        
        # Show approval dialog
        self.log_message("system", f"🔧 {agent} requests: {description}", "warning")
        self.log_message("system", "Type /approve or /deny to respond", "info")
        
        # Store pending approval (append to list for multiple requests)
        self.pending_approvals.append({
            "agent": agent,
            "tool": tool,
            "request_id": request_id,
            "description": description
        })
    
    async def request_chat_history(self):
        """Request recent chat history from orchestrator."""
        try:
            if self.client:
                history_request = chat_history_request_message(limit=self.orchestrator_config.chat_history_limit)
                await self.client.send_message(history_request)
                self.log_debug("system", f"Requesting recent chat history (limit: {self.orchestrator_config.chat_history_limit})", "info")
        except Exception as e:
            self.log_debug("system", f"Failed to request chat history: {e}", "error")
    
    def load_chat_history(self, messages: list):
        """Load chat history messages into the chat window."""
        try:
            chat_log = self.query_one("#chat_log", RichLog)
            
            if messages:
                # Add a separator to show where history starts (always scroll for history loading)
                chat_log.write("[dim]─── Chat History ───[/dim]", scroll_end=True)
                
                # Display each historical message (always scroll for history loading)
                for msg in messages:
                    sender = msg.get("sender", "unknown")
                    content = msg.get("content", "")
                    timestamp = msg.get("timestamp", "")
                    
                    # Temporarily force auto-scroll for loading history
                    old_auto_scroll = self.auto_scroll_enabled
                    self.auto_scroll_enabled = True
                    self.log_message(sender, content, "chat", timestamp)
                    self.auto_scroll_enabled = old_auto_scroll
                
                # Add another separator to show where new messages begin (always scroll)
                chat_log.write("[dim]─── Live Chat ───[/dim]", scroll_end=True)
                
                self.log_debug("system", f"Loaded {len(messages)} historical messages", "info")
            else:
                self.log_debug("system", "No chat history available", "info")
                
        except Exception as e:
            self.log_debug("system", f"Error loading chat history: {e}", "error")

    async def on_unmount(self) -> None:
        """Cleanup when app is closing."""
        # Cancel status polling task
        if self.status_poll_task and not self.status_poll_task.done():
            self.status_poll_task.cancel()
            try:
                await self.status_poll_task
            except asyncio.CancelledError:
                pass
        
        # Stop TTS service
        if self.tts_service:
            await self.tts_service.stop_service()
                
        if self.client:
            await self.client.disconnect()


def run_console():
    """Run the NeuroDeck console."""
    app = NeuroDeckTUI()
    app.run()