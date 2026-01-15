# NeuroDeck Specification

## Overview

NeuroDeck is a Python-based console application that orchestrates multiple AI agents through a process-based architecture. Similar to Claude Code, it provides a terminal interface for seamless collaboration between AI models and humans, where each AI runs in its own process with dedicated tools and capabilities.

## Vision

To create a powerful console-based AI collaboration platform where multiple AI processes work together under a central orchestrator, each maintaining their own context and tools while enabling sophisticated inter-agent communication.

## Core Features

### 1. Multi-Agent Support
- **Supported Models:**
  - Anthropic Claude (Opus 4, Sonnet 3.5)
  - xAI Grok (Grok-4, Grok-3)
  - OpenAI ChatGPT (GPT-4o, GPT-4o-mini)
  - GroqAI Models (Kimi K2, Llama 3.3, etc.)

### 2. Agent Personas
- Custom names for each agent instance
- Configurable personality traits and expertise areas
- Persistent persona memory across sessions
- Role-based capabilities (e.g., researcher, coder, analyst)

### 3. Inter-Agent Communication
- Agents can directly address and respond to each other
- Collaborative problem-solving through agent discussions
- Automatic context sharing between agents
- Conflict resolution mechanisms for disagreements

### 4. Unified API Interface
- OpenAI-compatible endpoint support
- Standardized request/response format
- Provider-agnostic implementation
- Automatic API translation layer

## Architecture

### Process-Based Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Console UI Process                        │
│               (Python Terminal Interface)                    │
│                  - Dynamic text input                        │
│                  - Message rendering                         │
│                  - Command parsing                           │
└─────────────────────────────┬───────────────────────────────┘
                              │ TLS Socket
┌─────────────────────────────┴───────────────────────────────┐
│              Orchestrator Daemon Process                     │
│                  (TCP Socket Server)                         │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐    │
│  │Message Queue│  │Process Mgmt  │  │ Agent Registry  │    │
│  └─────────────┘  └──────────────┘  └─────────────────┘    │
└─────────┬──────────────┬──────────────┬────────────────────┘
          │              │              │
     ┌────┴─────┐   ┌────┴─────┐   ┌───┴──────┐
     │ Claude   │   │  Grok    │   │ ChatGPT  │  ...
     │ Process  │   │ Process  │   │ Process  │
     │          │   │          │   │          │
     │ - API    │   │ - API    │   │ - API    │
     │ - Tools  │   │ - Tools  │   │ - Tools  │
     │ - MCP    │   │ - MCP    │   │ - MCP    │
     │ - Context│   │ - Context│   │ - Context│
     └──────────┘   └──────────┘   └──────────┘
```

### Process Communication Protocol

Each AI agent runs as a separate Python process with:
- Dedicated context management (persisted in SQLite3)
- Tool access (filesystem, MCP servers, etc.)
- API client for its respective provider
- **Same line-based JSON protocol** for communication with orchestrator
- Agents connect back to orchestrator via TCP socket using same protocol as TUI client

### Core Components

1. **Orchestrator Daemon**
	- TCP socket server with TLS
	- Process lifecycle management
	- Message routing between agents
	- Central command processing
	- Connection management
		
2. **Agent Processes**
	- Independent Python processes spawned by orchestrator
	- Context persisted in SQLite3 database (survives crashes/restarts)
	- Tool execution with user approval system (once/same/all/YOLO modes)
	- API communication with providers
	- Receive all broadcast messages via same JSON protocol
	- System prompt includes their name/persona
	- Working directory = orchestrator's CWD

3. **Console UI Client**
	- Modern TUI built with Textual framework
	- Claude Code-style interface with dynamic input
	- Rich text rendering with syntax highlighting
	- Real-time message updates
	- TLS socket client to orchestrator

4. **IPC Message Protocol**
	- Line-based JSON messages (one per line)
	- Simple newline-delimited format
	- Human-readable and debuggable
	- Extensible message types

## API Integration Strategy

### Unified API Format

All providers will be accessed through a standardized interface:

```python
class UnifiedAPIRequest:
    model: str              # Provider-specific model identifier
    messages: List[Message] # Conversation history
    temperature: float      # Creativity parameter
    max_tokens: int        # Response length limit
    stream: bool           # Streaming response flag
    
class Message:
    role: str              # "user", "assistant", "system"
    content: str           # Message content
    name: Optional[str]    # Agent name for multi-agent context
    metadata: Dict         # Provider-specific metadata
```

### Provider Adapters

Each AI provider will have a dedicated adapter that:
- Translates unified requests to provider-specific format
- Handles authentication and API keys
- Manages provider-specific features
- Normalizes responses

### OpenAI-Compatible Endpoint

Primary integration method using `/v1/chat/completions` endpoint:

```bash
POST /v1/chat/completions
{
  "model": "provider:model-name",
  "messages": [...],
  "temperature": 0.7,
  "stream": true
}
```

## Agent Communication Protocol

### Chat-Based Broadcasting

Every message is broadcast to all agents - it's just like a group chat where everyone sees everything.

```
User: "Hey @grok, can you analyze this code?"
Grok: "Sure, I'll analyze it for security issues..."
Claudius: "I notice there's also a performance issue in that code..."
User: "Thanks both! Claudius, tell me more"
Claudius: "The nested loops here could be optimized..."
```

### Message Cascade Pattern

Each user message triggers a response cycle where agents decide to participate:

```
User: "Hey everyone, who's there?"
Claude: "Claude here, ready to help!"
Grok: "Grok online, security analysis ready!"
Claudius: "Claudius reporting for duty!"

User: "Can someone help me debug this function?"
Claude: "I can help analyze the logic flow..."
Grok: "I'll check for potential vulnerabilities..."
```

### Agent Awareness

Each agent knows its name through its system prompt:
- "You are Grok, a security-focused AI assistant..."
- "You are Claudius, an AI assistant named by the user..."
- Agents recognize when they're being addressed directly
- Natural language mentions work: "@grok", "Hey Claudius", "Claude, can you..."

### Response Behavior

The message → response cycle works like a natural chat:
1. **Direct mentions** - Always respond when named ("@grok", "Hey Claude")
2. **General questions** - All agents evaluate if they should respond ("Can someone help?", "Who's online?")
3. **Natural flow** - No response ordering or timeouts; agents respond when they have something to contribute
4. **Multiple responses** - Multiple agents can respond to the same message naturally
5. **System prompt guidance** - Configure response patterns:
   - "Always respond to roll calls and direct mentions"
   - "Only speak when you have something unique to add"
   - "Be proactive and helpful in conversations"

### Message Format

All messages include the sender's name naturally:
```
[Grok]: I found a security issue...
[Claudius]: Building on what Grok said...
[User]: Thanks everyone!
```

## Console UI Design

### Textual-Based Terminal Interface

1. **Modern TUI Layout**
	- Full-screen Textual application
	- Docked input area with auto-resizing
	- Scrollable chat history panel
	- Agent status sidebar
	- CSS-styled components

2. **Input Widget Features**
	- Rich text input with markdown support
	- Multi-line editing (Shift+Enter for newline)
	- Command history and auto-completion
	- Real-time typing indicators

3. **Display Capabilities**
	- Syntax-highlighted code blocks
	- Markdown rendering with Rich
	- Color-coded agent messages
	- Animated message updates
	- **Scrollable chat history** (mouse and keyboard scrolling)
	- Tool approval prompts with interactive buttons

4. **Textual UI Structure**

```python
from textual.app import App
from textual.containers import Horizontal, Vertical
from textual.widgets import Input, RichLog, Static

class NeuroDeckUI(App):
    def compose(self):
        yield Vertical(
            Static("NeuroDeck - Multi-Agent Chat", id="header"),
            Horizontal(
                RichLog(id="chat_log"),
                Static("Agents:\n• Claudius\n• Grok\n• ChatGPT", id="sidebar")
            ),
            Input(placeholder="Type your message...", id="input")
        )
    
    def handle_message(self, message):
        """Handle incoming messages from orchestrator"""
        if message["type"] == "chat_message":
            # Regular chat - display with agent colors
            self.query_one("#chat_log").write(
                f"[bold {self.get_agent_color(message['sender'])}]{message['sender']}[/]: {message['content']}"
            )
        elif message["type"] == "system_log":
            # System logs - display with log level styling, not broadcast to agents
            level_color = {"info": "blue", "warning": "yellow", "error": "red"}[message["level"]]
            self.query_one("#chat_log").write(
                f"[dim {level_color}]🔧 {message['sender']}: {message['content']}[/]"
            )
```

### Command System

```
/add-agent claude-opus as "Research Assistant"
/set-persona "Research Assistant" expert-in="quantum physics"
/create-group "Development Team" with [claude-sonnet, grok-4]
/broadcast "Let's brainstorm feature ideas"
```

## Context Persistence

### SQLite3 Database Schema

**Agent conversation history persisted in SQLite3 for crash recovery:**

```sql
CREATE TABLE agent_context (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_name TEXT NOT NULL,
    message_type TEXT NOT NULL,  -- 'user', 'assistant', 'system'
    content TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    session_id TEXT NOT NULL
);

CREATE INDEX idx_agent_session ON agent_context(agent_name, session_id);
```

**Context Management:**
- **Automatic persistence**: Every message saved to database
- **Crash recovery**: Agents reload context on restart
- **Context clearing**: TUI commands to clear specific agent contexts
- **Session isolation**: Each orchestrator run gets unique session_id

## Tool Approval System

### Claude-Style Permission Model

**Four approval modes for tool execution:**
1. **Once**: Approve this single tool execution
2. **Same**: Approve this and identical future commands 
3. **All**: Approve all tools for this agent (YOLO mode)
4. **Deny**: Reject this tool execution

**Approval Flow:**
1. Agent requests tool execution via `tool_request` message
2. Orchestrator shows approval prompt to user in TUI
3. User responds with approval level
4. Orchestrator sends approval result to agent
5. Agent executes tool (if approved) or shows denial message

## Technical Implementation

### Technology Stack

- **Core Language**: Python 3.10+
- **Orchestrator**: asyncio-based TCP server
- **Console UI**: Textual framework for modern terminal interface
- **IPC**: Line-based JSON protocol over TLS sockets
- **Process Management**: multiprocessing module
- **Configuration**: INI files with Python configparser
- **Database**: SQLite3 for context persistence
- **Logging**: structured logging with loguru

## Configuration

### Agent Configuration File

Each agent is defined in `agents.ini`:

```ini
[orchestrator]
host = localhost
port = 9999
tls_cert = certs/server.crt
tls_key = certs/server.key
auth_token = your-secret-orchestrator-token-here
log_level = INFO

# Built-in tools: filesystem, web_search
# External MCP tools defined in [mcp:toolname] sections

[agent:claudius]
provider = anthropic
model = claude-3-5-sonnet-20241022
api_endpoint = https://api.anthropic.com/v1/messages
api_key_env = ANTHROPIC_API_KEY
max_tokens = 4096
temperature = 0.7
tools = filesystem,aidderall
system_prompt = You are Claudius, a helpful AI assistant. You excel at code analysis, 
    writing, and general problem solving. Always respond to direct mentions 
    and contribute when you have valuable insights.

[agent:grok]
provider = xai
model = grok-4
api_endpoint = https://api.x.ai/v1/chat/completions
api_key_env = XAI_API_KEY
max_tokens = 4096
temperature = 0.3
tools = filesystem,security_scanner
system_prompt = You are Grok, a security-focused AI with a direct communication style. 
    You specialize in finding vulnerabilities, code security, and system analysis.
    Respond to mentions and security-related questions.

[agent:chatgpt]
provider = openai
model = gpt-4o
api_endpoint = https://api.openai.com/v1/chat/completions
api_key_env = OPENAI_API_KEY
max_tokens = 4096
temperature = 0.7
tools = filesystem,web_search
system_prompt = You are ChatGPT, a versatile AI assistant. You're good at creative tasks,
    explanations, and general help. Join conversations when you can add value.

[agent:kimi]
provider = groq
model = moonshotai/kimi-k2-instruct
api_endpoint = https://api.groq.com/openai/v1/chat/completions
api_key_env = GROQ_API_KEY
max_tokens = 16384
temperature = 0.8
tools = filesystem,web_search,postgres_mcp
system_prompt = You are Kimi, an advanced AI with exceptional reasoning capabilities. 
    You have a 131k context window and excel at complex analysis, research, and 
    multi-step problem solving. Join conversations when deep thinking is needed.

# MCP Tool Definitions for external tools
[mcp:aidderall]
command = python
args = /path/to/aidderall/server.py
description = AI-powered code assistant with advanced analysis capabilities

[mcp:security_scanner]
command = node
args = /opt/security-tools/mcp-server.js
description = Security vulnerability scanner for code analysis

[mcp:postgres_mcp]
command = /usr/local/bin/postgres-mcp-server
args = --connection-string=postgresql://user:pass@localhost/db
description = PostgreSQL database interface via MCP
```

### Provider-Specific Client Selection

The `provider` field determines which API client library and request format each agent uses:

```python
# Agent process selects client based on provider
def create_api_client(agent_config):
    provider = agent_config['provider']
    
    if provider == 'anthropic':
        import anthropic
        return anthropic.Anthropic(api_key=os.getenv(agent_config['api_key_env']))
    
    elif provider == 'openai':
        import openai
        return openai.OpenAI(api_key=os.getenv(agent_config['api_key_env']))
    
    elif provider == 'xai':
        import openai  # xAI uses OpenAI-compatible client
        return openai.OpenAI(
            api_key=os.getenv(agent_config['api_key_env']),
            base_url=agent_config['api_endpoint']
        )
    
    elif provider == 'groq':
        import groq
        return groq.Groq(api_key=os.getenv(agent_config['api_key_env']))
```

### Configuration Loading

```python
import configparser
from pathlib import Path

def load_config(config_path: str = "config/agents.ini"):
    """Load agent configuration from INI file"""
    config = configparser.ConfigParser()
    config.read(config_path)
    
    # Load agent configurations
    agents = {}
    for section in config.sections():
        if section.startswith('agent:'):
            agent_name = section.split(':', 1)[1]
            agents[agent_name] = {
                'provider': config.get(section, 'provider'),  # Determines API client type
                'model': config.get(section, 'model'),
                'api_endpoint': config.get(section, 'api_endpoint'),
                'api_key_env': config.get(section, 'api_key_env'),
                'max_tokens': config.getint(section, 'max_tokens'),
                'temperature': config.getfloat(section, 'temperature'),
                'tools': config.get(section, 'tools').split(','),
                'system_prompt': config.get(section, 'system_prompt')
            }
    
    # Load MCP tool definitions
    mcp_tools = {}
    for section in config.sections():
        if section.startswith('mcp:'):
            tool_name = section.split(':', 1)[1]
            mcp_tools[tool_name] = {
                'command': config.get(section, 'command'),
                'args': config.get(section, 'args'),
                'description': config.get(section, 'description')
            }
    
    return agents, dict(config['orchestrator']), mcp_tools

# Built-in tools that don't require MCP server startup
BUILT_IN_TOOLS = {'filesystem', 'web_search'}

def is_mcp_tool(tool_name: str, mcp_tools: dict) -> bool:
    """Check if a tool requires MCP server startup"""
    return tool_name not in BUILT_IN_TOOLS and tool_name in mcp_tools
```

### Data Models

```python
class AgentProcess:
    pid: int
    agent_id: str
    provider: str
    model: str
    name: str
    socket_path: str
    status: str  # 'starting', 'ready', 'busy', 'error'
    
class IPCMessage:
    id: str
    type: str  # 'user_message', 'agent_response', 'system_command'
    source: str
    target: Optional[str]
    content: str
    metadata: Dict[str, Any]
    timestamp: datetime
    
class AgentContext:
    agent_id: str
    messages: List[Message]
    system_prompt: str
    max_tokens: int
    temperature: float
```

### Line-Based JSON Protocol

**Protocol**: One JSON message per line, newline-delimited over TLS socket

**⚠️ CRITICAL: JSON Escaping Requirements**
- **All chat content MUST be properly JSON-escaped**
- Use `json.dumps()` for encoding - NEVER manual string concatenation
- Chat messages can contain quotes, newlines, backslashes, unicode
- Example: `"Hello \"world\"\nNew line"` → `"Hello \\\"world\\\"\\nNew line"`

#### **TUI → Orchestrator Messages**

```json
// Authentication (first message after connection)
{"type": "auth", "token": "secret-token", "client_id": "tui-session-123"}

// User chat message
{"type": "chat", "content": "Hey everyone, who's online?"}

// System commands
{"type": "command", "cmd": "list_agents"}
{"type": "command", "cmd": "clear_context", "agents": ["claude", "grok"]}
{"type": "command", "cmd": "clear_context", "agents": ["all"]}

// Tool approval responses
{"type": "tool_approval", "agent": "claude", "approval": "once|same|all|deny"}
```

#### **Agent → Orchestrator Messages** (Same Protocol)

```json
// Agent authentication
{"type": "auth", "token": "secret-token", "client_id": "agent-claude"}

// Agent chat response
{"type": "chat", "sender": "claude", "content": "I'll help you with that code analysis..."}

// System log from agent
{"type": "system_log", "sender": "claude", "level": "info", "message": "API request completed successfully"}

// Tool execution request (requires user approval)
{"type": "tool_request", "agent": "claude", "tool": "filesystem", "command": "read_file", "args": {"path": "/etc/hosts"}}
```

#### **Orchestrator → TUI Messages**

```json
// Authentication response
{"type": "auth_result", "success": true, "message": "Authenticated successfully"}

// Chat message from agent (broadcast from agents)
{"type": "chat", "sender": "claude", "content": "Claude here, ready to help!", "timestamp": "2024-01-01T12:00:00Z"}

// User message echo (for consistency)
{"type": "chat", "sender": "user", "content": "Hey everyone, who's online?", "timestamp": "2024-01-01T12:00:00Z"}

// System log (only to TUI, not broadcast to agents)
{"type": "system_log", "sender": "grok", "level": "error", "message": "API rate limit exceeded, retrying in 30 seconds"}

// Command responses
{"type": "command_result", "cmd": "list_agents", "success": true, "data": ["claude", "grok", "chatgpt"]}
{"type": "command_result", "cmd": "clear_context", "success": true, "message": "Context cleared for claude, grok"}

// Tool approval request (user must approve before agent can execute)
{"type": "tool_approval_request", "agent": "claude", "tool": "filesystem", "command": "read_file", "args": {"path": "/etc/hosts"}, "description": "Read system hosts file"}
```

#### **Orchestrator → Agent Messages** (Same Protocol)

```json
// Authentication response
{"type": "auth_result", "success": true, "message": "Agent authenticated"}

// Broadcast chat message from user or other agents
{"type": "chat", "sender": "user", "content": "Can someone analyze this code?", "timestamp": "2024-01-01T12:00:00Z"}

// Tool approval response
{"type": "tool_approval_result", "agent": "claude", "approval": "once", "tool_request_id": "req-123"}

// Context clear command
{"type": "clear_context", "agent": "claude"}
```

#### **Implementation Example**

```python
# ✅ CORRECT: Always use json.dumps() for proper escaping
message = {"type": "chat", "content": "Hello \"world\"\nWith newlines!"}
await writer.write(f"{json.dumps(message)}\n".encode())

# ❌ WRONG: Manual string formatting will break with quotes/newlines
# DO NOT DO THIS:
# line = f'{{"type": "chat", "content": "{content}"}}\n'  # BROKEN!

# Orchestrator receiving with error handling
async def handle_client(reader, writer):
    while True:
        try:
            line = await reader.readline()
            if not line:
                break
            # Proper JSON parsing handles all escaping automatically
            message = json.loads(line.decode().strip())
            await process_message(message, writer)
        except json.JSONDecodeError as e:
            error_response = {"type": "error", "message": f"Invalid JSON: {str(e)}"}
            await writer.write(f"{json.dumps(error_response)}\n".encode())

# Message processing with safe content handling
async def process_message(message, writer):
    if message["type"] == "chat":
        # Content is already properly unescaped by json.loads()
        safe_content = message["content"]  # Can contain any characters now
        await broadcast_to_agents(safe_content)
        
        # Echo to TUI - json.dumps() handles re-escaping
        response = {"type": "chat", "sender": "user", "content": safe_content}
        await writer.write(f"{json.dumps(response)}\n".encode())
```

#### **JSON Escaping Examples**

```python
# Examples of content that MUST be properly escaped:
problematic_content = [
    'Message with "quotes"',
    'Multi-line\nmessage\nhere',
    'Backslash \\ characters',
    'Code: if (x == "test") { return "value"; }',
    'Unicode: 你好 🚀 émojis',
    'JSON-like: {"fake": "json", "data": [1,2,3]}'
]

# ✅ ALWAYS use json.dumps() - handles ALL cases correctly
for content in problematic_content:
    message = {"type": "chat", "content": content}
    safe_line = f"{json.dumps(message)}\n"  # Properly escaped
    
# The json.loads() on receiving end automatically unescapes everything
```

## Security & Privacy

### Authentication & Security

#### Client Authentication
- Token-based authentication between client and orchestrator
- Orchestrator token stored in `agents.ini` config file
- Client reads token from `.env` file (`NEURODECK_TOKEN`)
- Token transmitted in TLS-encrypted connection header

#### TLS Configuration
- TLS 1.3 for all socket communications
- Certificate-based encryption
- Self-signed certs for local development
- Proper CA certs for production

#### Process Isolation & Recovery
- Each agent runs with limited permissions
- Sandboxed filesystem access
- Resource limits per process
- No direct inter-agent communication
- **Agent crash recovery**: Orchestrator automatically restarts failed agents
- **Error visibility**: Agent crashes/restarts logged as system messages to user UI

#### API Key Management
- AI provider keys stored in environment variables
- Per-agent key isolation
- No key exposure in logs or config files

## Implementation Phases

### Phase 1: Core Infrastructure (Week 1-2)
- Orchestrator daemon with TCP socket
- Basic console UI client
- Single agent process spawning
- TLS socket communication

### Phase 2: Multi-Agent Support (Week 3-4)
- Multiple agent process management
- Inter-agent message routing
- Claude and OpenAI adapters
- Basic @ mention support

### Phase 3: Tool Integration (Week 5-6)
- MCP server support
- Filesystem access controls
- Tool execution framework
- Context persistence

### Phase 4: Enhanced Features (Week 7-8)
- xAI and GroqAI adapters
- Advanced command system
- Performance optimization
- Configuration management

## Directory Structure

```
neurodeck/
├── orchestrator/
│   ├── __main__.py       # Daemon entry point
│   ├── server.py         # TCP socket server
│   ├── process_manager.py # Agent process management
│   └── router.py         # Message routing
├── agents/
│   ├── base_agent.py     # Base agent class
│   ├── claude_agent.py   # Anthropic Claude
│   ├── openai_agent.py   # OpenAI ChatGPT
│   ├── xai_agent.py      # xAI Grok
│   └── groq_agent.py     # GroqAI models
├── console/
│   ├── __main__.py       # Console UI entry
│   ├── client.py         # Socket client
│   ├── ui.py            # Terminal interface
│   └── commands.py      # Command parsing
├── common/
│   ├── protocol.py      # Message protocol
│   ├── security.py      # TLS handling
│   └── config.py        # Configuration
├── tools/
│   ├── mcp_client.py    # MCP integration using Anthropic's Python SDK
│   └── filesystem.py    # File operations
├── config/
│   ├── agents.ini      # Agent definitions & orchestrator token
│   ├── certs/          # TLS certificates
│   └── .env            # Environment variables (NEURODECK_TOKEN, API keys)
└── data/
    └── context.db      # SQLite3 database for agent context persistence
```

## Key Design Decisions

1. **Process per Agent**: Each AI runs in its own process for isolation and resource management
2. **TCP Sockets**: Reliable communication between components with TLS security
3. **Console-First**: Terminal interface optimized for developers and power users
4. **Stateful Agents**: Each agent maintains its own context and conversation history
5. **Tool Flexibility**: Agents can have different tool sets based on their roles

## Example Usage Flow

```bash
# Start the orchestrator daemon (loads agents from config)
$ python -m neurodeck.orchestrator --config config/agents.ini

# In another terminal, start the console UI (reads NEURODECK_TOKEN from .env)
$ python -m neurodeck.console --connect localhost:9999

# In the console (agents auto-loaded from config):
> Hey everyone, who's online?
[Claudius] Ready to help with analysis and coding!
🔧 Grok: API connection established, security tools loaded
[Grok] Security scanner online, ready for threats!
[ChatGPT] Here and ready for creative tasks!
🔧 Kimi: Rate limit warning - requests may be delayed
[Kimi] Kimi here with 131k context window, ready for complex reasoning!

> @grok can you scan this file for vulnerabilities?
[Grok] I'll analyze it for security issues...
🔧 Grok: Running security scan with 3 different tools...
[Claudius] I can also check the code quality while Grok handles security...
🔧 Claudius: MCP tool 'aidderall' started successfully

> Thanks both! What did you find?
[Grok] Found 2 potential SQL injection points...
🔧 Grok: Detailed report saved to security_scan_2024-01-01.json
[Claudius] The code structure looks good overall, but here are some improvements...
```

## Conclusion

NeuroDeck brings the power of multi-agent AI collaboration to the command line, enabling developers to orchestrate multiple AI models in a unified console environment. By combining process isolation, secure communication, and a familiar terminal interface, NeuroDeck provides a robust platform for complex AI-assisted workflows.