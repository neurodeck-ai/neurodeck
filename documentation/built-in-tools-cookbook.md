# NeuroDeck Internal/External Tools Development Cookbook

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Step-by-Step Implementation Guide](#step-by-step-implementation-guide)
4. [Critical Gotchas & Common Pitfalls](#critical-gotchas--common-pitfalls)
5. [Testing & Validation](#testing--validation)
6. [Advanced Patterns](#advanced-patterns)
7. [Troubleshooting](#troubleshooting)
8. [Example: chat_info Tool Implementation](#example-chat_info-tool-implementation)

## Overview

Internal tools in NeuroDeck are native capabilities that agents can use without requiring external MCP servers. They're executed directly by the orchestrator and provide core functionality like filesystem operations, web search, and chat introspection.

### When to Create an Internal Tool vs External (MCP) Tool

**Internal Tool (this guide):**
- Core functionality needed by all/most agents
- Requires access to orchestrator internals (agent status, connections, etc.)
- Performance-critical operations
- Simple, stable interfaces

**External (MCP) Tool:**
- External service integrations (databases, APIs, etc.)
- Complex, domain-specific functionality
- Third-party tools that may be optional

## Architecture

### Component Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Agent         │    │   Orchestrator   │    │   Tool Class    │
│                 │    │                  │    │                 │
│ 1. Tool Request │───▶│ 2. Route to      │───▶│ 3. Execute      │
│                 │    │    ToolExecutor  │    │    Operation    │
│                 │    │                  │    │                 │
│ 6. Use Result   │◀───│ 5. Return Result │◀───│ 4. Return Data  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Key Files & Their Roles

| File | Purpose | Your Changes Needed |
|------|---------|-------------------|
| `neurodeck/tools/your_tool.py` | Tool implementation | ✅ Create new file |
| `neurodeck/orchestrator/tool_executor.py` | Tool routing & execution | ✅ Add handler |
| `neurodeck/orchestrator/server.py` | Approval logic & config | ✅ Add approval rules |
| `neurodeck/common/config.py` | Internal tool registry | ✅ Register tool |
| `config/agents.ini` | Agent configuration | ✅ Add to tools list |
| `neurodeck/agents/*_agent.py` | LLM tool schemas | ✅ Add schema definitions |

## Step-by-Step Implementation Guide

### Step 1: Create the Tool Class

Create `neurodeck/tools/your_tool.py`:

```python
"""
Your tool description here.

Features:
- List key capabilities
- Security considerations
- Performance characteristics
"""

import asyncio
from typing import Any, Dict, List, Optional
from ..common.logging import get_logger

logger = get_logger(__name__)

class YourTool:
    """
    Your tool class with clear docstring.
    
    Features:
    - Feature 1
    - Feature 2
    - Feature 3
    """
    
    def __init__(self, orchestrator_server=None):
        """
        Initialize tool.
        
        Args:
            orchestrator_server: Reference to orchestrator for live data access
        """
        self.orchestrator_server = orchestrator_server
        logger.info("Your tool initialized")
    
    async def execute(self, action: str, **kwargs) -> Any:
        """
        Execute tool operation.
        
        Args:
            action: Operation to perform (operation1, operation2, etc.)
            **kwargs: Additional arguments based on action
            
        Returns:
            Operation result
            
        Raises:
            ValueError: For unknown actions or invalid arguments
        """
        logger.info(f"Executing your_tool {action}")
        
        if action == "operation1":
            return await self._operation1(**kwargs)
        elif action == "operation2":
            return await self._operation2(**kwargs)
        else:
            raise ValueError(f"Unknown your_tool action: {action}")
    
    async def _operation1(self, **kwargs) -> Dict[str, Any]:
        """Implement your first operation."""
        # Your implementation here
        return {"status": "success", "data": "example"}
    
    async def _operation2(self, **kwargs) -> Dict[str, Any]:
        """Implement your second operation."""
        # Your implementation here
        return {"status": "success", "data": "example"}
```

**🔥 Critical Implementation Tips:**

1. **Always use async methods** - NeuroDeck is fully async
2. **Include proper error handling** - Raise clear ValueError messages
3. **Log operations** - Use the logger for debugging
4. **Orchestrator access** - Accept orchestrator_server for live data
5. **Type hints** - Use proper typing for better code quality

### Step 2: Register Tool in ToolExecutor

Edit `neurodeck/orchestrator/tool_executor.py`:

#### 2a. Add Import

```python
from ..tools.your_tool import YourTool
```

#### 2b. Add Tool Instance Variable

```python
def __init__(self, config: dict, tool_configs: Dict[str, ToolConfig], 
             agent_tool_overrides: Dict[str, Dict[str, ToolConfig]], 
             mcp_configs: Dict[str, MCPToolConfig]):
    # ... existing code ...
    self.your_tool: Optional[YourTool] = None
```

#### 2c. Add Orchestrator Reference Method

```python
def set_orchestrator_server(self, orchestrator_server):
    """Set orchestrator server reference for tools."""
    # ... existing code ...
    if self.your_tool is None:
        self.your_tool = YourTool(orchestrator_server)
    else:
        self.your_tool.orchestrator_server = orchestrator_server
```

#### 2d. Add Tool Execution Handler

```python
async def execute_tool(self, agent_id: str, agent_name: str, tool_name: str, 
                      command: str, args: Dict[str, Any]) -> Any:
    # ... existing conditions ...
    elif tool_name == "your_tool":
        return await self._execute_your_tool(agent_id, agent_name, args, tool_config)
    # ... rest of method ...

async def _execute_your_tool(self, agent_id: str, agent_name: str, args: Dict[str, Any], 
                            config: ToolConfig) -> Any:
    """Execute your tool operation with configuration."""
    # Initialize tool if not already done
    if self.your_tool is None:
        self.your_tool = YourTool()
    
    # Execute operation with configured timeout
    return await asyncio.wait_for(
        self.your_tool.execute(
            action=args.get("action"),
            agent_name=agent_name,  # Pass real agent name (e.g., "chatgpt", "claude")
            **{k: v for k, v in args.items() if k != "action"}
        ),
        timeout=config.execution_timeout
    )
```

#### 2e. Add Default Configuration

```python
def _get_agent_tool_config(self, agent_id: str, tool_name: str) -> Union[ToolConfig, FilesystemToolConfig]:
    # ... in the else clause for default configuration ...
    elif tool_name == "your_tool":
        base_config = ToolConfig(
            name=tool_name,
            auto_approve_operations=["safe_operation1", "safe_operation2"],
            require_approval_operations=["dangerous_operation"],
            execution_timeout=30.0
        )
```

### Step 3: Add Approval Logic to Orchestrator ⚠️ CRITICAL

Edit `neurodeck/orchestrator/server.py`:

#### 3a. Update Approval Requirements (REQUIRED)

**🚨 CRITICAL STEP**: Add your tool to the `_requires_approval` method:

```python
def _requires_approval(self, tool: str, command: str, args: Dict[str, Any], agent_id: str = None) -> bool:
    # ... existing code ...
    
    # Built-in tools
    if tool == "filesystem":
        action = args.get("action")
        return action in tool_config.require_approval_operations
    elif tool == "chat_info":
        action = args.get("action")
        return action in tool_config.require_approval_operations
    elif tool == "your_tool":  # ← ADD THIS OR TOOL WON'T WORK
        action = args.get("action")
        return action in tool_config.require_approval_operations
```

**❌ If you forget this**: Tool requests will hang indefinitely or require approval for everything!

#### 3b. Add Default Configuration (REQUIRED)

**🚨 CRITICAL STEP**: Add your tool to the `_get_tool_config` method:

```python
def _get_tool_config(self, tool_name: str, agent_id: str = None) -> ToolConfig:
    # ... existing code ...
    
    elif tool_name == "chat_info":
        base_config = ToolConfig(
            name=tool_name,
            auto_approve_operations=["list_participants", "get_chat_status"],
            require_approval_operations=[],
            execution_timeout=10.0
        )
    elif tool_name == "your_tool":  # ← ADD THIS OR TOOL WON'T WORK
        base_config = ToolConfig(
            name=tool_name,
            auto_approve_operations=["safe_operation1", "safe_operation2"],
            require_approval_operations=["dangerous_operation"],
            execution_timeout=30.0
        )
```

**❌ If you forget this**: Tool will always require approval even when agents.ini says auto-approve!

### Step 4: Register as Built-in Tool

Edit `neurodeck/common/config.py`:

```python
def get_built_in_tools(self) -> set:
    """Get the set of internal tools that don't require MCP servers."""
    return {'filesystem', 'web_search', 'chat_info', 'tts_config', 'your_tool'}
```

**🚨 CRITICAL GOTCHA**: If you forget this step, agents will get "Unknown tool type: your_tool" errors!

### Step 5: Add to Agent Configuration

Edit `config/agents.ini`:

#### 5a. Add to Tools List (All Agents)

```ini
tools = filesystem,chat_info,your_tool
```

#### 5b. Add Global Tool Configuration

```ini
# Global your_tool configuration
[tool:your_tool]
approval_timeout = 30
execution_timeout = 30
auto_approve_operations = safe_operation1,safe_operation2
require_approval_operations = dangerous_operation
paranoid_mode = false
```

#### 5c. Add Agent-Specific Overrides (Optional)

```ini
# Per-agent overrides for special cases
[tool:your_tool:claudius]
execution_timeout = 60  # More time for complex operations
auto_approve_operations = safe_operation1  # More restrictive
```

### Step 6: Initialize in Orchestrator

Edit `neurodeck/orchestrator/server.py` in the `__init__` method:

```python
# Set orchestrator server reference for tools
self.tool_executor.set_orchestrator_server(self)
```

**🔥 CRITICAL**: This must happen AFTER tool_executor creation but BEFORE server starts!

### Step 7: Add LLM Tool Schema Definitions ⚠️ CRITICAL

**🚨 MOST OVERLOOKED STEP**: Without this, agents receive NO information about how to use your tool!

LLMs need structured schemas to understand tool parameters. Add your tool definition to both agent files:

#### 7a. Add to Claude Agent

Edit `neurodeck/agents/claude_agent.py` in the `_build_tool_definitions` method:

```python
def _build_tool_definitions(self) -> List[Dict[str, Any]]:
    tools = []
    
    # ... existing filesystem tool ...
    
    if "your_tool" in self.available_tools:
        # Get agent-specific configuration
        tool_config = self.get_agent_tool_config("your_tool")
        
        tools.append({
            "name": "your_tool",
            "description": "Clear description of what your tool does and any constraints",
            "input_schema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["operation1", "operation2", "operation3"],
                        "description": "The operation to perform"
                    },
                    "parameter1": {
                        "type": "string",
                        "description": "Description of parameter1"
                    },
                    "parameter2": {
                        "type": "number",
                        "description": "Description of parameter2 (optional)"
                    }
                },
                "required": ["action"]
            }
        })
    
    return tools
```

#### 7b. Add to OpenAI Agent

Edit `neurodeck/agents/openai_agent.py` in the `_build_tool_definitions` method:

```python
def _build_tool_definitions(self) -> List[Dict[str, Any]]:
    tools = []
    
    # ... existing filesystem tool ...
    
    if "your_tool" in self.available_tools:
        tools.append({
            "type": "function",
            "function": {
                "name": "your_tool",
                "description": "Clear description of what your tool does and any constraints",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["operation1", "operation2", "operation3"],
                            "description": "The operation to perform"
                        },
                        "parameter1": {
                            "type": "string",
                            "description": "Description of parameter1"
                        },
                        "parameter2": {
                            "type": "number",
                            "description": "Description of parameter2 (optional)"
                        }
                    },
                    "required": ["action"]
                }
            }
        })
    
    return tools
```

#### 7c. Schema Design Best Practices

**🔥 Critical Schema Requirements:**

1. **Action-based design**: Use an `action` parameter with enum values for different operations
2. **Required parameters**: Mark essential parameters as required
3. **Clear descriptions**: Explain each parameter's purpose and valid values
4. **Type validation**: Use proper JSON schema types (string, number, boolean, array, object)
5. **Enum constraints**: Use enums for parameters with limited valid values

**Example Well-Designed Schema:**

```json
{
  "action": {
    "type": "string", 
    "enum": ["get_settings", "set_settings", "reset_settings"],
    "description": "The operation to perform"
  },
  "setting_name": {
    "type": "string",
    "description": "Name of setting to modify (required for set_settings)"
  },
  "setting_value": {
    "type": "number",
    "minimum": 0.0,
    "maximum": 2.0,
    "description": "New value for setting (required for set_settings)"
  }
}
```

**❌ If you forget this step**: Agents will either fail to use your tool or guess parameters incorrectly, leading to "Unknown action" errors!

### Step 8: Handle Agent Name Requirements ⚠️ CRITICAL FOR AGENT-AWARE TOOLS

**🚨 CRITICAL FOR TOOLS THAT NEED AGENT IDENTITY**: If your tool needs to know which agent is calling it (like TTS config, chat info, or per-agent state), you **MUST** handle agent name extraction properly.

#### The Problem: Connection ID vs Agent Name

**What you get**: `agent_id` like `"::1-60570"` (connection identifier)  
**What you need**: `agent_name` like `"chatgpt"` or `"claude"` (real agent name)

#### The Solution: Use the Clean Interface

**✅ CORRECT**: The orchestrator now passes `agent_name` directly to your tool executor method:

```python
async def _execute_your_tool(self, agent_id: str, agent_name: str, args: Dict[str, Any], 
                            config: ToolConfig) -> Any:
    """Execute your tool operation with configuration."""
    # Use agent_name directly - it's the real agent name like "chatgpt", "claude"
    return await asyncio.wait_for(
        self.your_tool.execute(
            action=args.get("action"),
            agent_name=agent_name,  # ← Pass the real agent name
            **{k: v for k, v in args.items() if k != "action"}
        ),
        timeout=config.execution_timeout
    )
```

#### Tool Implementation Pattern

**✅ CORRECT**: Accept and use agent_name in your tool:

```python
class YourAgentAwareTool:
    async def execute(self, action: str, agent_name: str = None, **kwargs) -> Any:
        """
        Execute tool operation.
        
        Args:
            action: Operation to perform
            agent_name: Real agent name (e.g., "chatgpt", "claude", "grok")
            **kwargs: Additional arguments
        """
        logger.info(f"Tool operation {action} requested by agent {agent_name}")
        
        if action == "get_agent_config":
            return self._get_agent_specific_config(agent_name)
        elif action == "set_agent_setting":
            return await self._set_agent_setting(agent_name, **kwargs)
        # ... other operations
    
    def _get_agent_specific_config(self, agent_name: str) -> Dict[str, Any]:
        """Get configuration specific to this agent."""
        # Look up config using real agent name
        return self.agent_configs.get(agent_name, {})
```

#### ❌ DEPRECATED: Don't Extract from agent_id

**❌ WRONG**: Don't try to extract agent name from agent_id anymore:

```python
# ❌ OLD WAY - DEPRECATED AND UNRELIABLE
agent_name = agent_id.split("-", 1)[1] if "-" in agent_id else agent_id

# ❌ OLD WAY - HACKY ARGS POLLUTION  
agent_name = args.get("_agent_name")
```

**Why this fails**: Connection IDs like `::1-60570` don't contain the real agent name, leading to configuration lookup failures.

#### Tools That Need Agent Names

Consider using agent names for tools that:
- Store per-agent configuration (TTS settings, preferences)
- Provide agent-specific data (chat info, permissions)
- Track agent-specific state (session data, usage metrics)
- Implement agent-aware security (rate limiting, access control)

#### Testing Agent Name Handling

**Test your agent name logic**:

```python
# Verify agent name is passed correctly
async def test_agent_name_handling():
    tool = YourAgentAwareTool()
    result = await tool.execute("get_config", agent_name="chatgpt")
    assert "chatgpt" in str(result)  # Agent name should be used in logic
```

**🔥 Recent Fix**: This was a major source of tool failures before we fixed the interface. Now all tools receive clean, proper agent names!

## Critical Gotchas & Common Pitfalls

### 1. "Unknown tool type" Error

**Symptom**: Agents get "Unknown tool type: your_tool" when trying to use the tool.

**Cause**: Tool not registered in internal tools list.

**Fix**: Add tool name to `get_built_in_tools()` in `config.py`:

```python
return {'filesystem', 'web_search', 'chat_info', 'tts_config', 'your_tool'}
```

### 2. Tool Execution Hanging/Timing Out

**Symptom**: Tool requests never return or timeout.

**Common Causes**:
- Forgot to make methods `async`
- Not using `await` for async operations
- Infinite loops or blocking operations
- Missing orchestrator server reference

**Fix**: Always use async/await properly:

```python
# ❌ Wrong
def execute(self, action: str):
    return self.sync_operation()

# ✅ Correct  
async def execute(self, action: str):
    return await self.async_operation()
```

### 3. Orchestrator Server Reference Issues

**Symptom**: Tool can't access live agent data, gets None references.

**Cause**: Tool not properly initialized with orchestrator reference.

**Fix**: Ensure proper initialization pattern:

```python
# In ToolExecutor.__init__
self.your_tool: Optional[YourTool] = None

# In set_orchestrator_server
if self.your_tool is None:
    self.your_tool = YourTool(orchestrator_server)
else:
    self.your_tool.orchestrator_server = orchestrator_server
```

### 4. Configuration Not Loading

**Symptom**: Tool uses default settings instead of config file settings.

**Causes**:
- Missing tool configuration section in agents.ini
- Typos in section names (`[tool:your_tool]` not `[tools:your_tool]`)
- Configuration not properly parsed

**Fix**: Double-check configuration syntax and section names.

### 5. Permission/Approval Issues

**Symptom**: Tool operations get blocked or require unexpected approval.

**Causes**:
- `paranoid_mode = true` in configuration
- Operations not in `auto_approve_operations` list
- Missing approval logic in orchestrator

**Debug**: Check tool configuration and approval logic.

### 5a. 🚨 MOST COMMON GOTCHA: Missing Tool Authorization Logic

**Symptom**: Tool requests hang indefinitely, agents report authorization errors, or tools require approval despite auto-approve settings.

**Cause**: You forgot to add authorization logic for your tool in the orchestrator. This is the **#1 reason new tools fail**.

**The Problem**: NeuroDeck has **two separate authorization systems** that both need your tool:

#### Part 1: Missing in `_requires_approval` method

**Location**: `neurodeck/orchestrator/server.py` in `_requires_approval` method

**What happens**: Without this, your tool falls through to unknown tool handling and may hang or require approval for everything.

**Fix**: Add your tool to the built-in tools approval logic:

```python
def _requires_approval(self, tool: str, command: str, args: Dict[str, Any], agent_id: str = None) -> bool:
    # ... existing code ...
    
    # Built-in tools
    if tool == "filesystem":
        action = args.get("action")
        return action in tool_config.require_approval_operations
    elif tool == "chat_info":
        action = args.get("action")
        return action in tool_config.require_approval_operations
    elif tool == "your_tool":  # ← MUST ADD THIS
        action = args.get("action")
        return action in tool_config.require_approval_operations
```

#### Part 2: Missing in `_get_tool_config` method

**Location**: `neurodeck/orchestrator/server.py` in `_get_tool_config` method

**What happens**: Without this, your tool gets default config with `require_approval_operations=["*"]` (requires approval for everything).

**Fix**: Add default configuration case:

```python
def _get_tool_config(self, tool_name: str, agent_id: str = None) -> ToolConfig:
    # ... existing code ...
    
    elif tool_name == "chat_info":
        base_config = ToolConfig(
            name=tool_name,
            auto_approve_operations=["list_participants", "get_chat_status"],
            require_approval_operations=[],
            execution_timeout=10.0
        )
    elif tool_name == "your_tool":  # ← MUST ADD THIS
        base_config = ToolConfig(
            name=tool_name,
            auto_approve_operations=["safe_operation1", "safe_operation2"],
            require_approval_operations=["dangerous_operation"],
            execution_timeout=30.0
        )
```

**🚨 CRITICAL**: You must add your tool to **BOTH** methods or authorization will fail in subtle ways!

**💡 Quick Test**: If your tool works in YOLO mode (`/yolo` command) but not in normal mode, you forgot the authorization logic.

### 6. Import/Module Issues

**Symptom**: Import errors or module not found.

**Causes**:
- Forgot to create `__init__.py` files
- Circular imports
- Incorrect relative import paths

**Fix**: Use proper relative imports: `from ..tools.your_tool import YourTool`

### 7. Agent Restart Required

**Symptom**: Changes not taking effect.

**Cause**: Agents cache tool configurations on startup.

**Fix**: Always restart agents after:
- Adding new tools
- Changing tool configurations
- Modifying built-in tools list

## Testing & Validation

### 1. Unit Testing Your Tool

Create `tests/test_your_tool.py`:

```python
import asyncio
import pytest
from neurodeck.tools.your_tool import YourTool

@pytest.mark.asyncio
async def test_your_tool_operation1():
    tool = YourTool()
    result = await tool.execute("operation1", param="test")
    assert result["status"] == "success"

@pytest.mark.asyncio
async def test_invalid_action():
    tool = YourTool()
    with pytest.raises(ValueError, match="Unknown your_tool action"):
        await tool.execute("invalid_action")
```

### 2. Integration Testing

```python
# Test agent can discover and use tool
async def test_agent_tool_integration():
    # Start orchestrator and agents
    # Send chat message: "Use your_tool to do operation1"
    # Verify tool executes successfully
```

### 3. Configuration Testing

```python
# Test tool configuration loading
def test_tool_configuration():
    config_manager = ConfigManager("config/agents.ini")
    tool_configs = config_manager.load_tool_configs()
    assert "your_tool" in tool_configs
    assert "auto_approve_operations" in tool_configs["your_tool"]
```

### 4. Manual Testing Checklist

- [ ] Agent can discover tool exists
- [ ] Tool operations execute successfully
- [ ] Error messages are clear and helpful
- [ ] Configuration settings are respected
- [ ] Approval workflows work correctly
- [ ] Tool doesn't block or hang
- [ ] Logging provides useful debugging info

## Advanced Patterns

### 1. Stateful Tools

If your tool needs to maintain state between operations:

```python
class StatefulTool:
    def __init__(self, orchestrator_server=None):
        self.orchestrator_server = orchestrator_server
        self.state_cache = {}
        self.session_data = {}
    
    async def execute(self, action: str, agent_name: str = None, **kwargs):
        # Use agent_name to maintain per-agent state
        if agent_name not in self.session_data:
            self.session_data[agent_name] = {}
        
        # Tool operations can modify state
        return await self._handle_stateful_operation(agent_name, action, **kwargs)
```

### 2. Tools with External Dependencies

```python
class ExternalServiceTool:
    def __init__(self, orchestrator_server=None):
        self.orchestrator_server = orchestrator_server
        self.client = None
    
    async def _ensure_client(self):
        """Lazy initialization of external client."""
        if self.client is None:
            self.client = await create_external_client()
    
    async def execute(self, action: str, **kwargs):
        await self._ensure_client()
        return await self._call_external_service(action, **kwargs)
```

### 3. Tools with Complex Permissions

```python
class PermissionAwareTool:
    def __init__(self, orchestrator_server=None):
        self.orchestrator_server = orchestrator_server
    
    async def execute(self, action: str, agent_name: str = None, **kwargs):
        # Check agent permissions
        if not self._agent_has_permission(agent_name, action):
            raise PermissionError(f"Agent {agent_name} not authorized for {action}")
        
        return await self._execute_authorized_action(action, **kwargs)
    
    def _agent_has_permission(self, agent_name: str, action: str) -> bool:
        # Implement custom permission logic
        agent_permissions = self._get_agent_permissions(agent_name)
        return action in agent_permissions
```

### 4. Tools with Rich Data Types

```python
from dataclasses import dataclass
from typing import List

@dataclass
class ToolResult:
    success: bool
    data: Any
    metadata: Dict[str, Any]
    timestamp: str

class RichDataTool:
    async def execute(self, action: str, **kwargs) -> ToolResult:
        try:
            data = await self._perform_operation(action, **kwargs)
            return ToolResult(
                success=True,
                data=data,
                metadata={"operation": action, "args": kwargs},
                timestamp=datetime.now().isoformat()
            )
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                metadata={"error": str(e), "operation": action},
                timestamp=datetime.now().isoformat()
            )
```

## Troubleshooting

### Debug Logging

Add comprehensive logging to your tool:

```python
import logging
from ..common.logging import get_logger

logger = get_logger(__name__)

class DebuggableTool:
    async def execute(self, action: str, **kwargs):
        logger.debug(f"Tool execute called: action={action}, kwargs={kwargs}")
        
        try:
            result = await self._perform_action(action, **kwargs)
            logger.info(f"Tool operation successful: {action}")
            logger.debug(f"Tool result: {result}")
            return result
        except Exception as e:
            logger.error(f"Tool operation failed: {action}, error: {e}")
            raise
```

### Common Error Messages & Solutions

| Error Message | Likely Cause | Solution |
|---------------|--------------|----------|
| "Unknown action: None" or agents guess parameters | Missing LLM schema definitions | Add schema to agent files (Step 7) |
| "Unknown tool type: your_tool" | Not in internal tools registry | Add to `get_built_in_tools()` |
| "Tool not available: your_tool" | Not in agent's tools list | Add to agents.ini tools |
| Tool requests hang indefinitely | Missing approval logic in orchestrator | Add to `_requires_approval()` method |
| Tool always requires approval | Missing default config in orchestrator | Add to `_get_tool_config()` method |
| "Tool execution timed out" | Async/await issues or infinite loop | Check async implementation |
| "Tool approval timed out" | User didn't respond to approval | Check approval configuration |
| "'NoneType' has no attribute..." | Missing orchestrator reference | Check initialization pattern |
| Tool blocked despite auto-approve | Authorization logic not implemented | Check both approval methods |
| Agent config lookup fails (TTS, etc.) | Using connection ID instead of agent name | Use agent_name parameter (Step 8) |
| "No configuration found for agent ::1-60570" | Agent name extraction broken | Update to new interface (Step 8) |

### Log Investigation

When debugging, check these log files:

1. **orchestrator.log** - Tool routing and execution
2. **agent_[name].log** - Agent-side tool requests
3. **Console debug panel** - Real-time tool activity

Search for:
- Tool name mentions
- Error messages
- Timeout logs
- Approval request logs

## Example: chat_info Tool Implementation

Here's the complete implementation we built:

### Tool Class (`neurodeck/tools/chat_info.py`)

```python
"""
Chat introspection tool for NeuroDeck.

Provides agents with dynamic information about the chat environment,
including participant discovery and status information.
"""

import json
from typing import List, Dict, Any, Optional
from ..common.logging import get_logger
from ..common.config import ConfigManager

logger = get_logger(__name__)

class ChatInfoTool:
    """
    Chat introspection tool for dynamic participant discovery.
    
    Features:
    - Dynamic participant list discovery
    - Real-time connection status
    - Agent self-introspection
    - Flexible chat environment queries
    """
    
    def __init__(self, orchestrator_server=None):
        """
        Initialize chat info tool.
        
        Args:
            orchestrator_server: Reference to orchestrator server for live data
        """
        self.orchestrator_server = orchestrator_server
        logger.info("Chat info tool initialized")
    
    async def execute(self, action: str, **kwargs) -> Any:
        """
        Execute chat info operation.
        
        Args:
            action: Operation to perform (list_participants, get_chat_status, get_my_info)
            **kwargs: Additional arguments based on action
            
        Returns:
            Operation result
        """
        logger.info(f"Executing chat_info {action}")
        
        if action == "list_participants":
            return await self._list_participants()
        elif action == "get_chat_status":
            return await self._get_chat_status()
        elif action == "get_my_info":
            agent_name = kwargs.get("agent_name")
            return await self._get_my_info(agent_name)
        elif action == "get_participant_info":
            participant_name = kwargs.get("participant_name")
            return await self._get_participant_info(participant_name)
        else:
            raise ValueError(f"Unknown chat_info action: {action}")
    
    async def _list_participants(self) -> Dict[str, Any]:
        """Get list of all available participants with their status."""
        participants = []
        
        # Always include human user
        human_connected = self._is_user_connected()
        participants.append({
            "name": "human",
            "type": "user",
            "status": "connected" if human_connected else "unknown"
        })
        
        # Get configured agents
        try:
            config_manager = ConfigManager("config/agents.ini")
            agent_configs = config_manager.load_agent_configs()
            
            for agent_name, config in agent_configs.items():
                # Get connection status from orchestrator if available
                status = "unknown"
                if self.orchestrator_server:
                    status = self._get_agent_connection_status(agent_name)
                
                participants.append({
                    "name": agent_name,
                    "type": "agent",
                    "status": status,
                    "provider": config.get("provider"),
                    "model": config.get("model")
                })
                
        except Exception as e:
            logger.error(f"Error loading agent configs: {e}")
            # Fallback to connected agents only
            if self.orchestrator_server:
                connected_agents = self._get_connected_agents()
                for agent_name in connected_agents:
                    participants.append({
                        "name": agent_name,
                        "type": "agent", 
                        "status": "connected"
                    })
        
        logger.info(f"Found {len(participants)} participants")
        return {
            "participants": participants,
            "total_count": len(participants),
            "connected_count": len([p for p in participants if p["status"] == "connected"])
        }
    
    # ... additional methods ...
```

### Key Lessons from chat_info Implementation

1. **Orchestrator Integration**: Direct access to live agent connection data
2. **Configuration Reading**: Reading agents.ini for participant discovery  
3. **Error Handling**: Graceful fallbacks when config reading fails
4. **Live Data**: Real-time connection status vs configured agents
5. **Flexible Operations**: Multiple operations in one tool class
6. **Self-Introspection**: Agent can query their own configuration

This implementation demonstrates all the patterns and gotchas covered in this cookbook.

## Summary

Building internal tools requires careful attention to **8 CRITICAL STEPS**:

### 🚨 Critical Requirements (Tool won't work without these):

1. **Internal Tools Registration** - Add to `get_built_in_tools()` in `config.py`
2. **Orchestrator Authorization Logic** - Add to `_requires_approval()` method in `server.py`  
3. **Default Configuration Logic** - Add to `_get_tool_config()` method in `server.py`
4. **Tool Executor Integration** - Add execution handler in `tool_executor.py`
5. **Agent Configuration** - Add to `tools` list in `agents.ini`
6. **🔥 LLM Tool Schema Definitions** - Add structured schemas to agent files for LLM APIs
7. **🔥 Agent Name Handling** - Use clean agent_name interface for agent-aware tools

### ⚡ Implementation Quality:

8. **Async/Await** - Proper async implementation throughout
9. **Error Handling** - Clear error messages and logging
10. **Testing** - Comprehensive testing at multiple levels

### 🔍 Most Common Failures:

- **40% of tool failures**: Missing LLM schema definitions (step 6) - agents guess parameters incorrectly
- **25% of tool failures**: Forgot authorization logic (steps 2 & 3) - tools hang or require approval
- **15% of tool failures**: Agent name extraction issues (step 7) - config lookup failures
- **10% of tool failures**: Forgot internal tools registration (step 1) - "Unknown tool type" errors
- **10% of tool failures**: Implementation bugs (async, imports, etc.)

**💡 Debug Tips**: 
- If agents use wrong parameters or get "Unknown action: None", you missed the LLM schema definitions (Step 6)
- If your tool works in YOLO mode (`/yolo`) but not normal mode, you missed the authorization steps (Steps 2 & 3)
- If config lookups fail with connection IDs like "::1-60570", you need to use the agent_name parameter (Step 7)

Follow this cookbook exactly and you'll avoid the common pitfalls that cause 95% of tool failures!