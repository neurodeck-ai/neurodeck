# Adding Commands to NeuroDeck: Complete Cookbook

This comprehensive guide covers everything you need to know about adding new commands to NeuroDeck, based on real implementation experience and lessons learned.

## Table of Contents

1. [Overview](#overview)
2. [Command Architecture](#command-architecture)
3. [Step-by-Step Implementation](#step-by-step-implementation)
4. [Real-World Example: YOLO Mode](#real-world-example-yolo-mode)
5. [Common Pitfalls & Solutions](#common-pitfalls--solutions)
6. [Testing Your Command](#testing-your-command)
7. [Best Practices](#best-practices)
8. [Troubleshooting Guide](#troubleshooting-guide)

## Overview

NeuroDeck commands are system-level operations that users can invoke from the UI to control orchestrator behavior. Examples include `/wipe`, `/yolo`, `/ping`, etc.

### Command Flow
```
UI Input (/command) → UI Handler → Protocol Message → Server Handler → Response → UI Display
```

## Command Architecture

### Components Involved

1. **UI Layer** (`neurodeck/console/ui.py`)
   - Parses user input
   - Routes commands to appropriate handlers
   - Sends protocol messages to server
   - Displays responses

2. **Protocol Layer** (`neurodeck/common/protocol.py`)
   - Defines message types and builders
   - Handles JSON encoding/decoding

3. **Server Layer** (`neurodeck/orchestrator/server.py`)
   - Receives command messages
   - Executes command logic
   - Sends responses back to UI

4. **Help/Reference** (`neurodeck/console/ui.py`)
   - Command list displayed in UI sidebar

## Step-by-Step Implementation

### Step 1: Add Command to UI Help Reference

**File:** `neurodeck/console/ui.py`

Find the commands help section (around line 198):

```python
yield Static("@agent message\n/help\n/clear\n/wipe [agent]\n/yolo\n/debug\n/autoscroll\n/quit", id="commands_help")
```

Add your new command to this list:

```python
yield Static("@agent message\n/help\n/clear\n/wipe [agent]\n/yolo\n/mynewcmd\n/debug\n/autoscroll\n/quit", id="commands_help")
```

### Step 2: Add UI Command Handler

**File:** `neurodeck/console/ui.py`

In the `handle_command` method (around line 575), add your command case:

```python
elif command.startswith("/mynewcmd"):
    # Handle my new command
    await self.send_mynewcmd_command(command)
```

**⚠️ Critical:** Place this BEFORE the final `else:` clause that handles unknown commands!

### Step 3: Implement UI Command Sender

**File:** `neurodeck/console/ui.py`

Add the sender method after existing command senders (around line 625):

```python
async def send_mynewcmd_command(self, command: str):
    """Send my new command to orchestrator."""
    if not self.client or not self.client.authenticated:
        self.log_message("system", "❌ Not connected to orchestrator", "error")
        return
    
    try:
        # Parse command arguments if needed
        parts = command.strip().split()
        # Example: "/mynewcmd arg1 arg2"
        
        # Send command to server
        cmd_msg = {
            "type": "command",
            "cmd": "mynewcmd",
            # Add any parameters here
            "param1": parts[1] if len(parts) > 1 else None
        }
        
        await self.client.send_message(cmd_msg)
        self.log_debug("system", "📤 MyNewCmd command sent", "info")
        
    except Exception as e:
        self.log_message("system", f"❌ Failed to send command: {e}", "error")
```

### Step 4: Add Server Command Handler

**File:** `neurodeck/orchestrator/server.py`

In the `handle_command` method (around line 487), add your command case:

```python
elif cmd == "mynewcmd":
    # Handle my new command
    await self.handle_mynewcmd_command(connection, message_dict)
```

**⚠️ Critical:** Place this BEFORE the final `else:` clause that returns "Unknown command"!

### Step 5: Implement Server Command Logic

**File:** `neurodeck/orchestrator/server.py`

Add the actual command handler method (around line 574, after other handlers):

```python
async def handle_mynewcmd_command(self, connection: ClientConnection, message_dict: Dict):
    """Handle my new command with proper error handling."""
    cmd = message_dict.get("cmd", "mynewcmd")
    param1 = message_dict.get("param1")
    
    try:
        # Your command logic here
        result = self.do_something_with_param(param1)
        
        # Send success response
        response = {
            "type": MessageType.COMMAND_RESULT.value,
            "cmd": cmd,
            "success": True,
            "data": f"Command executed successfully: {result}"
        }
        await connection.send_message(response)
        
        # Log the action
        logger.info(f"MyNewCmd executed by {connection.client_id}")
        
    except Exception as e:
        logger.error(f"MyNewCmd failed: {e}")
        error_response = {
            "type": MessageType.COMMAND_RESULT.value,
            "cmd": cmd,
            "success": False,
            "error": f"Command failed: {str(e)}"
        }
        await connection.send_message(error_response)
```

### Step 6: Add UI Response Handler

**File:** `neurodeck/console/ui.py`

Commands need response handlers to display results to users. In the `handle_client_message` method (around line 290), add your command response handler:

```python
elif cmd == "mynewcmd":
    # Handle my new command results
    await self.handle_mynewcmd_result(message)
```

Then implement the response handler method (around line 666):

```python
async def handle_mynewcmd_result(self, message: Dict[str, Any]):
    """Handle my new command result."""
    success = message.get("success", False)
    data = message.get("data", "")
    
    if success:
        # Success case
        self.log_message("system", f"✅ {data}", "info")
    else:
        # Error case or confirmation required
        error_message = message.get("error", data)
        requires_confirmation = message.get("requires_confirmation", False)
        
        if requires_confirmation:
            self.log_message("system", f"⚠️ {error_message}", "warning")
        else:
            self.log_message("system", f"❌ Command failed: {error_message}", "error")
```

**⚠️ Critical:** Without this handler, users won't see command responses!

### Step 7: Add Protocol Message Types (If Needed)

**File:** `neurodeck/common/protocol.py`

If your command needs special message types, add them:

```python
class MessageType(Enum):
    # ... existing types ...
    MYNEWCMD_STATUS = "mynewcmd_status"
```

And add message builder functions:

```python
def mynewcmd_status_message(status: str, data: Any = None) -> Dict[str, Any]:
    """Create my new command status message."""
    msg = {
        "type": MessageType.MYNEWCMD_STATUS.value,
        "status": status
    }
    if data is not None:
        msg["data"] = data
    return msg
```

Don't forget to import in server.py:

```python
from ..common.protocol import (
    # ... existing imports ...
    mynewcmd_status_message
)
```

## Real-World Example: YOLO Mode

Let's examine the complete YOLO mode implementation as a reference:

### 1. UI Help Reference
```python
yield Static("@agent message\n/help\n/clear\n/wipe [agent]\n/yolo\n/debug\n/autoscroll\n/quit", id="commands_help")
```

### 2. UI Command Handler
```python
elif command.startswith("/yolo"):
    # Handle YOLO mode toggle command
    await self.send_yolo_command(command)
```

### 3. UI Command Sender
```python
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
```

### 4. Server Command Handler
```python
elif cmd == "yolo":
    # Handle YOLO mode toggle
    await self.handle_yolo_command(connection, message_dict)
```

### 5. Server Command Logic
```python
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
```

### 6. UI Response Handler
```python
# In handle_client_message method
elif cmd == "yolo":
    # Handle YOLO command results
    await self.handle_yolo_result(message)

# Response handler method
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
```

### 7. Protocol Support
```python
class MessageType(Enum):
    # ... existing types ...
    YOLO_STATUS = "yolo_status"

def yolo_status_message(enabled: bool, message: str) -> Dict[str, Any]:
    """Create YOLO mode status message."""
    return {
        "type": MessageType.YOLO_STATUS.value,
        "enabled": enabled,
        "message": message
    }
```

### 7. Server State Integration
```python
# In OrchestratorServer.__init__()
self.yolo_mode = False

# In _requires_approval() method
def _requires_approval(self, tool: str, command: str, args: Dict[str, Any], agent_id: str = None) -> bool:
    """Determine if tool operation requires user approval."""
    # YOLO mode auto-approves everything
    if self.yolo_mode:
        return False
    # ... rest of approval logic
```

## Common Pitfalls & Solutions

### 1. "Unknown command" Error
**Problem:** Server says command is unknown even after restart.

**Solution:** 
- ✅ Check UI handler is added BEFORE the `else:` clause
- ✅ Check server handler is added BEFORE the `else:` clause  
- ✅ Verify command name matches exactly between UI and server
- ✅ Clear Python bytecode cache: `find . -name "*.pyc" -delete`

### 2. Command Executes But No Response Visible
**Problem:** Command is sent (visible in debug logs) but user sees no response.

**Solution:**
- ✅ Add UI response handler in `handle_client_message` method
- ✅ Implement `handle_[command]_result` method
- ✅ Check server is sending proper `command_result` messages

### 3. Command Not in Help Reference
**Problem:** Users don't know command exists.

**Solution:**
- ✅ Add command to UI help text
- ✅ Consider adding usage examples in help

### 4. Indentation Issues
**Problem:** Method not recognized as part of class.

**Solution:**
- ✅ Use exactly 4 spaces for class method indentation
- ✅ Check with: `python3 -m py_compile filename.py`

### 5. Import Errors
**Problem:** Protocol functions not found.

**Solution:**
- ✅ Add imports to server.py if using custom protocol functions
- ✅ Check import statements are complete

### 6. Python Caching Issues
**Problem:** Changes not reflected after restart.

**Solution:**
```bash
# Clear bytecode cache
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
```

## Testing Your Command

### 1. Syntax Check
```bash
python3 -m py_compile neurodeck/orchestrator/server.py
python3 -m py_compile neurodeck/console/ui.py
python3 -m py_compile neurodeck/common/protocol.py
```

### 2. Import Test
```bash
python3 -c "from neurodeck.common.protocol import MessageType; print('Protocol imports OK')"
```

### 3. Method Verification
```bash
python3 -c "
import ast
with open('neurodeck/orchestrator/server.py', 'r') as f:
    tree = ast.parse(f.read())
methods = []
for node in ast.walk(tree):
    if isinstance(node, ast.ClassDef) and node.name == 'OrchestratorServer':
        methods = [item.name for item in node.body if isinstance(item, ast.FunctionDef)]
        break
print('Found handle_mynewcmd_command:', 'handle_mynewcmd_command' in methods)
"
```

### 4. Integration Test
1. Start orchestrator: `bash run_orchestrator.sh`
2. Connect UI and test command
3. Check logs: `tail -f logs/orchestrator.log`

## Best Practices

### 1. Error Handling
Always wrap command logic in try/catch and return proper error responses:

```python
try:
    # Command logic here
    result = do_something()
    
    response = {
        "type": MessageType.COMMAND_RESULT.value,
        "cmd": cmd,
        "success": True,
        "data": result
    }
except Exception as e:
    logger.error(f"Command failed: {e}")
    response = {
        "type": MessageType.COMMAND_RESULT.value,
        "cmd": cmd,
        "success": False,
        "error": str(e)
    }
finally:
    await connection.send_message(response)
```

### 2. Parameter Validation
Validate and sanitize all parameters:

```python
def validate_agent_name(self, agent_name: str) -> bool:
    """Validate agent name against configured agents."""
    config_manager = ConfigManager("config/agents.ini")
    available_agents = list(config_manager.load_agent_configs().keys())
    return agent_name in available_agents
```

### 3. Logging
Add appropriate logging for debugging and monitoring:

```python
logger.info(f"Command {cmd} executed by {connection.client_id}")
logger.warning(f"Dangerous operation {cmd} by {connection.client_id}")
logger.error(f"Command {cmd} failed: {error}")
```

### 4. Confirmation for Dangerous Operations
For potentially destructive commands, require explicit confirmation:

```python
if not confirmed and is_dangerous_operation():
    response = {
        "type": MessageType.COMMAND_RESULT.value,
        "cmd": cmd,
        "success": False,
        "data": "⚠️ This is dangerous. Use '/command confirm' to proceed.",
        "requires_confirmation": True
    }
    return
```

### 5. State Management
If your command affects server state, initialize it properly:

```python
# In __init__()
self.my_feature_enabled = False

# In command handler
self.my_feature_enabled = not self.my_feature_enabled
```

### 6. Broadcasting Updates
If other clients need to know about state changes:

```python
# Broadcast to UI clients only
status_msg = my_status_message(self.my_state, "State changed")
await self.broadcast_message(status_msg, target_type="ui")
```

## Troubleshooting Guide

### Server Not Loading Changes

**Symptoms:** Command still shows "unknown" after restart

**Debug Steps:**
1. Check server process: `ps aux | grep orchestrator`
2. Verify syntax: `python3 -m py_compile neurodeck/orchestrator/server.py`
3. Clear cache: `find . -name "*.pyc" -delete`
4. Check logs: `tail -f logs/orchestrator.log`
5. Restart properly: `pkill -f neurodeck && bash run_orchestrator.sh`

### UI Not Recognizing Command

**Symptoms:** UI shows "Unknown command" 

**Debug Steps:**
1. Check UI handler placement before `else:` clause
2. Verify command parsing logic
3. Check connection: UI must be connected to orchestrator
4. Test with simple command first

### Protocol Issues

**Symptoms:** JSON encoding/decoding errors

**Debug Steps:**
1. Check message format matches protocol expectations
2. Verify all required fields are present
3. Test message creation manually:
   ```python
   from neurodeck.common.protocol import MessageProtocol
   msg = {"type": "command", "cmd": "test"}
   encoded = MessageProtocol.encode_message(msg)
   decoded = MessageProtocol.decode_message(encoded)
   print(decoded)
   ```

### Permission Issues

**Symptoms:** Command executes but has no effect

**Debug Steps:**
1. Check if command is restricted to UI clients only
2. Verify authentication status
3. Check agent permissions if applicable

## Advanced Patterns

### 1. Stateful Commands
Commands that maintain state across invocations:

```python
# In __init__()
self.feature_state = {
    "enabled": False,
    "settings": {},
    "last_used": None
}

# In command handler
async def handle_feature_command(self, connection, message_dict):
    action = message_dict.get("action", "toggle")
    
    if action == "enable":
        self.feature_state["enabled"] = True
    elif action == "disable":
        self.feature_state["enabled"] = False
    elif action == "status":
        # Return current status
        pass
```

### 2. Multi-Step Commands
Commands that require multiple interactions:

```python
# Store pending operations
self.pending_operations = {}

async def handle_complex_command(self, connection, message_dict):
    step = message_dict.get("step", 1)
    operation_id = message_dict.get("operation_id")
    
    if step == 1:
        # Start operation, store state
        operation_id = str(uuid.uuid4())
        self.pending_operations[operation_id] = {
            "connection": connection,
            "state": "waiting_confirmation"
        }
        # Ask for confirmation
    elif step == 2:
        # Complete operation
        if operation_id in self.pending_operations:
            # Finish the operation
            del self.pending_operations[operation_id]
```

### 3. Batch Commands
Commands that operate on multiple items:

```python
async def handle_batch_command(self, connection, message_dict):
    items = message_dict.get("items", [])
    results = []
    
    for item in items:
        try:
            result = await self.process_item(item)
            results.append({"item": item, "success": True, "result": result})
        except Exception as e:
            results.append({"item": item, "success": False, "error": str(e)})
    
    response = {
        "type": MessageType.COMMAND_RESULT.value,
        "cmd": message_dict.get("cmd"),
        "success": True,
        "data": results
    }
    await connection.send_message(response)
```

## Conclusion

Adding commands to NeuroDeck requires coordination between UI, protocol, and server layers. The key is ensuring all components are properly connected and handling the command consistently.

Remember:
- ✅ Add UI handler BEFORE `else:` clause
- ✅ Add server handler BEFORE `else:` clause  
- ✅ Update help reference
- ✅ Test thoroughly
- ✅ Handle errors gracefully
- ✅ Clear Python cache when debugging

Follow this cookbook and you'll have robust, well-integrated commands that enhance NeuroDeck's functionality! 🚀