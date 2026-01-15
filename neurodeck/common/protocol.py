"""
Line-based JSON protocol for NeuroDeck communication.

CRITICAL: All message encoding MUST use json.dumps() for proper escaping.
Chat content can contain quotes, newlines, backslashes, unicode.
Manual string formatting WILL break the protocol.
"""

import json
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum


class MessageType(Enum):
    """Message types in the NeuroDeck protocol."""
    AUTH = "auth"
    AUTH_RESULT = "auth_result"
    CHAT = "chat"
    SYSTEM_LOG = "system_log"
    COMMAND = "command"
    COMMAND_RESULT = "command_result"
    TOOL_REQUEST = "tool_request"
    TOOL_RESULT = "tool_result"
    TOOL_APPROVAL = "tool_approval"
    TOOL_APPROVAL_REQUEST = "tool_approval_request"
    TOOL_APPROVAL_RESULT = "tool_approval_result"
    CLEAR_CONTEXT = "clear_context"
    CHAT_HISTORY_REQUEST = "chat_history_request"
    CHAT_HISTORY_RESPONSE = "chat_history_response"
    YOLO_STATUS = "yolo_status"
    ERROR = "error"


class LogLevel(Enum):
    """Log levels for system messages."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class ApprovalLevel(Enum):
    """Tool approval levels."""
    ONCE = "once"
    SAME = "same"
    ALL = "all"
    DENY = "deny"


class MessageProtocol:
    """
    Handles encoding and decoding of line-based JSON messages.
    
    CRITICAL SAFETY RULES:
    1. NEVER manually format JSON strings
    2. ALWAYS use json.dumps() for encoding
    3. ALWAYS use json.loads() for decoding
    4. Test with problematic content (quotes, newlines, unicode)
    """
    
    @staticmethod
    def encode_message(msg_dict: Dict[str, Any]) -> bytes:
        """
        Encode message as JSON line with proper escaping.
        
        CRITICAL: ALWAYS use json.dumps() for encoding!
        Chat content can contain quotes, newlines, backslashes, unicode.
        Manual string formatting WILL break the protocol.
        """
        try:
            return f"{json.dumps(msg_dict)}\n".encode('utf-8')
        except (TypeError, ValueError) as e:
            raise ValueError(f"Message encoding failed: {e}")
    
    @staticmethod
    def decode_message(line: bytes) -> Dict[str, Any]:
        """
        Decode JSON line to message dict.
        
        json.loads() automatically handles all JSON unescaping.
        """
        try:
            return json.loads(line.decode('utf-8').strip())
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"Message decoding failed: {e}")
    
    @staticmethod
    def validate_message(message: Dict[str, Any]) -> bool:
        """Validate that message has required fields."""
        return "type" in message and message["type"] in [e.value for e in MessageType]


# Message builders for common message types

def auth_message(token: str, client_id: str) -> Dict[str, Any]:
    """Create authentication message."""
    return {
        "type": MessageType.AUTH.value,
        "token": token,
        "client_id": client_id
    }


def auth_result_message(success: bool, message: str) -> Dict[str, Any]:
    """Create authentication result message."""
    return {
        "type": MessageType.AUTH_RESULT.value,
        "success": success,
        "message": message
    }


def chat_message(sender: str, content: str, timestamp: Optional[str] = None, message_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Create chat message.
    
    CRITICAL: Content can contain ANY characters - quotes, newlines, unicode, etc.
    json.dumps() will handle all escaping automatically.
    """
    if timestamp is None:
        timestamp = datetime.now().isoformat()
    
    if message_id is None:
        message_id = str(uuid.uuid4())
    
    return {
        "type": MessageType.CHAT.value,
        "sender": sender,
        "content": content,
        "timestamp": timestamp,
        "message_id": message_id
    }


def system_log_message(sender: str, level: LogLevel, message: str, timestamp: Optional[str] = None) -> Dict[str, Any]:
    """
    Create system log message (only sent to TUI, not broadcast to agents).
    
    CRITICAL: Message content may contain error details, file paths, etc.
    Must be properly JSON escaped.
    """
    if timestamp is None:
        timestamp = datetime.now().isoformat()
    
    return {
        "type": MessageType.SYSTEM_LOG.value,
        "sender": sender,
        "level": level.value,
        "message": message,
        "timestamp": timestamp
    }


def command_message(cmd: str, **params) -> Dict[str, Any]:
    """Create command message."""
    msg = {
        "type": MessageType.COMMAND.value,
        "cmd": cmd
    }
    msg.update(params)
    return msg


def command_result_message(cmd: str, success: bool, data: Any = None, error: str = None) -> Dict[str, Any]:
    """Create command result message."""
    result = {
        "type": MessageType.COMMAND_RESULT.value,
        "cmd": cmd,
        "success": success
    }
    
    if data is not None:
        result["data"] = data
    if error is not None:
        result["error"] = error
    
    return result


def tool_request_message(agent: str, tool: str, command: str, args: Dict[str, Any], request_id: Optional[str] = None) -> Dict[str, Any]:
    """Create tool execution request message."""
    if request_id is None:
        request_id = str(uuid.uuid4())
    
    return {
        "type": MessageType.TOOL_REQUEST.value,
        "agent": agent,
        "tool": tool,
        "command": command,
        "args": args,
        "request_id": request_id
    }


def tool_approval_request_message(agent: str, tool: str, command: str, args: Dict[str, Any], description: str, request_id: str) -> Dict[str, Any]:
    """Create tool approval request message (sent to TUI for user approval)."""
    return {
        "type": MessageType.TOOL_APPROVAL_REQUEST.value,
        "agent": agent,
        "tool": tool,
        "command": command,
        "args": args,
        "description": description,
        "request_id": request_id
    }


def tool_approval_message(agent: str, approval: ApprovalLevel, request_id: str) -> Dict[str, Any]:
    """Create tool approval response message (from TUI)."""
    return {
        "type": MessageType.TOOL_APPROVAL.value,
        "agent": agent,
        "approval": approval.value,
        "request_id": request_id
    }


def tool_approval_result_message(agent: str, approval: ApprovalLevel, request_id: str) -> Dict[str, Any]:
    """Create tool approval result message (sent to agent)."""
    return {
        "type": MessageType.TOOL_APPROVAL_RESULT.value,
        "agent": agent,
        "approval": approval.value,
        "request_id": request_id
    }


def tool_result_message(request_id: str, success: bool, result: Any = None, error: str = None) -> Dict[str, Any]:
    """Create tool execution result message."""
    msg = {
        "type": MessageType.TOOL_RESULT.value,
        "request_id": request_id,
        "success": success
    }
    if result is not None:
        msg["result"] = result
    if error is not None:
        msg["error"] = error
    return msg


def clear_context_message(agent: str) -> Dict[str, Any]:
    """Create context clear message."""
    return {
        "type": MessageType.CLEAR_CONTEXT.value,
        "agent": agent
    }


def chat_history_request_message(limit: int = 20) -> Dict[str, Any]:
    """Create chat history request message."""
    return {
        "type": MessageType.CHAT_HISTORY_REQUEST.value,
        "limit": limit
    }


def chat_history_response_message(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create chat history response message."""
    return {
        "type": MessageType.CHAT_HISTORY_RESPONSE.value,
        "messages": messages
    }


def yolo_status_message(enabled: bool, message: str) -> Dict[str, Any]:
    """Create YOLO mode status message."""
    return {
        "type": MessageType.YOLO_STATUS.value,
        "enabled": enabled,
        "message": message
    }


def error_message(message: str, details: Optional[str] = None) -> Dict[str, Any]:
    """
    Create error message.
    
    CRITICAL: Error messages often contain user input, file paths, stack traces.
    Must be properly JSON escaped.
    """
    msg = {
        "type": MessageType.ERROR.value,
        "message": message
    }
    
    if details is not None:
        msg["details"] = details
    
    return msg


# Test content that MUST be handled correctly by the protocol
TEST_PROBLEMATIC_CONTENT = [
    'Message with "quotes"',
    'Multi-line\nmessage\nwith\nbreaks',
    'Backslash \\ characters',
    'Code: if (x == "test") { return "value"; }',
    'Unicode: 你好 🚀 émojis',
    'JSON-like: {"fake": "json", "data": [1,2,3]}',
    'Complex: "He said \\"Hello\\nWorld\\"" \\backslash',
    'File path: C:\\Users\\name\\file.txt',
    'SQL injection: \'; DROP TABLE users; --',
    'XSS attempt: <script>alert("hack")</script>'
]


def test_protocol_safety():
    """Test that the protocol correctly handles all problematic content."""
    print("Testing protocol safety with problematic content...")
    
    for content in TEST_PROBLEMATIC_CONTENT:
        try:
            # Create chat message with problematic content
            msg = chat_message("test_user", content)
            
            # Encode using protocol
            encoded = MessageProtocol.encode_message(msg)
            
            # Decode back
            decoded = MessageProtocol.decode_message(encoded)
            
            # Verify content is identical
            assert decoded["content"] == content, f"Content mismatch for: {content}"
            assert decoded["sender"] == "test_user"
            assert decoded["type"] == "chat"
            
            print(f"✅ Safely handled: {repr(content)}")
            
        except Exception as e:
            print(f"❌ Failed to handle: {repr(content)} - Error: {e}")
            raise
    
    print("✅ All problematic content handled safely!")


if __name__ == "__main__":
    test_protocol_safety()