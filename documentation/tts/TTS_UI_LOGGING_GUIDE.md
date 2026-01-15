# TTS UI Logging Guide

## Problem: Logger Output Corrupting Textual UI

When implementing TTS functionality in a Textual-based TUI application, standard Python logging can corrupt the terminal user interface display. This happens because logger output goes directly to stdout/stderr, interfering with Textual's screen management.

## The Two Logging Systems

NeuroDeck uses two distinct logging approaches:

### 1. System Logging (loguru)
- **Purpose**: File logging and stderr output for development
- **Location**: `neurodeck/common/logging.py`
- **Problem**: Goes to stderr → corrupts TUI display
- **Usage**: `logger.info("message")`

### 2. TUI Debug Panel Logging
- **Purpose**: Display debug messages within the TUI interface
- **Location**: UI's `log_debug()` method writes to `#debug_log` RichLog widget
- **Benefit**: Displays properly within TUI without corruption
- **Usage**: `self.log_debug("sender", "message", "system-info")`

## Solution: Log Callback Pattern

### Implementation Steps

1. **Add callback parameter to service constructor**:
```python
def __init__(self, agent_voices: Dict[str, str], log_callback: Optional[Callable[[str, str, str], None]] = None):
    self.log_callback = log_callback
```

2. **Create internal logging method**:
```python
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
            logger.warning(f"TTS log callback failed: {sender}: {message}")
    # For critical errors, always log to file regardless of callback
    elif msg_type == "system-error":
        logger.error(f"{sender}: {message}")
```

3. **Replace all logger calls**:
```python
# Before (corrupts TUI):
logger.info("TTS model loaded successfully")

# After (displays in debug panel):
self._log("TTS", "Model loaded successfully", "system-success")
```

4. **Pass callback during service initialization**:
```python
# In UI class:
self.tts_service = TTSService(agent_voices, log_callback=self.log_debug)
```

### Message Types and Styling

The TUI debug panel supports different message types with appropriate styling:

- `system-info`: Blue, general information
- `system-success`: Green, successful operations  
- `system-warning`: Yellow, warnings and issues
- `system-error`: Red, errors and failures

### Critical Error Handling

Keep critical errors in both systems:
```python
# Audio failures should still go to log files for debugging
except Exception as e:
    self._log("TTS", f"Synthesis failed: {e}", "system-error")  # TUI display
    logger.error(f"Audio synthesis failed: {e}")  # File logging
```

## Benefits

1. **Clean TUI Display**: No more corrupted terminal output
2. **Better User Experience**: Debug info visible in organized panel
3. **Dual Logging**: Critical errors still go to files for debugging
4. **Styled Messages**: Color-coded message types for quick identification
5. **Centralized Control**: Easy to enable/disable debug panel visibility

## Key Takeaway

**Never use standard Python logging directly in Textual applications**. Always route service logging through the TUI's debug panel system to maintain clean display and proper user experience.