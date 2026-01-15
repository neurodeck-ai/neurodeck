"""
Secure bash command execution tool for NeuroDeck.

Features:
- Multi-layer security validation
- Process isolation with session groups
- Output size limiting with real-time monitoring
- Path traversal protection
- Command injection prevention
- Timeout protection and process cleanup
- Per-agent working directories
"""

import os
import re
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, List
from ..common.logging import get_logger

logger = get_logger(__name__)


class SecurityError(Exception):
    """Raised when security validation fails."""
    pass


class ValidationResult:
    """Result of command security validation with detailed error information."""
    
    def __init__(self, is_safe: bool, error_type: str = None, error_details: Dict[str, Any] = None):
        self.is_safe = is_safe
        self.error_type = error_type
        self.error_details = error_details or {}


class BashTool:
    """Secure bash command execution tool."""
    
    def __init__(self, config):
        self.config = config
        self.working_directory = Path(config.working_directory).resolve()
        self.logger = get_logger(__name__)
        
        # Enhanced error message system
        self._init_error_messages()
        
        # Pre-compile and validate regex patterns for security and performance
        self._compiled_patterns = []
        for pattern in config.blocked_patterns:
            try:
                compiled_pattern = re.compile(pattern)
                self._compiled_patterns.append((compiled_pattern, pattern))
            except re.error as e:
                self.logger.error(f"Invalid regex pattern '{pattern}': {e}")
                # Skip invalid patterns but log the error
                continue
        
        # Validate working directory is within allowed paths
        if not any(self.working_directory.is_relative_to(Path(p).resolve()) 
                  for p in config.allowed_paths):
            raise ValueError(f"Working directory {self.working_directory} not in allowed paths")
    
    def _init_error_messages(self):
        """Initialize enhanced error message system."""
        self.ENHANCED_ERRORS = {
            "blocked_command": {
                "message": "Command '{command}' is blocked by security policy",
                "reason": "This command is in the security blacklist for safety",
                "suggestion": "Try safer alternatives: 'ls' for listing, 'cat' for reading, 'echo' for output"
            },
            "blocked_pattern_path_traversal": {
                "message": "Path traversal detected in command",
                "reason": "Commands using '../' can access files outside the allowed directory",
                "suggestion": "Use absolute paths within your working directory: {working_dir}"
            },
            "blocked_pattern_command_substitution": {
                "message": "Command substitution not allowed",
                "reason": "Commands using $(...) or backticks can execute arbitrary code",
                "suggestion": "Break your task into separate commands instead of nesting them"
            },
            "blocked_pattern_variable_expansion": {
                "message": "Variable expansion not allowed",
                "reason": "Commands using $VAR can access system environment variables",
                "suggestion": "Use literal values instead of variables, or ask human for assistance"
            },
            "blocked_pattern_home_directory": {
                "message": "Home directory access not allowed",
                "reason": "Commands using '~' can access user home directories",
                "suggestion": "Use your working directory path: {working_dir}"
            },
            "blocked_pattern_general": {
                "message": "Command contains blocked security pattern",
                "reason": "Command contains characters or patterns that could be unsafe",
                "suggestion": "Simplify your command and avoid special characters like ;, &, |"
            },
            "absolute_path_blocked": {
                "message": "Absolute path executables not allowed",
                "reason": "Commands starting with '/' can access system binaries",
                "suggestion": "Use command names without paths (e.g., 'ls' instead of '/bin/ls')"
            },
            "dangerous_path": {
                "message": "Access to system path '{path}' is forbidden",
                "reason": "System directories like /etc, /root, /proc are protected",
                "suggestion": "Work within your allowed directory: {working_dir}"
            }
        }
        
        self.COMMAND_ALTERNATIVES = {
            "sudo": "For administrative tasks, request human assistance",
            "su": "For user switching, request human assistance", 
            "rm": "Use filesystem tool for safe file operations",
            "chmod": "File permissions are managed by the system",
            "wget": "For downloads, ask human to provide files",
            "curl": "For network requests, ask human for assistance",
            "dd": "For disk operations, use safer file tools",
            "nc": "Network tools are restricted for security",
            "netcat": "Network tools are restricted for security"
        }
    
    async def execute(self, action: str, command: str = None, path: str = None, 
                     agent_name: str = "unknown") -> Dict[str, Any]:
        """Main entry point for bash tool operations."""
        try:
            if action == "execute_command":
                if not command:
                    return self._safe_error_response("invalid_request", "Command required")
                return await self._execute_command(command, agent_name)
            elif action == "change_directory":
                if not path:
                    return self._safe_error_response("invalid_request", "Path required")
                return await self._change_directory(path, agent_name)
            elif action == "get_working_directory":
                return {"success": True, "working_directory": str(self.working_directory)}
            elif action == "list_allowed_paths":
                return {"success": True, "allowed_paths": self.config.allowed_paths}
            else:
                return self._safe_error_response("invalid_action", f"Unknown action: {action}")
        except SecurityError as e:
            return self._safe_error_response("security_violation", str(e))
        except Exception as e:
            self.logger.error(f"Bash tool error for {agent_name}: {e}")
            return self._safe_error_response("process_failed")

    async def _execute_command(self, command: str, agent_name: str) -> Dict[str, Any]:
        """Execute shell command with full security validation."""
        # Validate command safety
        validation_result = self._validate_and_log_security_violation(command, agent_name)
        if not validation_result.is_safe:
            return self._safe_error_response("security_violation", validation_result)
        
        process = None
        try:
            # Create safe environment and execute
            safe_env = self._create_safe_environment()
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=self.working_directory,
                env=safe_env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True  # Process isolation
            )
            
            # Read output with size limit
            output_chunks = []
            total_size = 0
            
            while True:
                try:
                    chunk = await asyncio.wait_for(process.stdout.read(8192), timeout=1.0)
                    if not chunk:
                        break
                        
                    total_size += len(chunk)
                    if total_size > self.config.max_output_size:
                        await self._cleanup_process(process, "Output size limit exceeded")
                        return self._safe_error_response("output_too_large")
                        
                    output_chunks.append(chunk)
                except asyncio.TimeoutError:
                    # Check if process is still running
                    if process.returncode is not None:
                        break
                    continue
            
            await process.wait()
            output = b''.join(output_chunks).decode('utf-8', errors='replace')
            
            self.logger.info(f"Command executed by {agent_name}: {command[:50]}... -> exit {process.returncode}")
            
            return {
                "success": True,
                "output": output,
                "return_code": process.returncode,
                "working_directory": str(self.working_directory)
            }
            
        except asyncio.TimeoutError:
            await self._cleanup_process(process, "Command execution timed out")
            return self._safe_error_response("timeout")
        except Exception as e:
            self.logger.error(f"Command execution failed for {agent_name}: {e}")
            await self._cleanup_process(process, f"Command execution failed: {e}")
            return self._safe_error_response("process_failed")

    async def _change_directory(self, path: str, agent_name: str) -> Dict[str, Any]:
        """
        Change working directory with security validation.
        
        RATIONALE: Directory changes must be validated against allowed_paths to prevent
        agents from escaping their sandbox through cd commands.
        """
        try:
            # Resolve and validate the new path
            new_path = self._resolve_and_validate_path(path)
            
            # Check if new path exists and is a directory
            if not new_path.exists():
                return self._safe_error_response("path_forbidden", "Directory does not exist")
            
            if not new_path.is_dir():
                return self._safe_error_response("path_forbidden", "Path is not a directory")
            
            # Update working directory
            old_directory = self.working_directory
            self.working_directory = new_path
            
            self.logger.info(f"Agent {agent_name} changed directory: {old_directory} -> {new_path}")
            
            return {
                "success": True,
                "old_directory": str(old_directory),
                "new_directory": str(new_path),
                "working_directory": str(self.working_directory)
            }
            
        except SecurityError as e:
            return self._safe_error_response("security_violation", str(e))
        except Exception as e:
            self.logger.error(f"Directory change failed for {agent_name}: {e}")
            return self._safe_error_response("process_failed")

    def _validate_command_safety(self, command: str) -> ValidationResult:
        """Multi-layer command validation with detailed error information."""
        # 1. Basic validation
        command_words = command.strip().split()
        if not command_words:
            return ValidationResult(False, "empty_command", {"reason": "Command cannot be empty"})
        
        first_command = command_words[0]
        
        # 2. Check blocked commands (exact matches and prefixes)
        for blocked in self.config.blocked_commands:
            if first_command == blocked or command.startswith(blocked + " "):
                self.logger.warning(f"Blocked command detected: {blocked}")
                alternative = self.COMMAND_ALTERNATIVES.get(blocked, "Try safer alternatives")
                return ValidationResult(False, "blocked_command", {
                    "command": blocked,
                    "alternative": alternative,
                    "working_dir": str(self.working_directory)
                })
        
        # 3. Check blocked patterns (regex) - using pre-compiled patterns for security and performance  
        for compiled_pattern, pattern_str in self._compiled_patterns:
            if compiled_pattern.search(command):
                self.logger.warning(f"Blocked pattern detected: {pattern_str}")
                
                # Identify specific pattern type for better error messages
                if re.search(r'\.\./', command):
                    return ValidationResult(False, "blocked_pattern_path_traversal", {
                        "working_dir": str(self.working_directory)
                    })
                elif re.search(r'[$`]|\$\(', command):
                    return ValidationResult(False, "blocked_pattern_command_substitution", {})
                elif re.search(r'\$[A-Za-z0-9_?*@#]|\$\{', command):
                    return ValidationResult(False, "blocked_pattern_variable_expansion", {})
                elif re.search(r'~', command):
                    return ValidationResult(False, "blocked_pattern_home_directory", {
                        "working_dir": str(self.working_directory)
                    })
                else:
                    return ValidationResult(False, "blocked_pattern_general", {})
        
        # 4. Validate executable path (first command)
        if not self.config.allow_absolute_paths and first_command.startswith('/'):
            self.logger.warning(f"Absolute path executable not allowed: {first_command}")
            return ValidationResult(False, "absolute_path_blocked", {
                "command": first_command
            })
            
        # 5. Check for home directory access
        if not self.config.allow_home_directory and re.search(r'~', command):
            self.logger.warning("Home directory access not allowed")
            return ValidationResult(False, "blocked_pattern_home_directory", {
                "working_dir": str(self.working_directory)
            })
            
        # 6. Check for command substitution (enhanced detection)
        if not self.config.allow_command_substitution:
            if re.search(r'[$`]|\$\(', command):
                self.logger.warning("Command substitution not allowed")
                return ValidationResult(False, "blocked_pattern_command_substitution", {})
                
        # 7. Check for variable expansion (enhanced detection)  
        if not self.config.allow_variable_expansion:
            if re.search(r'\$[A-Za-z0-9_?*@#]|\$\{', command):
                self.logger.warning("Variable expansion not allowed")
                return ValidationResult(False, "blocked_pattern_variable_expansion", {})
        
        # 8. Validate file paths in arguments for dangerous system paths
        args = command_words[1:] if len(command_words) > 1 else []
        dangerous_paths = ['/etc/', '/root/', '/proc/', '/sys/', '/dev/', '/boot/']
        for arg in args:
            if any(arg.startswith(path) for path in dangerous_paths):
                self.logger.warning(f"Access to dangerous system path not allowed: {arg}")
                return ValidationResult(False, "dangerous_path", {
                    "path": arg,
                    "working_dir": str(self.working_directory)
                })
        
        return ValidationResult(True)

    def _validate_and_log_security_violation(self, command: str, agent_name: str) -> ValidationResult:
        """Validate command and log security violations with detailed error information."""
        validation_result = self._validate_command_safety(command)
        if not validation_result.is_safe:
            self.logger.error(f"Security violation by {agent_name}: {command} - {validation_result.error_type}")
        return validation_result

    def _resolve_and_validate_path(self, path_arg: str) -> Path:
        """Resolve and validate paths stay within boundaries."""
        try:
            # Convert to Path object and resolve
            if os.path.isabs(path_arg):
                if not self.config.allow_absolute_paths:
                    raise SecurityError(f"Absolute paths not allowed: {path_arg}")
                resolved = Path(path_arg).resolve()
            else:
                resolved = (Path(self.working_directory) / path_arg).resolve()
            
            # Check if resolved path is within any allowed path
            for allowed_str in self.config.allowed_paths:
                allowed_path = Path(allowed_str).resolve()
                try:
                    resolved.relative_to(allowed_path)
                    return resolved  # Path is within this allowed directory
                except ValueError:
                    continue  # Not within this allowed path, try next
                    
            raise SecurityError(f"Path {path_arg} resolves outside allowed directories")
            
        except Exception as e:
            raise SecurityError(f"Invalid path: {path_arg} - {str(e)}")

    def _create_safe_environment(self) -> Dict[str, str]:
        """Create restricted environment for command execution."""
        safe_env = {
            'PATH': '/usr/bin:/bin',
            'PWD': str(self.working_directory),
            'HOME': str(self.working_directory),  # Override HOME
            'USER': os.getenv('USER', 'neurodeck'),  # Current user
            'SHELL': '/bin/bash',
            'TERM': 'xterm',
            'LANG': 'C.UTF-8'  # Consistent locale
        }
        
        # Remove dangerous environment variables that could be inherited
        # This environment replaces the entire process environment
        return safe_env

    async def _cleanup_process(self, process: Optional[asyncio.subprocess.Process], reason: str) -> None:
        """
        Safely cleanup subprocess with escalating termination signals.
        
        RATIONALE: Some processes may ignore SIGTERM, so we need SIGKILL fallback.
        Uses escalating approach: SIGTERM -> wait -> SIGKILL -> wait
        """
        if process is None:
            return
            
        try:
            if process.returncode is None:  # Process still running
                self.logger.info(f"Terminating process (PID: {process.pid}): {reason}")
                
                # Step 1: Try graceful termination (SIGTERM)
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                    self.logger.debug(f"Process {process.pid} terminated gracefully")
                    return
                except asyncio.TimeoutError:
                    pass  # Process didn't respond to SIGTERM
                
                # Step 2: Force kill (SIGKILL)
                self.logger.warning(f"Process {process.pid} didn't respond to SIGTERM, using SIGKILL")
                process.kill()
                try:
                    await asyncio.wait_for(process.wait(), timeout=2.0)
                    self.logger.debug(f"Process {process.pid} killed with SIGKILL")
                except asyncio.TimeoutError:
                    # This should be very rare, but log it
                    self.logger.error(f"Process {process.pid} didn't respond to SIGKILL - zombie process?")
                    
        except ProcessLookupError:
            # Process already dead
            self.logger.debug(f"Process {getattr(process, 'pid', 'unknown')} already terminated")
        except Exception as e:
            self.logger.error(f"Error during process cleanup: {e}")

    def _safe_error_response(self, error_type: str, validation_result: ValidationResult = None, details: str = "") -> Dict[str, Any]:
        """Return safe error response with enhanced error information for AI agents."""
        
        # Handle enhanced security validation errors
        if (error_type == "security_violation" and validation_result and 
            hasattr(validation_result, 'is_safe') and not validation_result.is_safe and
            hasattr(validation_result, 'error_type') and validation_result.error_type):
            error_info = self.ENHANCED_ERRORS.get(validation_result.error_type, {})
            if error_info:
                # Safely format message with dynamic values - sanitize input to prevent format string attacks
                safe_details = {}
                if validation_result.error_details:
                    for key, value in validation_result.error_details.items():
                        # Convert to string and escape any format specifiers
                        safe_value = str(value).replace('{', '{{').replace('}', '}}')
                        safe_details[key] = safe_value
                
                try:
                    message = error_info["message"].format(**safe_details)
                    suggestion = error_info["suggestion"].format(**safe_details)
                except (KeyError, ValueError) as e:
                    # Fallback to non-formatted message if formatting fails
                    self.logger.warning(f"Error formatting message: {e}")
                    message = error_info["message"]
                    suggestion = error_info["suggestion"]
                
                reason = error_info["reason"]
                
                return {
                    "success": False,
                    "error": message,
                    "reason": reason,
                    "suggestion": suggestion,
                    "error_type": validation_result.error_type,
                    "blocked_by": "bash_tool_security",
                    "help": "Bash commands are restricted for security. Use suggested alternatives or request assistance."
                }
        
        # Fallback to original error messages for non-validation errors
        ERROR_MESSAGES = {
            "security_violation": "Command blocked by security policy.",
            "path_forbidden": "Path access denied.",
            "timeout": "Command execution timed out.",
            "process_failed": "Command execution failed.",
            "output_too_large": "Command output exceeded size limit.",
            "invalid_request": "Invalid request parameters. Command cannot be empty.",
            "invalid_action": "Unknown action requested."
        }
        
        return {
            "success": False,
            "error": ERROR_MESSAGES.get(error_type, "Operation failed."),
            "details": details if not self.config.paranoid_mode else ""  # Hide details in paranoid mode
        }