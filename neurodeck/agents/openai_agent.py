"""
OpenAI agent implementation for NeuroDeck.

Uses OpenAI's ChatGPT API for AI responses.
"""

import asyncio
import json
import os
import sys
from typing import List, Dict, Any, Optional

try:
    import openai
except ImportError:
    print("Error: openai package not installed. Run: pip install openai")
    sys.exit(1)

from .base_agent import BaseAgent
from ..common.logging import get_logger
from ..common.protocol import LogLevel
from ..common.config import BashToolConfig


class OpenAIAgent(BaseAgent):
    """
    OpenAI agent implementation using ChatGPT API.
    
    Handles communication with GPT models and converts between
    NeuroDeck's message format and OpenAI's format.
    """
    
    def __init__(self, name: str, host: str, port: int, token: str, config_path: str = "config/agents.ini"):
        super().__init__(name, host, port, token, config_path)
        self.openai_client: Optional[openai.AsyncOpenAI] = None
        
    async def initialize_api_client(self):
        """Initialize OpenAI API client."""
        api_key = os.getenv(self.agent_config.api_key_env)
        if not api_key:
            error_msg = f"API key environment variable {self.agent_config.api_key_env} not set"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        self.logger.info(f"Agent {self.name} received {self.agent_config.api_key_env}: {api_key[:10]}...")
        
        try:
            # Configure timeout to prevent hanging API calls
            import httpx
            
            api_timeout = getattr(self.agent_config, 'api_timeout', 30)
            self.openai_client = openai.AsyncOpenAI(
                api_key=api_key,
                base_url=self.agent_config.api_endpoint,
                timeout=httpx.Timeout(api_timeout, connect=10.0),  # API timeout + 10s connect timeout
                max_retries=2  # Allow reasonable retries for transient failures
            )
            self.logger.info(f"OpenAI API client initialized for model {self.agent_config.model} with {api_timeout}s timeout")
            self.logger.info(f"Connected to OpenAI API ({self.agent_config.model})")
            
        except Exception as e:
            error_msg = f"Failed to initialize OpenAI client: {e}"
            self.logger.error(error_msg)
            raise
    
    def get_provider_name(self) -> str:
        """Get the provider name."""
        return "openai"
    
    def _convert_to_openai_format(self, conversation: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert conversation context to OpenAI's message format.
        
        OpenAI expects:
        - Messages with 'role' and 'content'
        - System messages can be included
        - Supports user, assistant, system roles
        """
        openai_messages = []
        
        # Add system prompt as first message
        system_prompt = self._get_system_prompt()
        if system_prompt:
            openai_messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        # Add conversation messages
        for msg in conversation:
            role = msg["role"]
            content = msg["content"]
            
            # OpenAI uses same role names
            openai_messages.append({
                "role": role,
                "content": content
            })
        
        return openai_messages
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        base_prompt = self.agent_config.system_prompt
        
        # Add context about available tools
        if self.available_tools:
            tools_list = ", ".join(sorted(self.available_tools))
            tool_prompt = f"\n\nYou have access to these tools: {tools_list}"
            base_prompt += tool_prompt
        
        return base_prompt

    def _build_tool_definitions(self) -> List[Dict[str, Any]]:
        """Build tool definitions in OpenAI format with agent-specific constraints."""
        tools = []
        
        if "filesystem" in self.available_tools:
            # Get agent-specific filesystem configuration
            fs_config = self.get_agent_tool_config("filesystem")
            
            # Extract configuration values with defaults
            allowed_paths = fs_config.get("allowed_paths", ["/tmp"])
            max_file_size = fs_config.get("max_file_size", 10485760)
            auto_approve = fs_config.get("auto_approve_operations", ["read", "list"])
            require_approval = fs_config.get("require_approval_operations", ["write", "delete"])
            
            # Build enhanced description with agent-specific constraints
            paths_str = ", ".join(allowed_paths)
            size_mb = max_file_size // (1024 * 1024)
            
            description = (
                f"Access files within permitted directories: {paths_str}. "
                f"Maximum file size: {size_mb}MB. "
                f"Auto-approved operations: {', '.join(auto_approve)}. "
                f"Operations requiring approval: {', '.join(require_approval)}. "
                "IMPORTANT: Always READ a file before using WRITE to modify it. "
                "Use APPEND to add content without overwriting. "
                "WRITE replaces entire file contents."
            )

            tools.append({
                "type": "function",
                "function": {
                    "name": "filesystem",
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": ["read", "write", "append", "list", "delete"],
                                "description": "The filesystem operation: read (view file), write (replace entire file - read first!), append (add to end of file), list (directory contents), delete (remove file)"
                            },
                            "path": {
                                "type": "string",
                                "description": f"File or directory path (must be within: {paths_str})"
                            },
                            "content": {
                                "type": "string",
                                "description": "Content for write (replaces file) or append (adds to end) actions"
                            }
                        },
                        "required": ["action", "path"]
                    }
                }
            })
        
        if "bash" in self.available_tools:
            # Get agent-specific bash configuration
            bash_config = self._get_bash_tool_config()
            paths_str = ", ".join(bash_config.allowed_paths)
            
            tools.append({
                "type": "function",
                "function": {
                    "name": "bash",
                    "description": f"Execute shell commands securely within allowed directories: {paths_str}. Commands are validated for security and may require approval.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": ["execute_command", "change_directory", "get_working_directory", "list_allowed_paths"],
                                "description": "The bash operation to perform"
                            },
                            "command": {
                                "type": "string",
                                "description": "Shell command to execute (required for execute_command action)"
                            },
                            "path": {
                                "type": "string",
                                "description": f"Directory path for change_directory (must be within: {paths_str})"
                            }
                        },
                        "required": ["action"]
                    }
                }
            })
        
        if "tts_config" in self.available_tools:
            tools.append({
                "type": "function",
                "function": {
                    "name": "tts_config",
                    "description": "Dynamically adjust your text-to-speech voice parameters during conversations to match your personality, mood, or speaking style. Control speech speed, expressiveness, timing, and voice selection.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": ["get_voice_settings", "set_voice_settings", "list_available_voices", "reset_voice_settings", "get_help"],
                                "description": "The TTS operation to perform"
                            },
                            "length_scale": {
                                "type": "number",
                                "minimum": 0.5,
                                "maximum": 2.0,
                                "description": "Speech speed: 0.5=2x faster, 1.0=normal, 2.0=2x slower (for set_voice_settings)"
                            },
                            "inference_noise_scale": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 1.5,
                                "description": "Expressiveness: 0.0=robotic, 0.667=natural, 1.5=very expressive (for set_voice_settings)"
                            },
                            "inference_noise_scale_dp": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 1.5,
                                "description": "Timing variation: 0.0=mechanical, 0.8=natural, 1.5=dramatic pauses (for set_voice_settings)"
                            },
                            "voice_id": {
                                "type": "string",
                                "description": "Voice ID to switch to (for set_voice_settings). Use list_available_voices to see options"
                            }
                        },
                        "required": ["action"]
                    }
                }
            })
        
        if "chat_info" in self.available_tools:
            tools.append({
                "type": "function",
                "function": {
                    "name": "chat_info",
                    "description": "Get information about the chat environment, discover available participants, and check connection status.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": ["list_participants", "get_chat_status", "get_my_info", "get_participant_info"],
                                "description": "The chat info operation to perform"
                            },
                            "participant_name": {
                                "type": "string",
                                "description": "Name of participant to get info about (required for get_participant_info)"
                            }
                        },
                        "required": ["action"]
                    }
                }
            })
        
        return tools
    
    
    async def generate_response(self, conversation: List[Dict[str, Any]]) -> Optional[str]:
        """
        Generate response using OpenAI API with tool support.
        
        Args:
            conversation: List of messages in OpenAI format
            
        Returns:
            Generated response text or None if no response
        """
        if not self.openai_client:
            await self.send_system_log(LogLevel.ERROR, "OpenAI client not initialized")
            return None
        
        if not conversation:
            return None
        
        try:
            # Convert conversation to OpenAI format
            messages = self._convert_to_openai_format(conversation)
            tools = self._build_tool_definitions()
            
            if not messages:
                return None
            
            self.logger.debug(f"Sending request to OpenAI with {len(messages)} messages")
            await self.send_system_log(LogLevel.INFO, f"Generating response using {self.agent_config.model}")
            
            # Tool execution loop - continues until we get a text response or hit max iterations
            max_tool_iterations = 10
            iteration = 0

            while iteration < max_tool_iterations:
                iteration += 1

                # API call (with tools available for chained calls)
                response = await self.openai_client.chat.completions.create(
                    model=self.agent_config.model,
                    messages=messages,
                    max_tokens=self.agent_config.max_tokens,
                    temperature=self.agent_config.temperature,
                    tools=tools if tools else None,
                    tool_choice="auto" if tools else None
                )

                message = response.choices[0].message

                # Check if this is a tool call response
                if message.tool_calls:
                    self.logger.info(f"Tool use iteration {iteration}/{max_tool_iterations}")
                    await self.send_system_log(LogLevel.DEBUG, f"Processing tool calls (iteration {iteration})")

                    # Add assistant message with tool calls
                    messages.append({
                        "role": "assistant",
                        "content": message.content,
                        "tool_calls": [
                            {
                                "id": tool_call.id,
                                "type": tool_call.type,
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": tool_call.function.arguments
                                }
                            } for tool_call in message.tool_calls
                        ]
                    })

                    # Execute each tool call
                    for tool_call in message.tool_calls:
                        try:
                            # Parse arguments
                            args = json.loads(tool_call.function.arguments)

                            # Execute tool
                            result = await self.handle_tool_request(
                                tool_call.function.name,
                                args
                            )

                            # Add tool result
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": tool_call.function.name,
                                "content": self.truncate_tool_result(result)
                            })
                        except Exception as e:
                            self.logger.error(f"Tool execution failed: {e}")
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": tool_call.function.name,
                                "content": f"Error: {str(e)}"
                            })

                    # Continue loop to get next response
                    continue

                # Not a tool call response - extract text content
                if message.content and message.content.strip():
                    if iteration > 1:
                        self.logger.info(f"Generated response with tools after {iteration} iterations ({len(message.content)} chars)")
                    else:
                        self.logger.info(f"Generated response ({len(message.content)} chars)")
                    return message.content.strip()

                # Response had no text content
                self.logger.warning(f"Empty response from OpenAI API (finish_reason: {response.choices[0].finish_reason})")
                return None

            # Hit max iterations
            self.logger.warning(f"Hit max tool iterations ({max_tool_iterations}), forcing text extraction")
            await self.send_system_log(LogLevel.WARNING, f"Tool chain exceeded {max_tool_iterations} iterations")

            # Try to extract any text from the last response
            if message.content and message.content.strip():
                return message.content.strip()

            return None
            
        except openai.RateLimitError as e:
            error_msg = f"Rate limit exceeded: {str(e)}"
            self.logger.error(error_msg)
            await self.send_system_log(LogLevel.WARNING, error_msg)
            self._inject_error_feedback("RATE_LIMIT", error_msg)
            return None

        except openai.APIStatusError as e:
            # Capture full error details
            error_details = f"OpenAI API error: Status={e.status_code}, Message={getattr(e, 'message', str(e))}, Body={getattr(e, 'body', 'N/A')}"
            self.logger.error(error_details)
            await self.send_system_log(LogLevel.ERROR, error_details)
            # Extract error code for better feedback
            error_code = getattr(e, 'body', {}).get('error', {}).get('code', 'api_error') if isinstance(getattr(e, 'body', None), dict) else 'api_error'
            self._inject_error_feedback(error_code.upper(), error_details)
            return None

        except openai.APIConnectionError as e:
            error_msg = f"Connection error: {str(e)}"
            self.logger.error(error_msg)
            await self.send_system_log(LogLevel.WARNING, error_msg)
            self._inject_error_feedback("CONNECTION_ERROR", error_msg)
            return None

        except openai.APITimeoutError as e:
            api_timeout = getattr(self.agent_config, 'api_timeout', 30)
            error_msg = f"OpenAI API timeout after {api_timeout}s: {str(e)}"
            self.logger.error(error_msg)
            await self.send_system_log(LogLevel.WARNING, error_msg)
            self._inject_error_feedback("TIMEOUT", error_msg)
            return None

        except Exception as e:
            # Enhanced error capture with type and full details
            error_type = type(e).__name__
            error_msg = f"Unexpected {error_type} in OpenAI response generation: {str(e)}"
            self.logger.error(error_msg)
            import traceback
            tb = traceback.format_exc()
            self.logger.error(f"Full traceback:\n{tb}")
            await self.send_system_log(LogLevel.ERROR, error_msg)
            self._inject_error_feedback(error_type.upper(), error_msg)
            return None

    def _get_bash_tool_config(self) -> BashToolConfig:
        """Get bash tool configuration for this agent."""
        from ..common.config import ConfigManager, BashToolConfig
        
        # This would normally be done through tool executor, but for schema generation
        # we need a simplified version
        try:
            config_manager = ConfigManager(self.config_path)
            tool_configs = config_manager.load_tool_configs()
            agent_overrides = config_manager.load_agent_tool_overrides()
            
            # Get base config
            base_config = tool_configs.get('bash', BashToolConfig(name='bash'))
            
            # Apply agent-specific overrides
            if self.name in agent_overrides and 'bash' in agent_overrides[self.name]:
                override = agent_overrides[self.name]['bash']
                # Simple merge - override non-None values
                merged_config = BashToolConfig(
                    name='bash',
                    working_directory=override.working_directory or base_config.working_directory,
                    allowed_paths=override.allowed_paths or base_config.allowed_paths,
                    max_output_size=override.max_output_size or base_config.max_output_size,
                    blocked_commands=override.blocked_commands or base_config.blocked_commands,
                    blocked_patterns=override.blocked_patterns or base_config.blocked_patterns,
                    allow_absolute_paths=override.allow_absolute_paths if override.allow_absolute_paths is not None else base_config.allow_absolute_paths,
                    allow_home_directory=override.allow_home_directory if override.allow_home_directory is not None else base_config.allow_home_directory,
                    allow_command_substitution=override.allow_command_substitution if override.allow_command_substitution is not None else base_config.allow_command_substitution,
                    allow_variable_expansion=override.allow_variable_expansion if override.allow_variable_expansion is not None else base_config.allow_variable_expansion,
                    auto_approve_operations=override.auto_approve_operations or base_config.auto_approve_operations,
                    require_approval_operations=override.require_approval_operations or base_config.require_approval_operations
                )
                return merged_config
            
            return base_config
        except Exception:
            # Fallback to default config for schema generation
            return BashToolConfig(
                name='bash',
                allowed_paths=['/tmp/neurodeck'],
                working_directory='/tmp/neurodeck'
            )


def main():
    """Main entry point for OpenAI agent process."""
    import argparse
    
    parser = argparse.ArgumentParser(description="NeuroDeck OpenAI Agent")
    parser.add_argument("--name", required=True, help="Agent name")
    parser.add_argument("--host", required=True, help="Orchestrator host")
    parser.add_argument("--port", type=int, required=True, help="Orchestrator port")
    parser.add_argument("--token", required=True, help="Authentication token")
    parser.add_argument("--config", default="config/agents.ini", help="Configuration file")
    
    args = parser.parse_args()
    
    # Create and run OpenAI agent
    agent = OpenAIAgent(
        name=args.name,
        host=args.host,
        port=args.port,
        token=args.token,
        config_path=args.config
    )
    
    try:
        asyncio.run(agent.start())
    except KeyboardInterrupt:
        print(f"\nOpenAI agent {args.name} shutting down...")
    except Exception as e:
        print(f"OpenAI agent {args.name} crashed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()