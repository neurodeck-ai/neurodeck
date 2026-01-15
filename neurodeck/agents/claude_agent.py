"""
Claude agent implementation for NeuroDeck.

Uses Anthropic's Claude API for AI responses.
"""

import asyncio
import os
import sys
from typing import List, Dict, Any, Optional

try:
    import anthropic
except ImportError:
    print("Error: anthropic package not installed. Run: pip install anthropic")
    sys.exit(1)

from .base_agent import BaseAgent
from ..common.logging import get_logger
from ..common.protocol import LogLevel
from ..common.config import BashToolConfig


class ClaudeAgent(BaseAgent):
    """
    Claude agent implementation using Anthropic's API.
    
    Handles communication with Claude models and converts between
    NeuroDeck's message format and Anthropic's format.
    """
    
    def __init__(self, name: str, host: str, port: int, token: str, config_path: str = "config/agents.ini"):
        super().__init__(name, host, port, token, config_path)
        self.anthropic_client: Optional[anthropic.Anthropic] = None
        
    async def initialize_api_client(self):
        """Initialize Anthropic API client."""
        api_key = os.getenv(self.agent_config.api_key_env)
        if not api_key:
            error_msg = f"API key environment variable {self.agent_config.api_key_env} not set"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        try:
            # Configure timeout to prevent hanging API calls
            api_timeout = getattr(self.agent_config, 'api_timeout', 30)
            self.anthropic_client = anthropic.Anthropic(
                api_key=api_key,
                timeout=api_timeout,  # Anthropic client accepts timeout directly
                max_retries=2  # Allow reasonable retries for transient failures
            )
            self.logger.info(f"Anthropic API client initialized for model {self.agent_config.model} with {api_timeout}s timeout")
            self.logger.info(f"Connected to Anthropic API ({self.agent_config.model})")
            
        except Exception as e:
            error_msg = f"Failed to initialize Anthropic client: {e}"
            self.logger.error(error_msg)
            raise
    
    def get_provider_name(self) -> str:
        """Get the provider name."""
        return "anthropic"
    
    def _convert_to_anthropic_format(self, conversation: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert conversation context to Anthropic's message format.
        
        Anthropic expects:
        - Messages with 'role' and 'content'
        - Alternating user/assistant messages
        - System messages handled separately
        """
        anthropic_messages = []
        
        for msg in conversation:
            role = msg["role"]
            content = msg["content"]
            
            # Skip system messages (handled via system parameter)
            if role == "system":
                continue
            
            # Anthropic uses same role names as OpenAI
            anthropic_messages.append({
                "role": role,
                "content": content
            })
        
        return anthropic_messages
    
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
        """Build tool definitions in Claude format with agent-specific constraints."""
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
            
            description = f"Access files within permitted directories: {paths_str}. Maximum file size: {size_mb}MB. Auto-approved operations: {', '.join(auto_approve)}. Operations requiring approval: {', '.join(require_approval)}."
            
            tools.append({
                "name": "filesystem",
                "description": description,
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["read", "write", "list", "delete"],
                            "description": "The filesystem operation to perform"
                        },
                        "path": {
                            "type": "string",
                            "description": f"File or directory path (must be within: {paths_str})"
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to write (required for write action)"
                        }
                    },
                    "required": ["action", "path"]
                }
            })
        
        if "bash" in self.available_tools:
            # Get agent-specific bash configuration
            bash_config = self._get_bash_tool_config()
            paths_str = ", ".join(bash_config.allowed_paths)
            
            tools.append({
                "name": "bash",
                "description": f"Execute shell commands securely within allowed directories: {paths_str}. Commands are validated for security and may require approval.",
                "input_schema": {
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
            })
        
        if "tts_config" in self.available_tools:
            tools.append({
                "name": "tts_config",
                "description": "Dynamically adjust your text-to-speech voice parameters during conversations to match your personality, mood, or speaking style. Control speech speed, expressiveness, timing, and voice selection.",
                "input_schema": {
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
            })
        
        if "chat_info" in self.available_tools:
            tools.append({
                "name": "chat_info",
                "description": "Get information about the chat environment, discover available participants, and check connection status.",
                "input_schema": {
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
            })
        
        return tools
    
    def should_respond_to_general_question(self, content: str) -> bool:
        """
        Claude-specific logic for responding to general questions.
        
        Claude is helpful and tends to respond to most questions,
        but avoids being overly chatty in group conversations.
        """
        # Skip very short messages that might be casual chat
        if len(content.strip()) < 10:
            return False
        
        # Respond to questions that seem substantive
        question_indicators = [
            "?", "how", "what", "when", "where", "why", "who",
            "can you", "could you", "would you", "please",
            "help", "explain", "analyze", "review", "check"
        ]
        
        content_lower = content.lower()
        for indicator in question_indicators:
            if indicator in content_lower:
                return True
        
        # Be selective about other general messages
        return len(content.strip()) > 30  # Only respond to longer messages
    
    async def generate_response(self, conversation: List[Dict[str, Any]]) -> Optional[str]:
        """
        Generate response using Claude API with tool support.
        
        Args:
            conversation: List of messages in OpenAI format
            
        Returns:
            Generated response text or None if no response
        """
        if not self.anthropic_client:
            await self.send_system_log(LogLevel.ERROR, "Anthropic client not initialized")
            return None
        
        if not conversation:
            return None
        
        try:
            # Convert conversation to Anthropic format
            messages = self._convert_to_anthropic_format(conversation)
            
            if not messages:
                return None
            
            # Get tool definitions
            tools = self._build_tool_definitions()
            system_prompt = self._get_system_prompt()
            
            self.logger.debug(f"Sending request to Claude with {len(messages)} messages")
            await self.send_system_log(LogLevel.INFO, f"Generating response using {self.agent_config.model}")
            
            # Tool execution loop - continues until we get a text response or hit max iterations
            max_tool_iterations = 10
            iteration = 0

            while iteration < max_tool_iterations:
                iteration += 1

                # API call (with tools available for chained calls)
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.anthropic_client.messages.create(
                        model=self.agent_config.model,
                        max_tokens=self.agent_config.max_tokens,
                        temperature=self.agent_config.temperature,
                        system=system_prompt,
                        messages=messages,
                        tools=tools if tools else None
                    )
                )

                # Check if this is a tool use response
                if response.stop_reason == "tool_use":
                    self.logger.info(f"Tool use iteration {iteration}/{max_tool_iterations}")
                    await self.send_system_log(LogLevel.DEBUG, f"Processing tool calls (iteration {iteration})")

                    tool_results = []

                    for content_block in response.content:
                        if content_block.type == "tool_use":
                            # Execute tool
                            try:
                                result = await self.handle_tool_request(
                                    content_block.name,
                                    content_block.input
                                )
                                tool_results.append({
                                    "type": "tool_result",
                                    "tool_use_id": content_block.id,
                                    "content": self.truncate_tool_result(result)
                                })
                            except Exception as e:
                                self.logger.error(f"Tool execution failed: {e}")
                                tool_results.append({
                                    "type": "tool_result",
                                    "tool_use_id": content_block.id,
                                    "content": f"Error: {str(e)}",
                                    "is_error": True
                                })

                    # Append assistant message and tool results for next iteration
                    messages.append({
                        "role": "assistant",
                        "content": response.content
                    })
                    messages.append({
                        "role": "user",
                        "content": tool_results
                    })

                    # Continue loop to get next response
                    continue

                # Not a tool use response - extract text content
                if response.content and len(response.content) > 0:
                    content_text = ""
                    for content_block in response.content:
                        if content_block.type == "text":
                            content_text += content_block.text

                    if content_text.strip():
                        if iteration > 1:
                            self.logger.info(f"Generated response with tools after {iteration} iterations ({len(content_text)} chars)")
                        else:
                            self.logger.info(f"Generated response ({len(content_text)} chars)")
                        return content_text.strip()

                # Response had no text content
                self.logger.warning(f"Empty response from Claude API (stop_reason: {response.stop_reason})")
                return None

            # Hit max iterations
            self.logger.warning(f"Hit max tool iterations ({max_tool_iterations}), forcing text extraction")
            await self.send_system_log(LogLevel.WARNING, f"Tool chain exceeded {max_tool_iterations} iterations")

            # Try to extract any text from the last response
            if response.content:
                content_text = ""
                for content_block in response.content:
                    if content_block.type == "text":
                        content_text += content_block.text
                if content_text.strip():
                    return content_text.strip()

            return None
            
        except anthropic.RateLimitError as e:
            error_msg = f"Rate limit exceeded: {e}"
            self.logger.warning(error_msg)
            await self.send_system_log(LogLevel.WARNING, "Rate limit exceeded, will retry later")
            self._inject_error_feedback("RATE_LIMIT", error_msg)
            return None

        except anthropic.APIStatusError as e:
            error_msg = f"Anthropic API error: {e.status_code} - {e.message}"
            self.logger.error(error_msg)
            await self.send_system_log(LogLevel.ERROR, f"API error: {e.status_code}")
            error_type = getattr(e, 'error', {}).get('type', 'api_error') if hasattr(e, 'error') else 'api_error'
            self._inject_error_feedback(error_type.upper(), error_msg)
            return None

        except anthropic.APIConnectionError as e:
            error_msg = f"Connection error: {e}"
            self.logger.error(error_msg)
            await self.send_system_log(LogLevel.WARNING, "Connection error, retrying...")
            self._inject_error_feedback("CONNECTION_ERROR", error_msg)
            return None

        except anthropic.APITimeoutError as e:
            api_timeout = getattr(self.agent_config, 'api_timeout', 30)
            error_msg = f"Anthropic API timeout after {api_timeout}s: {str(e)}"
            self.logger.error(error_msg)
            await self.send_system_log(LogLevel.WARNING, error_msg)
            self._inject_error_feedback("TIMEOUT", error_msg)
            return None

        except Exception as e:
            error_msg = f"Unexpected error generating response: {e}"
            self.logger.error(error_msg)
            await self.send_system_log(LogLevel.ERROR, f"Response generation failed: {e}")
            error_type = type(e).__name__
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
    """Main entry point for Claude agent process."""
    import argparse
    
    parser = argparse.ArgumentParser(description="NeuroDeck Claude Agent")
    parser.add_argument("--name", required=True, help="Agent name")
    parser.add_argument("--host", required=True, help="Orchestrator host")
    parser.add_argument("--port", type=int, required=True, help="Orchestrator port")
    parser.add_argument("--token", required=True, help="Authentication token")
    parser.add_argument("--config", default="config/agents.ini", help="Configuration file")
    
    args = parser.parse_args()
    
    # Create and run Claude agent
    agent = ClaudeAgent(
        name=args.name,
        host=args.host,
        port=args.port,
        token=args.token,
        config_path=args.config
    )
    
    try:
        asyncio.run(agent.start())
    except KeyboardInterrupt:
        print(f"\nClaude agent {args.name} shutting down...")
    except Exception as e:
        print(f"Claude agent {args.name} crashed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()