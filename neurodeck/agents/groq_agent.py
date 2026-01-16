"""
Groq agent implementation for NeuroDeck.

Uses Groq's API for AI responses (including Kimi and other models).
"""

import asyncio
import json
import os
import sys
from typing import List, Dict, Any, Optional

try:
    import groq
except ImportError:
    print("Error: groq package not installed. Run: pip install groq")
    sys.exit(1)

from .base_agent import BaseAgent
from ..common.logging import get_logger
from ..common.protocol import LogLevel
from ..common.config import BashToolConfig


class GroqAgent(BaseAgent):
    """
    Groq agent implementation using Groq API.
    
    Handles communication with various models available through Groq,
    including Kimi K2 and other high-performance models.
    """
    
    def __init__(self, name: str, host: str, port: int, token: str, config_path: str = "config/agents.ini"):
        super().__init__(name, host, port, token, config_path)
        self.groq_client: Optional[groq.AsyncGroq] = None
        
    async def initialize_api_client(self):
        """Initialize Groq API client."""
        api_key = os.getenv(self.agent_config.api_key_env)
        if not api_key:
            error_msg = f"API key environment variable {self.agent_config.api_key_env} not set"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        try:
            # Configure timeout to prevent hanging API calls
            import httpx
            
            api_timeout = getattr(self.agent_config, 'api_timeout', 30)
            self.groq_client = groq.AsyncGroq(
                api_key=api_key,
                timeout=httpx.Timeout(api_timeout, connect=10.0),  # API timeout + 10s connect timeout
                max_retries=2  # Allow reasonable retries for transient failures
            )
            self.logger.info(f"Groq API client initialized for model {self.agent_config.model} with {api_timeout}s timeout")
            await self.send_system_log(LogLevel.INFO, f"Connected to Groq API ({self.agent_config.model})")
            
        except Exception as e:
            error_msg = f"Failed to initialize Groq client: {e}"
            self.logger.error(error_msg)
            # Temporarily disable send_system_log to avoid crash
            raise
    
    def get_provider_name(self) -> str:
        """Get the provider name."""
        return "groq"
    
    def _convert_to_groq_format(self, conversation: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert conversation context to Groq's message format.
        
        Groq uses OpenAI-compatible format:
        - Messages with 'role' and 'content'
        - System messages can be included
        - Supports user, assistant, system roles
        """
        groq_messages = []
        
        # Add system prompt as first message
        system_prompt = self._get_system_prompt()
        if system_prompt:
            groq_messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        # Add conversation messages
        for msg in conversation:
            role = msg["role"]
            content = msg["content"]
            
            # Groq uses same role names as OpenAI
            groq_messages.append({
                "role": role,
                "content": content
            })
        
        return groq_messages
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        base_prompt = self.agent_config.system_prompt
        
        # Add context about available tools
        if self.available_tools:
            tools_list = ", ".join(sorted(self.available_tools))
            tool_prompt = f"\n\nYou have access to these tools: {tools_list}"
            base_prompt += tool_prompt
        
        # Add context about model capabilities
        if "kimi" in self.agent_config.model.lower():
            context_prompt = "\n\nYou have a large context window and excel at complex analysis and reasoning tasks."
            base_prompt += context_prompt
        
        return base_prompt

    def _build_tool_definitions(self) -> List[Dict[str, Any]]:
        """Build tool definitions in Groq (OpenAI-compatible) format with agent-specific constraints."""
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
    
    def should_respond_to_general_question(self, content: str) -> bool:
        """
        Groq-specific logic for responding to general questions.
        
        Kimi and other Groq models are good at complex reasoning,
        so they respond to analytical and research questions.
        """
        content_lower = content.lower()
        
        # Skip very short messages
        if len(content.strip()) < 10:
            return False
        
        # Respond to complex analysis requests
        analysis_indicators = [
            "analyze", "analysis", "research", "investigate",
            "compare", "evaluate", "assess", "review",
            "explain", "reason", "think", "complex",
            "detailed", "comprehensive", "deep", "thorough"
        ]
        
        for indicator in analysis_indicators:
            if indicator in content_lower:
                return True
        
        # Respond to multi-step questions
        if "?" in content and len(content.strip()) > 30:
            return True
        
        # Respond to brainstorming and planning requests
        planning_indicators = [
            "plan", "strategy", "approach", "solution",
            "brainstorm", "ideas", "options", "alternatives",
            "steps", "process", "method", "way"
        ]
        
        for indicator in planning_indicators:
            if indicator in content_lower:
                return True
        
        # Respond to research and data requests
        research_indicators = [
            "research", "data", "information", "facts",
            "statistics", "study", "report", "findings",
            "evidence", "sources", "references"
        ]
        
        for indicator in research_indicators:
            if indicator in content_lower:
                return True
        
        # For Kimi specifically, be more responsive due to large context window
        if "kimi" in self.agent_config.model.lower():
            if len(content.strip()) > 20:
                return True
        
        # More selective for other models
        return len(content.strip()) > 50
    
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
            
        except Exception as e:
            self.logger.warning(f"Failed to load bash config, using defaults: {e}")
            return BashToolConfig(name='bash')
    
    async def generate_response(self, conversation: List[Dict[str, Any]]) -> Optional[str]:
        """
        Generate response using Groq API with tool support.
        
        Args:
            conversation: List of messages in OpenAI format
            
        Returns:
            Generated response text or None if no response
        """
        if not self.groq_client:
            await self.send_system_log(LogLevel.ERROR, "Groq client not initialized")
            return None
        
        if not conversation:
            return None
        
        try:
            # Convert conversation to Groq format
            messages = self._convert_to_groq_format(conversation)
            tools = self._build_tool_definitions()
            
            if not messages:
                return None
            
            self.logger.debug(f"Sending request to Groq with {len(messages)} messages")
            await self.send_system_log(LogLevel.INFO, f"Generating response using {self.agent_config.model}")
            
            # Tool execution loop - continues until we get a text response or hit max iterations
            max_tool_iterations = 10
            iteration = 0

            while iteration < max_tool_iterations:
                iteration += 1

                # API call (with tools available for chained calls)
                response = await self.groq_client.chat.completions.create(
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
                self.logger.warning(f"Empty response from Groq API (finish_reason: {response.choices[0].finish_reason})")
                return None

            # Hit max iterations
            self.logger.warning(f"Hit max tool iterations ({max_tool_iterations}), forcing text extraction")
            await self.send_system_log(LogLevel.WARNING, f"Tool chain exceeded {max_tool_iterations} iterations")

            # Try to extract any text from the last response
            if message.content and message.content.strip():
                return message.content.strip()

            return None
            
        except groq.RateLimitError as e:
            error_msg = f"Rate limit exceeded: {e}"
            self.logger.warning(error_msg)
            await self.send_system_log(LogLevel.WARNING, "Rate limit exceeded, will retry later")
            self._inject_error_feedback("RATE_LIMIT", error_msg)
            return None

        except groq.APIStatusError as e:
            error_msg = f"Groq API error: {e.status_code} - {e.response.text if hasattr(e, 'response') else 'Unknown error'}"
            self.logger.error(error_msg)
            await self.send_system_log(LogLevel.ERROR, f"API error: {e.status_code}")
            self._inject_error_feedback(f"API_ERROR_{e.status_code}", error_msg)
            return None

        except groq.APIConnectionError as e:
            error_msg = f"Connection error: {e}"
            self.logger.error(error_msg)
            await self.send_system_log(LogLevel.WARNING, "Connection error, retrying...")
            self._inject_error_feedback("CONNECTION_ERROR", error_msg)
            return None

        except groq.APITimeoutError as e:
            api_timeout = getattr(self.agent_config, 'api_timeout', 30)
            error_msg = f"Groq API timeout after {api_timeout}s: {str(e)}"
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


def main():
    """Main entry point for Groq agent process."""
    import argparse
    
    parser = argparse.ArgumentParser(description="NeuroDeck Groq Agent")
    parser.add_argument("--name", required=True, help="Agent name")
    parser.add_argument("--host", required=True, help="Orchestrator host")
    parser.add_argument("--port", type=int, required=True, help="Orchestrator port")
    parser.add_argument("--token", required=True, help="Authentication token")
    parser.add_argument("--config", default="config/agents.ini", help="Configuration file")
    
    args = parser.parse_args()
    
    # Create and run Groq agent
    agent = GroqAgent(
        name=args.name,
        host=args.host,
        port=args.port,
        token=args.token,
        config_path=args.config
    )
    
    try:
        asyncio.run(agent.start())
    except KeyboardInterrupt:
        print(f"\nGroq agent {args.name} shutting down...")
    except Exception as e:
        print(f"Groq agent {args.name} crashed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()