"""Configuration management for NeuroDeck using INI files."""

import configparser
import os
import secrets
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass, field


@dataclass
class AgentConfig:
    """Configuration for an individual agent."""
    name: str
    provider: str
    model: str
    api_endpoint: str
    api_key_env: str
    max_tokens: int
    temperature: float
    max_context_messages: int
    response_delay_max: int
    heartbeat_interval: int
    tools: List[str]
    system_prompt: str
    api_timeout: int = 30  # API request timeout in seconds
    tts_voice: str = ""
    tts_length_scale: float = 1.0
    tts_inference_noise: float = 0.667
    tts_duration_noise: float = 0.8


@dataclass
class ToolConfig:
    """Base configuration for tools."""
    name: str
    enabled: bool = True
    execution_timeout: int = 30
    approval_timeout: int = 30
    auto_approve_operations: List[str] = field(default_factory=list)
    require_approval_operations: List[str] = field(default_factory=list)
    paranoid_mode: bool = False


@dataclass 
class FilesystemToolConfig(ToolConfig):
    """Configuration for filesystem tool."""
    allowed_paths: List[str] = field(default_factory=list)
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    auto_approve_operations: List[str] = field(default_factory=lambda: ["read", "list"])
    require_approval_operations: List[str] = field(default_factory=lambda: ["write", "delete"])
    
    # File locking configuration
    use_file_locking: bool = True
    lock_base_delay: float = 1.0
    lock_max_delay: float = 30.0 
    stale_lock_cleanup_age: float = 300.0  # 5 minutes


@dataclass 
class BashToolConfig(ToolConfig):
    """Configuration for bash tool."""
    working_directory: str = "/tmp/neurodeck"
    allowed_paths: List[str] = field(default_factory=lambda: ["/tmp/neurodeck"])
    max_output_size: int = 1024 * 1024  # 1MB
    blocked_commands: List[str] = field(default_factory=lambda: ["sudo", "su", "rm -rf"])
    blocked_patterns: List[str] = field(default_factory=lambda: [r'\.\./+', r'^\s*/', r'[$`]'])
    allow_absolute_paths: bool = False
    allow_home_directory: bool = False
    allow_command_substitution: bool = False
    allow_variable_expansion: bool = False
    auto_approve_operations: List[str] = field(default_factory=lambda: ["pwd", "ls", "echo", "cat"])
    require_approval_operations: List[str] = field(default_factory=lambda: ["rm", "mv", "cp", "chmod"])


@dataclass
class MCPToolConfig(ToolConfig):
    """Configuration for MCP-based tools."""
    # MCP server connection
    command: str = ""
    args: str = ""
    description: str = ""
    
    # MCP-specific settings
    startup_timeout: int = 30  # Time to wait for MCP server to start
    max_retries: int = 3
    keepalive_interval: int = 60
    
    # Tool behavior defaults for MCP tools
    execution_timeout: int = 60  # MCP tools often need more time
    approval_timeout: int = 45
    require_approval_operations: List[str] = field(default_factory=lambda: ["*"])  # Default: approve everything


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator daemon."""
    host: str
    port: int
    tls_cert: str
    tls_key: str
    auth_token: str
    log_level: str
    username: str
    chat_history_limit: int
    tts_enabled: bool = True


class ConfigManager:
    """Manages loading and parsing of NeuroDeck configuration."""
    
    def __init__(self, config_path: str = "config/agents.ini"):
        self.config_path = Path(config_path)
        self.config = configparser.ConfigParser()
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        self.config.read(self.config_path)
    
    def load_orchestrator_config(self) -> OrchestratorConfig:
        """Load orchestrator configuration."""
        if 'orchestrator' not in self.config:
            raise ValueError("Missing [orchestrator] section in config")

        section = self.config['orchestrator']
        auth_token = section.get('auth_token', '')

        # Auto-generate token if missing or placeholder
        if not auth_token or auth_token in ('', 'GENERATE_ON_FIRST_RUN', 'your-token-here'):
            auth_token = secrets.token_hex(16).upper()
            self._save_auth_token(auth_token)

        return OrchestratorConfig(
            host=section.get('host', 'localhost'),
            port=section.getint('port', 9999),
            tls_cert=section.get('tls_cert', 'certs/server.crt'),
            tls_key=section.get('tls_key', 'certs/server.key'),
            auth_token=auth_token,
            log_level=section.get('log_level', 'INFO'),
            username=section.get('username', 'human'),
            chat_history_limit=section.getint('chat_history_limit', 20),
            tts_enabled=section.getboolean('tts_enabled', True)
        )

    def _save_auth_token(self, token: str) -> None:
        """Save generated auth token back to config file and .env."""
        # Save to agents.ini
        self.config['orchestrator']['auth_token'] = token
        with open(self.config_path, 'w') as f:
            self.config.write(f)

        # Also save to config/.env as a convenience
        env_path = Path(self.config_path).parent / '.env'
        self._update_env_token(env_path, token)

    def _update_env_token(self, env_path: Path, token: str) -> None:
        """Update or add NEURODECK_TOKEN in .env file."""
        token_line = f"NEURODECK_TOKEN={token}"

        if env_path.exists():
            lines = env_path.read_text().splitlines()
            updated = False
            for i, line in enumerate(lines):
                if line.startswith('NEURODECK_TOKEN='):
                    lines[i] = token_line
                    updated = True
                    break
            if not updated:
                lines.insert(0, token_line)
            env_path.write_text('\n'.join(lines) + '\n')
        else:
            env_path.write_text(token_line + '\n')
    
    def load_agent_configs(self) -> Dict[str, AgentConfig]:
        """Load all agent configurations."""
        agents = {}
        
        for section_name in self.config.sections():
            if not section_name.startswith('agent:'):
                continue
            
            agent_name = section_name.split(':', 1)[1]
            section = self.config[section_name]
            
            # Parse tools list
            tools_str = section.get('tools', '')
            tools = [tool.strip() for tool in tools_str.split(',') if tool.strip()]
            
            agents[agent_name] = AgentConfig(
                name=agent_name,
                provider=section.get('provider'),
                model=section.get('model'),
                api_endpoint=section.get('api_endpoint'),
                api_key_env=section.get('api_key_env'),
                max_tokens=section.getint('max_tokens', 4096),
                temperature=section.getfloat('temperature', 0.7),
                max_context_messages=section.getint('max_context_messages', 20),
                response_delay_max=section.getint('response_delay_max', 3),
                heartbeat_interval=section.getint('heartbeat_interval', 60),
                tools=tools,
                system_prompt=section.get('system_prompt', ''),
                api_timeout=section.getint('api_timeout', 30),
                tts_voice=section.get('tts_voice', ''),
                tts_length_scale=section.getfloat('tts_length_scale', 1.0),
                tts_inference_noise=section.getfloat('tts_inference_noise', 0.667),
                tts_duration_noise=section.getfloat('tts_duration_noise', 0.8)
            )
        
        return agents
    
    def load_mcp_tool_configs(self) -> Dict[str, MCPToolConfig]:
        """Load all MCP tool configurations."""
        mcp_tools = {}
        
        for section_name in self.config.sections():
            if not section_name.startswith('mcp:'):
                continue
            
            tool_name = section_name.split(':', 1)[1]
            section = self.config[section_name]
            
            mcp_tools[tool_name] = MCPToolConfig(
                name=tool_name,
                command=section.get('command'),
                args=section.get('args', ''),
                description=section.get('description', '')
            )
        
        return mcp_tools
    
    def load_tool_configs(self) -> Dict[str, ToolConfig]:
        """Load all tool behavior configurations."""
        tool_configs = {}
        
        for section_name in self.config.sections():
            if not section_name.startswith('tool:'):
                continue
            
            # Handle both [tool:toolname] and [tool:toolname:agent] sections
            parts = section_name.split(':')
            if len(parts) < 2:
                continue
                
            tool_name = parts[1]
            # Skip agent-specific overrides in this method
            if len(parts) > 2:
                continue
                
            section = self.config[section_name]
            
            # Parse common tool parameters
            auto_approve_ops = self._parse_list(section.get('auto_approve_operations', ''))
            require_approval_ops = self._parse_list(section.get('require_approval_operations', ''))
            
            # Create appropriate config type based on tool
            if tool_name == 'filesystem':
                allowed_paths = self._parse_list(section.get('allowed_paths', ''))
                if not allowed_paths:
                    # Default fallback paths
                    allowed_paths = ['/tmp/neurodeck', os.getcwd()]
                
                tool_configs[tool_name] = FilesystemToolConfig(
                    name=tool_name,
                    enabled=section.getboolean('enabled', True),
                    execution_timeout=self._parse_int(section.get('execution_timeout', '30')),
                    approval_timeout=self._parse_int(section.get('approval_timeout', '30')),
                    auto_approve_operations=auto_approve_ops or ["read", "list"],
                    require_approval_operations=require_approval_ops or ["write", "delete"],
                    paranoid_mode=section.getboolean('paranoid_mode', False),
                    allowed_paths=allowed_paths,
                    max_file_size=self._parse_int(section.get('max_file_size', str(10 * 1024 * 1024))),
                    # File locking configuration
                    use_file_locking=section.getboolean('use_file_locking', True),
                    lock_base_delay=self._parse_float(section.get('lock_base_delay', '1.0')),
                    lock_max_delay=self._parse_float(section.get('lock_max_delay', '30.0')),
                    stale_lock_cleanup_age=self._parse_float(section.get('stale_lock_cleanup_age', '300.0'))
                )
            elif tool_name == 'bash':
                allowed_paths = self._parse_list(section.get('allowed_paths', ''))
                if not allowed_paths:
                    allowed_paths = ['/tmp/neurodeck']  # Safe default
                
                blocked_commands = self._parse_list(section.get('blocked_commands', ''))
                blocked_patterns = self._parse_list(section.get('blocked_patterns', ''))
                
                tool_configs[tool_name] = BashToolConfig(
                    name=tool_name,
                    enabled=section.getboolean('enabled', True),
                    execution_timeout=self._parse_int(section.get('execution_timeout', '30')),
                    approval_timeout=self._parse_int(section.get('approval_timeout', '30')),
                    auto_approve_operations=auto_approve_ops or ["pwd", "ls", "echo", "cat"],
                    require_approval_operations=require_approval_ops or ["rm", "mv", "cp", "chmod"],
                    paranoid_mode=section.getboolean('paranoid_mode', False),
                    working_directory=section.get('working_directory', '/tmp/neurodeck'),
                    allowed_paths=allowed_paths,
                    max_output_size=self._parse_int(section.get('max_output_size', '1048576')),
                    blocked_commands=blocked_commands,
                    blocked_patterns=blocked_patterns,
                    allow_absolute_paths=section.getboolean('allow_absolute_paths', False),
                    allow_home_directory=section.getboolean('allow_home_directory', False),
                    allow_command_substitution=section.getboolean('allow_command_substitution', False),
                    allow_variable_expansion=section.getboolean('allow_variable_expansion', False)
                )
            else:
                # Generic tool config for MCP and other tools
                tool_configs[tool_name] = ToolConfig(
                    name=tool_name,
                    enabled=section.getboolean('enabled', True),
                    execution_timeout=self._parse_int(section.get('execution_timeout', '60')),
                    approval_timeout=self._parse_int(section.get('approval_timeout', '45')),
                    auto_approve_operations=auto_approve_ops,
                    require_approval_operations=require_approval_ops or ["*"],
                    paranoid_mode=section.getboolean('paranoid_mode', False)
                )
        
        return tool_configs
    
    def load_agent_tool_overrides(self) -> Dict[str, Dict[str, ToolConfig]]:
        """Load per-agent tool configuration overrides."""
        agent_overrides = {}
        
        for section_name in self.config.sections():
            if not section_name.startswith('tool:'):
                continue
            
            parts = section_name.split(':')
            if len(parts) != 3:  # Must be [tool:toolname:agent]
                continue
                
            tool_name = parts[1]
            agent_name = parts[2]
            section = self.config[section_name]
            
            if agent_name not in agent_overrides:
                agent_overrides[agent_name] = {}
            
            # Parse override parameters
            auto_approve_ops = self._parse_list(section.get('auto_approve_operations', ''))
            require_approval_ops = self._parse_list(section.get('require_approval_operations', ''))
            
            # Create override config
            if tool_name == 'filesystem':
                allowed_paths = self._parse_list(section.get('allowed_paths', ''))
                agent_overrides[agent_name][tool_name] = FilesystemToolConfig(
                    name=tool_name,
                    enabled=section.getboolean('enabled') if section.get('enabled') else None,
                    execution_timeout=self._parse_int(section.get('execution_timeout')) if section.get('execution_timeout') else None,
                    approval_timeout=self._parse_int(section.get('approval_timeout')) if section.get('approval_timeout') else None,
                    auto_approve_operations=auto_approve_ops if auto_approve_ops else None,
                    require_approval_operations=require_approval_ops if require_approval_ops else None,
                    paranoid_mode=section.getboolean('paranoid_mode') if section.get('paranoid_mode') else None,
                    allowed_paths=allowed_paths if allowed_paths else None,
                    max_file_size=self._parse_int(section.get('max_file_size')) if section.get('max_file_size') else None
                )
            elif tool_name == 'bash':
                allowed_paths = self._parse_list(section.get('allowed_paths', ''))
                blocked_commands = self._parse_list(section.get('blocked_commands', ''))
                blocked_patterns = self._parse_list(section.get('blocked_patterns', ''))
                
                agent_overrides[agent_name][tool_name] = BashToolConfig(
                    name=tool_name,
                    enabled=section.getboolean('enabled') if section.get('enabled') else None,
                    execution_timeout=self._parse_int(section.get('execution_timeout')) if section.get('execution_timeout') else None,
                    approval_timeout=self._parse_int(section.get('approval_timeout')) if section.get('approval_timeout') else None,
                    auto_approve_operations=auto_approve_ops if auto_approve_ops else None,
                    require_approval_operations=require_approval_ops if require_approval_ops else None,
                    paranoid_mode=section.getboolean('paranoid_mode') if section.get('paranoid_mode') else None,
                    working_directory=section.get('working_directory') if section.get('working_directory') else None,
                    allowed_paths=allowed_paths if allowed_paths else None,
                    max_output_size=self._parse_int(section.get('max_output_size')) if section.get('max_output_size') else None,
                    blocked_commands=blocked_commands if blocked_commands else None,
                    blocked_patterns=blocked_patterns if blocked_patterns else None,
                    allow_absolute_paths=section.getboolean('allow_absolute_paths') if section.get('allow_absolute_paths') else None,
                    allow_home_directory=section.getboolean('allow_home_directory') if section.get('allow_home_directory') else None,
                    allow_command_substitution=section.getboolean('allow_command_substitution') if section.get('allow_command_substitution') else None,
                    allow_variable_expansion=section.getboolean('allow_variable_expansion') if section.get('allow_variable_expansion') else None
                )
            else:
                agent_overrides[agent_name][tool_name] = ToolConfig(
                    name=tool_name,
                    enabled=section.getboolean('enabled') if section.get('enabled') else None,
                    execution_timeout=self._parse_int(section.get('execution_timeout')) if section.get('execution_timeout') else None,
                    approval_timeout=self._parse_int(section.get('approval_timeout')) if section.get('approval_timeout') else None,
                    auto_approve_operations=auto_approve_ops if auto_approve_ops else None,
                    require_approval_operations=require_approval_ops if require_approval_ops else None,
                    paranoid_mode=section.getboolean('paranoid_mode') if section.get('paranoid_mode') else None
                )
        
        return agent_overrides
    
    def _parse_list(self, value: str) -> List[str]:
        """Parse comma-separated list from config value."""
        if not value:
            return []
        return [item.strip() for item in value.split(',') if item.strip()]
    
    def _parse_int(self, value: str) -> int:
        """Parse integer from config value, stripping comments."""
        if not value:
            return 0
        # Strip inline comments (everything after #)
        clean_value = value.split('#')[0].strip()
        return int(clean_value)
    
    def _parse_float(self, value: str) -> float:
        """Parse float from config value, stripping comments."""
        if not value:
            return 0.0
        # Strip inline comments (everything after #)
        clean_value = value.split('#')[0].strip()
        return float(clean_value)
    
    def get_built_in_tools(self) -> set:
        """Get the set of built-in tools that don't require MCP servers."""
        return {'filesystem', 'web_search', 'chat_info', 'tts_config', 'bash'}
    
    def is_mcp_tool(self, tool_name: str, mcp_tools: Dict[str, MCPToolConfig]) -> bool:
        """Check if a tool requires MCP server startup."""
        return tool_name not in self.get_built_in_tools() and tool_name in mcp_tools
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        # Check orchestrator config
        try:
            orchestrator_config = self.load_orchestrator_config()
            if not orchestrator_config.auth_token:
                errors.append("Missing auth_token in [orchestrator] section")
        except Exception as e:
            errors.append(f"Invalid orchestrator config: {e}")
        
        # Check agent configs
        try:
            agents = self.load_agent_configs()
            if not agents:
                errors.append("No agents configured")
            
            for agent_name, agent_config in agents.items():
                if not agent_config.provider:
                    errors.append(f"Agent {agent_name}: missing provider")
                if not agent_config.model:
                    errors.append(f"Agent {agent_name}: missing model")
                if not agent_config.api_key_env:
                    errors.append(f"Agent {agent_name}: missing api_key_env")
                if not agent_config.system_prompt:
                    errors.append(f"Agent {agent_name}: missing system_prompt")
                    
                # Check if API key environment variable exists
                if agent_config.api_key_env and not os.getenv(agent_config.api_key_env):
                    errors.append(f"Agent {agent_name}: API key environment variable {agent_config.api_key_env} not set")
        
        except Exception as e:
            errors.append(f"Invalid agent config: {e}")
        
        # Check MCP tool configs
        try:
            mcp_tools = self.load_mcp_tool_configs()
            for tool_name, tool_config in mcp_tools.items():
                if not tool_config.command:
                    errors.append(f"MCP tool {tool_name}: missing command")
        except Exception as e:
            errors.append(f"Invalid MCP tool config: {e}")
        
        # Check tool configurations
        try:
            tool_configs = self.load_tool_configs()
            for tool_name, tool_config in tool_configs.items():
                # Validate filesystem tool config
                if isinstance(tool_config, FilesystemToolConfig):
                    if not tool_config.allowed_paths:
                        errors.append(f"Tool {tool_name}: no allowed paths specified")
                    if tool_config.max_file_size <= 0:
                        errors.append(f"Tool {tool_name}: invalid max_file_size")
                    for path in tool_config.allowed_paths:
                        if not os.path.isabs(path):
                            errors.append(f"Tool {tool_name}: path must be absolute: {path}")
                
                # Validate timeout values
                if tool_config.execution_timeout <= 0:
                    errors.append(f"Tool {tool_name}: invalid execution_timeout")
                if tool_config.approval_timeout <= 0:
                    errors.append(f"Tool {tool_name}: invalid approval_timeout")
        except Exception as e:
            errors.append(f"Invalid tool config: {e}")
        
        # Check agent tool overrides
        try:
            agent_overrides = self.load_agent_tool_overrides()
            for agent_name, tool_overrides in agent_overrides.items():
                for tool_name, override_config in tool_overrides.items():
                    # Similar validation for overrides
                    if isinstance(override_config, FilesystemToolConfig):
                        if override_config.allowed_paths:
                            for path in override_config.allowed_paths:
                                if not os.path.isabs(path):
                                    errors.append(f"Tool override {tool_name}:{agent_name}: path must be absolute: {path}")
                        if override_config.max_file_size is not None and override_config.max_file_size <= 0:
                            errors.append(f"Tool override {tool_name}:{agent_name}: invalid max_file_size")
        except Exception as e:
            errors.append(f"Invalid agent tool override config: {e}")
        
        return errors


def load_config(config_path: str = "config/agents.ini") -> tuple[Dict[str, AgentConfig], OrchestratorConfig, Dict[str, MCPToolConfig], Dict[str, ToolConfig], Dict[str, Dict[str, ToolConfig]]]:
    """
    Load all configuration from INI file.
    
    Returns:
        Tuple of (agents, orchestrator_config, mcp_tools, tool_configs, agent_tool_overrides)
    """
    manager = ConfigManager(config_path)
    
    # Validate configuration
    errors = manager.validate_config()
    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
        raise ValueError(error_msg)
    
    agents = manager.load_agent_configs()
    orchestrator_config = manager.load_orchestrator_config()
    mcp_tools = manager.load_mcp_tool_configs()
    tool_configs = manager.load_tool_configs()
    agent_tool_overrides = manager.load_agent_tool_overrides()
    
    return agents, orchestrator_config, mcp_tools, tool_configs, agent_tool_overrides