"""
Tool execution manager for NeuroDeck orchestrator.

Executes tools on behalf of agents with security controls.
"""

import os
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from ..tools.filesystem import FilesystemTool
from ..tools.chat_info import ChatInfoTool
from ..tools.tts_config import TTSConfigTool
from ..tools.bash import BashTool
from ..common.logging import get_logger
from ..common.config import ToolConfig, FilesystemToolConfig, BashToolConfig, MCPToolConfig

logger = get_logger(__name__)

class ToolExecutor:
    """
    Executes tools on behalf of agents with security controls.
    """
    
    def __init__(self, config: dict, tool_configs: Dict[str, ToolConfig], 
                 agent_tool_overrides: Dict[str, Dict[str, ToolConfig]], 
                 mcp_configs: Dict[str, MCPToolConfig]):
        self.config = config
        self.tool_configs = tool_configs
        self.agent_tool_overrides = agent_tool_overrides
        self.mcp_configs = mcp_configs
        self.filesystem_tools: Dict[str, FilesystemTool] = {}
        self.bash_tools: Dict[str, BashTool] = {}
        self.chat_info_tool: Optional[ChatInfoTool] = None
        self.tts_config_tool: Optional[TTSConfigTool] = None
        self.mcp_clients: Dict[str, Any] = {}
        
        # Default fallback values if no configuration exists
        self.default_allowed_paths = [
            "/tmp/neurodeck",
            os.getcwd()  # Current working directory
        ]
    
    def set_orchestrator_server(self, orchestrator_server):
        """Set orchestrator server reference for chat_info tool."""
        if self.chat_info_tool is None:
            self.chat_info_tool = ChatInfoTool(orchestrator_server)
        else:
            self.chat_info_tool.orchestrator_server = orchestrator_server
    
    def set_tts_service(self, tts_service):
        """Set TTS service reference for tts_config tool."""
        if self.tts_config_tool is None:
            self.tts_config_tool = TTSConfigTool(tts_service)
        else:
            self.tts_config_tool.tts_service = tts_service
    
    async def execute_tool(self, agent_id: str, agent_name: str, tool_name: str, 
                          command: str, args: Dict[str, Any]) -> Any:
        """
        Execute tool for an agent.
        
        Args:
            agent_id: ID of requesting agent (connection identifier)
            agent_name: Name of requesting agent (e.g., "chatgpt", "claude")
            tool_name: Name of tool to execute
            command: Command to execute (usually "execute")
            args: Tool-specific arguments
            
        Returns:
            Tool execution result
        """
        # Get tool configuration with agent overrides
        tool_config = self._get_agent_tool_config(agent_id, tool_name)
        
        logger.info(f"Executing tool {tool_name} for agent {agent_name} ({agent_id})")
        
        if tool_name == "filesystem":
            return await self._execute_filesystem(agent_id, agent_name, args, tool_config)
        elif tool_name == "bash":
            return await self._execute_bash(agent_id, agent_name, args, tool_config)
        elif tool_name == "chat_info":
            return await self._execute_chat_info(agent_id, agent_name, args, tool_config)
        elif tool_name == "tts_config":
            return await self._execute_tts_config(agent_id, agent_name, args, tool_config)
        elif tool_name in self.mcp_configs:
            return await self._execute_mcp_tool(agent_id, agent_name, tool_name, command, args, tool_config)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
    
    async def _execute_filesystem(self, agent_id: str, agent_name: str, args: Dict[str, Any], 
                                 config: FilesystemToolConfig) -> Any:
        """Execute filesystem operation with configuration."""
        # Get or create filesystem tool for agent
        tool_key = f"{agent_id}_filesystem"
        if tool_key not in self.filesystem_tools:
            self.filesystem_tools[tool_key] = FilesystemTool(
                allowed_paths=config.allowed_paths,
                max_file_size=config.max_file_size
            )
        
        fs_tool = self.filesystem_tools[tool_key]
        
        # Execute operation with configured timeout
        return await asyncio.wait_for(
            fs_tool.execute(
                action=args.get("action"),
                path=args.get("path"),
                content=args.get("content"),
                agent_name=agent_name,
                # Pass locking configuration
                use_locking=config.use_file_locking,
                base_delay=config.lock_base_delay,
                max_delay=config.lock_max_delay
            ),
            timeout=config.execution_timeout
        )
    
    async def _execute_bash(self, agent_id: str, agent_name: str, args: Dict[str, Any], 
                           config) -> Any:
        """Execute bash operation with configuration."""
        # Get or create bash tool for agent
        tool_key = f"{agent_id}_bash"
        if tool_key not in self.bash_tools:
            try:
                # Ensure working directory exists with proper error handling
                work_dir = Path(config.working_directory)
                work_dir.mkdir(parents=True, mode=0o750, exist_ok=True)
                
                # Validate that working directory is within allowed paths
                allowed_paths = [Path(p).resolve() for p in config.allowed_paths]
                work_dir_resolved = work_dir.resolve()
                
                is_allowed = any(
                    work_dir_resolved == allowed_path or 
                    work_dir_resolved.is_relative_to(allowed_path)
                    for allowed_path in allowed_paths
                )
                
                if not is_allowed:
                    raise ValueError(f"Working directory {work_dir} not within allowed paths {config.allowed_paths}")
                
                self.bash_tools[tool_key] = BashTool(config=config)
                logger.info(f"Created bash tool for agent {agent_name} with working directory {work_dir}")
                
            except Exception as e:
                logger.error(f"Failed to create bash tool for agent {agent_name}: {e}")
                raise
        
        bash_tool = self.bash_tools[tool_key]
        
        # Execute operation with configured timeout
        try:
            return await asyncio.wait_for(
                bash_tool.execute(
                    action=args.get("action"),
                    command=args.get("command"),
                    path=args.get("path"),
                    agent_name=agent_name
                ),
                timeout=config.execution_timeout
            )
        except asyncio.TimeoutError:
            logger.error(f"Bash tool execution timed out for agent {agent_name} after {config.execution_timeout}s")
            return {
                "success": False,
                "error": "Bash tool execution timed out",
                "timeout": config.execution_timeout
            }
    
    async def _execute_chat_info(self, agent_id: str, agent_name: str, args: Dict[str, Any], 
                                config: ToolConfig) -> Any:
        """Execute chat info operation with configuration."""
        # Initialize chat_info_tool if not already done
        if self.chat_info_tool is None:
            self.chat_info_tool = ChatInfoTool()
        
        logger.info(f"Chat info: agent '{agent_name}' ({agent_id})")
        
        # Execute operation with configured timeout  
        return await asyncio.wait_for(
            self.chat_info_tool.execute(
                action=args.get("action"),
                agent_name=agent_name,
                participant_name=args.get("participant_name")
            ),
            timeout=config.execution_timeout
        )
    
    async def _execute_tts_config(self, agent_id: str, agent_name: str, args: Dict[str, Any], 
                                 config: ToolConfig) -> Any:
        """Execute TTS config operation with configuration."""
        # Initialize tts_config_tool if not already done
        if self.tts_config_tool is None:
            self.tts_config_tool = TTSConfigTool()
        
        logger.info(f"TTS config: agent '{agent_name}' ({agent_id})")
        
        # Execute operation with configured timeout
        return await asyncio.wait_for(
            self.tts_config_tool.execute(
                action=args.get("action"),
                agent_name=agent_name,
                **{k: v for k, v in args.items() if k != "action"}
            ),
            timeout=config.execution_timeout
        )
    
    async def _execute_mcp_tool(self, agent_id: str, agent_name: str, tool_name: str, command: str, 
                               args: Dict[str, Any], config: MCPToolConfig) -> Any:
        """Execute MCP tool with Phase 6 configuration."""
        # Get or create MCP client
        if tool_name not in self.mcp_clients:
            # TODO: Initialize MCP client with config.command, config.args
            # This would be implemented in Phase 5 MCP integration
            logger.warning(f"MCP tool {tool_name} not implemented yet")
            raise NotImplementedError(f"MCP tool {tool_name} not implemented")
        
        mcp_client = self.mcp_clients[tool_name]
        
        # Execute with configured timeout
        return await asyncio.wait_for(
            mcp_client.execute_tool(command, args),
            timeout=config.execution_timeout
        )
    
    def _get_agent_tool_config(self, agent_id: str, tool_name: str) -> Union[ToolConfig, FilesystemToolConfig, BashToolConfig]:
        """Get tool configuration with agent-specific overrides."""
        # Start with global tool config or create default
        if tool_name in self.tool_configs:
            base_config = self.tool_configs[tool_name]
        elif tool_name in self.mcp_configs:
            base_config = self.mcp_configs[tool_name]
        else:
            # Create default configuration
            if tool_name == "filesystem":
                base_config = FilesystemToolConfig(
                    name=tool_name,
                    allowed_paths=self.default_allowed_paths.copy()
                )
            elif tool_name == "bash":
                base_config = BashToolConfig(
                    name=tool_name,
                    working_directory="/tmp/neurodeck",
                    allowed_paths=["/tmp/neurodeck"],
                    auto_approve_operations=["pwd", "ls", "echo", "cat", "grep", "find"],
                    require_approval_operations=["rm", "mv", "cp", "chmod", "sudo", "su"]
                )
            elif tool_name == "chat_info":
                base_config = ToolConfig(
                    name=tool_name,
                    auto_approve_operations=["list_participants", "get_chat_status", "get_my_info", "get_participant_info"],
                    require_approval_operations=[],  # All chat_info operations are safe
                    execution_timeout=10.0  # Chat info operations should be fast
                )
            else:
                base_config = ToolConfig(name=tool_name)
        
        # Apply agent-specific overrides if they exist
        if (agent_id in self.agent_tool_overrides and 
            tool_name in self.agent_tool_overrides[agent_id]):
            override_config = self.agent_tool_overrides[agent_id][tool_name]
            return self._merge_tool_configs(base_config, override_config)
        
        return base_config
    
    def _merge_tool_configs(self, base: Union[ToolConfig, FilesystemToolConfig], 
                           override: Union[ToolConfig, FilesystemToolConfig]) -> Union[ToolConfig, FilesystemToolConfig]:
        """Merge tool configurations (override takes precedence)."""
        if isinstance(base, FilesystemToolConfig) and isinstance(override, FilesystemToolConfig):
            return FilesystemToolConfig(
                name=base.name,
                enabled=override.enabled if override.enabled is not None else base.enabled,
                execution_timeout=override.execution_timeout if override.execution_timeout is not None else base.execution_timeout,
                approval_timeout=override.approval_timeout if override.approval_timeout is not None else base.approval_timeout,
                auto_approve_operations=override.auto_approve_operations if override.auto_approve_operations is not None else base.auto_approve_operations,
                require_approval_operations=override.require_approval_operations if override.require_approval_operations is not None else base.require_approval_operations,
                paranoid_mode=override.paranoid_mode if override.paranoid_mode is not None else base.paranoid_mode,
                allowed_paths=override.allowed_paths if override.allowed_paths is not None else base.allowed_paths,
                max_file_size=override.max_file_size if override.max_file_size is not None else base.max_file_size
            )
        else:
            return ToolConfig(
                name=base.name,
                enabled=override.enabled if override.enabled is not None else base.enabled,
                execution_timeout=override.execution_timeout if override.execution_timeout is not None else base.execution_timeout,
                approval_timeout=override.approval_timeout if override.approval_timeout is not None else base.approval_timeout,
                auto_approve_operations=override.auto_approve_operations if override.auto_approve_operations is not None else base.auto_approve_operations,
                require_approval_operations=override.require_approval_operations if override.require_approval_operations is not None else base.require_approval_operations,
                paranoid_mode=override.paranoid_mode if override.paranoid_mode is not None else base.paranoid_mode
            )