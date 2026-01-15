"""
Process manager for NeuroDeck agent processes.

Handles spawning, monitoring, and lifecycle management of AI agent processes.
Each agent runs as a separate Python process that connects back to the orchestrator.
"""

import asyncio
import subprocess
import sys
import os
import time
import signal
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

from ..common.config import AgentConfig, MCPToolConfig, OrchestratorConfig
from ..common.logging import get_logger
from ..common.protocol import MessageType, system_log_message, LogLevel

logger = get_logger("orchestrator.process_manager")


class ProcessStatus(Enum):
    """Status of an agent process."""
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    CRASHED = "crashed"
    FAILED = "failed"


@dataclass
class MCPServer:
    """Represents an MCP server subprocess."""
    name: str
    config: MCPToolConfig
    process: Optional[subprocess.Popen] = None
    status: ProcessStatus = ProcessStatus.STOPPED
    restart_count: int = 0
    last_restart: float = 0.0


@dataclass
class AgentProcess:
    """Represents an agent process and its MCP servers."""
    name: str
    config: AgentConfig
    process: Optional[subprocess.Popen] = None
    status: ProcessStatus = ProcessStatus.STOPPED
    restart_count: int = 0
    last_restart: float = 0.0
    mcp_servers: Dict[str, MCPServer] = field(default_factory=dict)
    connected_to_orchestrator: bool = False
    last_heartbeat: float = 0.0
    # Log file handle for combined stdout/stderr output
    log_file_handle = None


class ProcessManager:
    """
    Manages agent processes and their MCP servers.
    
    Responsibilities:
    - Spawn agent processes from configuration
    - Monitor process health and restart on crashes
    - Manage MCP server subprocesses
    - Handle graceful shutdown
    - Track agent connection status
    """
    
    def __init__(
        self,
        orchestrator_config: OrchestratorConfig,
        agent_configs: Dict[str, AgentConfig],
        mcp_tool_configs: Dict[str, MCPToolConfig]
    ):
        self.orchestrator_config = orchestrator_config
        self.agent_configs = agent_configs
        self.mcp_tool_configs = mcp_tool_configs
        self.agents: Dict[str, AgentProcess] = {}
        self.running = False
        self.monitor_task: Optional[asyncio.Task] = None
        
        # Process restart settings
        self.max_restart_attempts = 5
        self.restart_delay_base = 2.0  # Base delay in seconds
        self.restart_delay_max = 60.0  # Max delay in seconds
        self.heartbeat_timeout = 60.0  # Agent must connect within 60 seconds
        
        # Callback for sending system logs to UI
        self.system_log_callback = None
        
        # Debug: Log what API keys the orchestrator has
        api_keys = ["ANTHROPIC_API_KEY", "XAI_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY"]
        for key in api_keys:
            value = os.getenv(key)
            if value:
                logger.info(f"Orchestrator has {key}: {value[:10]}...")
            else:
                logger.error(f"Orchestrator missing {key}")
        
        logger.info(f"Process manager initialized with {len(agent_configs)} agents")
    
    def set_system_log_callback(self, callback):
        """Set callback function to send system logs to UI clients."""
        self.system_log_callback = callback
    
    async def send_system_log(self, level: str, message: str, agent_name: str = "process_manager"):
        """Send system log message to UI clients."""
        if self.system_log_callback:
            # Convert string level to LogLevel enum
            if level == "error":
                log_level = LogLevel.ERROR
            elif level == "warning":
                log_level = LogLevel.WARNING
            else:
                log_level = LogLevel.INFO
            
            log_msg = system_log_message(agent_name, log_level, message)
            await self.system_log_callback(log_msg)
        
        # Also log locally
        if level == "error":
            logger.error(f"[{agent_name}] {message}")
        elif level == "warning":
            logger.warning(f"[{agent_name}] {message}")
        elif level == "debug":
            logger.debug(f"[{agent_name}] {message}")
        else:
            logger.info(f"[{agent_name}] {message}")
    
    async def start_all_agents(self):
        """Start all configured agents."""
        if self.running:
            raise RuntimeError("Process manager is already running")
        
        self.running = True
        logger.info("Starting all agent processes...")
        
        # Initialize agent processes
        logger.debug(f"Initializing {len(self.agent_configs)} agent processes...")
        for agent_name, agent_config in self.agent_configs.items():
            logger.debug(f"Creating AgentProcess for {agent_name}")
            try:
                agent_process = AgentProcess(name=agent_name, config=agent_config)
                logger.debug(f"AgentProcess created successfully for {agent_name}")
            except Exception as e:
                logger.error(f"Failed to create AgentProcess for {agent_name}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                raise
            
            # Initialize MCP servers for this agent
            logger.debug(f"Initializing MCP servers for {agent_name}, tools: {agent_config.tools}")
            for tool_name in agent_config.tools:
                logger.debug(f"Checking tool {tool_name} for agent {agent_name}")
                if tool_name in self.mcp_tool_configs:
                    logger.debug(f"Found MCP config for tool {tool_name}")
                    try:
                        mcp_config = self.mcp_tool_configs[tool_name]
                        agent_process.mcp_servers[tool_name] = MCPServer(
                            name=tool_name,
                            config=mcp_config
                        )
                        logger.debug(f"MCP server created for tool {tool_name}")
                    except Exception as e:
                        logger.error(f"Failed to create MCP server for tool {tool_name}: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                        raise
                else:
                    logger.debug(f"Tool {tool_name} is built-in, no MCP server needed")
            
            self.agents[agent_name] = agent_process
        
        # Start all agents
        logger.debug("Creating async tasks to start all agents...")
        start_tasks = []
        for agent_name in self.agents:
            logger.debug(f"Creating async task to start agent {agent_name}")
            try:
                task = asyncio.create_task(self._start_agent(agent_name))
                start_tasks.append(task)
                logger.debug(f"Task created successfully for agent {agent_name}")
            except Exception as e:
                logger.error(f"Failed to create task for agent {agent_name}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                raise
        
        # Wait for all agents to start
        logger.debug(f"Waiting for {len(start_tasks)} agent tasks to complete...")
        try:
            results = await asyncio.gather(*start_tasks, return_exceptions=True)
            logger.debug(f"Agent startup results: {results}")
            
            # Check for exceptions
            for i, result in enumerate(results):
                agent_name = list(self.agents.keys())[i]
                if isinstance(result, Exception):
                    logger.error(f"Agent {agent_name} failed to start: {result}")
                else:
                    logger.debug(f"Agent {agent_name} started successfully")
        except Exception as e:
            logger.error(f"Failed to start agents: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
        
        # Start monitoring task
        self.monitor_task = asyncio.create_task(self._monitor_processes())
        
        await self.send_system_log("info", f"Started {len(self.agents)} agent processes")
    
    async def _start_agent(self, agent_name: str) -> bool:
        """Start a single agent and its MCP servers."""
        agent = self.agents[agent_name]
        
        try:
            await self.send_system_log("info", f"Starting agent {agent_name}...", agent_name)
            agent.status = ProcessStatus.STARTING
            
            # Start MCP servers first
            await self._start_mcp_servers(agent)
            
            # Start the agent process
            await self._spawn_agent_process(agent)
            
            agent.status = ProcessStatus.RUNNING
            agent.last_restart = time.time()
            
            await self.send_system_log("info", f"Agent {agent_name} started successfully", agent_name)
            return True
            
        except Exception as e:
            agent.status = ProcessStatus.FAILED
            await self.send_system_log("error", f"Failed to start agent {agent_name}: {e}", agent_name)
            return False
    
    async def _start_mcp_servers(self, agent: AgentProcess):
        """Start all MCP servers for an agent."""
        if not agent.mcp_servers:
            return
        
        for tool_name, mcp_server in agent.mcp_servers.items():
            try:
                await self._start_mcp_server(mcp_server, agent.name)
            except Exception as e:
                await self.send_system_log(
                    "error", 
                    f"Failed to start MCP server {tool_name}: {e}", 
                    agent.name
                )
                # Continue with other servers even if one fails
    
    async def _start_mcp_server(self, mcp_server: MCPServer, agent_name: str):
        """Start a single MCP server subprocess."""
        config = mcp_server.config
        
        # Parse command and arguments
        cmd_parts = [config.command]
        if config.args:
            # Split args respecting quotes
            import shlex
            cmd_parts.extend(shlex.split(config.args))
        
        logger.info(f"Starting MCP server {mcp_server.name} for agent {agent_name}: {' '.join(cmd_parts)}")
        
        # Start the subprocess
        mcp_server.process = subprocess.Popen(
            cmd_parts,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.getcwd(),  # Use orchestrator's working directory
            env=os.environ.copy()
        )
        
        mcp_server.status = ProcessStatus.RUNNING
        
        await self.send_system_log(
            "info", 
            f"MCP server {mcp_server.name} started (PID: {mcp_server.process.pid})", 
            agent_name
        )
    
    async def _spawn_agent_process(self, agent: AgentProcess):
        """Spawn the agent Python process."""
        # Build command to run the agent
        python_exe = sys.executable
        logger.debug(f"Using Python executable: {python_exe}")
        agent_module = f"neurodeck.agents.{agent.config.provider}_agent"
        
        cmd = [
            python_exe, "-m", agent_module,
            "--name", agent.name,
            "--host", self.orchestrator_config.host,
            "--port", str(self.orchestrator_config.port),
            "--token", self.orchestrator_config.auth_token,
            "--config", "config/agents.ini"
        ]
        
        logger.info(f"Spawning agent {agent.name}: {' '.join(cmd)}")
        
        # Set up environment with API keys
        env = os.environ.copy()
        if agent.config.api_key_env:
            api_key = os.getenv(agent.config.api_key_env)
            if not api_key:
                raise ValueError(f"API key environment variable {agent.config.api_key_env} not set")
            logger.info(f"Orchestrator passing {agent.config.api_key_env} to agent {agent.name}: {api_key[:10]}...")
            # The agent will read it from the environment
        
        # Create log directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Open combined log file for agent stdout/stderr
        log_file_path = log_dir / f"agent_{agent.name}.log"
        agent.log_file_handle = open(log_file_path, "a", encoding="utf-8", buffering=1)  # Line buffered
        
        # Start the agent process with combined stdout/stderr to log file
        agent.process = subprocess.Popen(
            cmd,
            cwd=os.getcwd(),  # Use orchestrator's working directory
            env=env,
            stdout=agent.log_file_handle,  # Direct to combined log file
            stderr=subprocess.STDOUT      # Redirect stderr to stdout (same file)
        )
        
        logger.info(f"Agent {agent.name} spawned (PID: {agent.process.pid})")
    async def _monitor_processes(self):
        """Monitor all agent processes and restart if needed."""
        while self.running:
            try:
                # Check each agent
                for agent_name, agent in self.agents.items():
                    await self._check_agent_health(agent)
                
                # Sleep before next check
                await asyncio.sleep(5.0)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in process monitor: {e}")
                await asyncio.sleep(5.0)
    
    async def _check_agent_health(self, agent: AgentProcess):
        """Check health of a single agent and restart if needed."""
        if agent.status in [ProcessStatus.STOPPING, ProcessStatus.STOPPED]:
            return
        
        # Check if process is still running
        if agent.process and agent.process.poll() is not None:
            # Process has exited
            exit_code = agent.process.returncode
            agent.status = ProcessStatus.CRASHED
            
            # Capture stderr output to understand why it crashed
            stderr_output = ""
            if agent.process.stderr:
                try:
                    stderr_data = agent.process.stderr.read()
                    if stderr_data:
                        stderr_output = stderr_data.decode('utf-8').strip()
                except Exception as e:
                    logger.error(f"Failed to read stderr for agent {agent.name}: {e}")
            
            # Capture stdout output as well
            stdout_output = ""
            if agent.process.stdout:
                try:
                    stdout_data = agent.process.stdout.read()
                    if stdout_data:
                        stdout_output = stdout_data.decode('utf-8').strip()
                except Exception as e:
                    logger.error(f"Failed to read stdout for agent {agent.name}: {e}")
            
            crash_msg = f"Agent {agent.name} crashed with exit code {exit_code}"
            if stderr_output:
                crash_msg += f"\nSTDERR: {stderr_output}"
            if stdout_output:
                crash_msg += f"\nSTDOUT: {stdout_output}"
            
            await self.send_system_log("error", crash_msg, agent.name)
            
            # Attempt restart if under limit
            if agent.restart_count < self.max_restart_attempts:
                await self._restart_agent(agent)
            else:
                await self.send_system_log(
                    "error", 
                    f"Agent {agent.name} exceeded maximum restart attempts ({self.max_restart_attempts})", 
                    agent.name
                )
                agent.status = ProcessStatus.FAILED
        
        # Check connection timeout (agent should connect within timeout)
        elif (agent.status == ProcessStatus.STARTING and 
              time.time() - agent.last_restart > self.heartbeat_timeout and
              not agent.connected_to_orchestrator):
            
            await self.send_system_log(
                "warning", 
                f"Agent {agent.name} failed to connect within {self.heartbeat_timeout}s, restarting...", 
                agent.name
            )
            
            await self._restart_agent(agent)
        
        # Check MCP servers
        for mcp_server in agent.mcp_servers.values():
            await self._check_mcp_server_health(mcp_server, agent.name)
    
    async def _check_mcp_server_health(self, mcp_server: MCPServer, agent_name: str):
        """Check health of an MCP server subprocess."""
        if mcp_server.process and mcp_server.process.poll() is not None:
            # MCP server has exited
            exit_code = mcp_server.process.returncode
            mcp_server.status = ProcessStatus.CRASHED
            
            await self.send_system_log(
                "warning", 
                f"MCP server {mcp_server.name} crashed with exit code {exit_code}", 
                agent_name
            )
            
            # Restart MCP server if under limit
            if mcp_server.restart_count < self.max_restart_attempts:
                await self._restart_mcp_server(mcp_server, agent_name)
    
    async def _restart_agent(self, agent: AgentProcess):
        """Restart a crashed agent."""
        agent.restart_count += 1
        
        # Calculate restart delay with exponential backoff
        delay = min(
            self.restart_delay_base * (2 ** (agent.restart_count - 1)),
            self.restart_delay_max
        )
        
        await self.send_system_log(
            "info", 
            f"Restarting agent {agent.name} in {delay:.1f}s (attempt {agent.restart_count}/{self.max_restart_attempts})", 
            agent.name
        )
        
        # Stop current process if still running
        await self._stop_agent(agent, graceful=False)
        
        # Wait before restart
        await asyncio.sleep(delay)
        
        # Restart the agent
        await self._start_agent(agent.name)
    
    async def _restart_mcp_server(self, mcp_server: MCPServer, agent_name: str):
        """Restart a crashed MCP server."""
        mcp_server.restart_count += 1
        
        await self.send_system_log(
            "info", 
            f"Restarting MCP server {mcp_server.name} (attempt {mcp_server.restart_count})", 
            agent_name
        )
        
        # Stop current process
        if mcp_server.process:
            try:
                mcp_server.process.terminate()
                mcp_server.process.wait(timeout=5.0)
            except:
                pass
        
        # Restart the server
        await self._start_mcp_server(mcp_server, agent_name)
    
    def mark_agent_connected(self, agent_name: str):
        """Mark an agent as connected to the orchestrator."""
        if agent_name in self.agents:
            self.agents[agent_name].connected_to_orchestrator = True
            self.agents[agent_name].last_heartbeat = time.time()
            logger.info(f"Agent {agent_name} connected to orchestrator")
    
    def mark_agent_disconnected(self, agent_name: str):
        """Mark an agent as disconnected from the orchestrator."""
        if agent_name in self.agents:
            self.agents[agent_name].connected_to_orchestrator = False
            logger.info(f"Agent {agent_name} disconnected from orchestrator")
    
    async def stop_agent(self, agent_name: str):
        """Stop a specific agent and its MCP servers."""
        if agent_name not in self.agents:
            raise ValueError(f"Agent {agent_name} not found")
        
        agent = self.agents[agent_name]
        await self._stop_agent(agent)
        
        await self.send_system_log("info", f"Agent {agent_name} stopped", agent_name)
    
    async def _stop_agent(self, agent: AgentProcess, graceful: bool = True):
        """Stop an agent process and its MCP servers."""
        agent.status = ProcessStatus.STOPPING
        
        # Close log file handle
        if agent.log_file_handle:
            try:
                agent.log_file_handle.close()
                agent.log_file_handle = None
            except Exception as e:
                logger.error(f"Error closing log file for agent {agent.name}: {e}")
        
        # Stop MCP servers first (in parallel)
        if agent.mcp_servers:
            mcp_stop_tasks = [
                asyncio.create_task(self._stop_mcp_server(mcp_server))
                for mcp_server in agent.mcp_servers.values()
            ]
            await asyncio.gather(*mcp_stop_tasks, return_exceptions=True)
        
        # Stop the agent process
        if agent.process:
            try:
                if graceful:
                    # Try graceful shutdown first
                    agent.process.terminate()
                    try:
                        await asyncio.wait_for(
                            asyncio.create_task(asyncio.to_thread(agent.process.wait)),
                            timeout=3.0
                        )
                    except asyncio.TimeoutError:
                        # Force kill if graceful shutdown failed
                        agent.process.kill()
                        await asyncio.create_task(asyncio.to_thread(agent.process.wait))
                else:
                    # Force kill immediately
                    agent.process.kill()
                    await asyncio.create_task(asyncio.to_thread(agent.process.wait))
                    
            except Exception as e:
                logger.error(f"Error stopping agent {agent.name}: {e}")
            
            agent.process = None
        
        agent.status = ProcessStatus.STOPPED
        agent.connected_to_orchestrator = False
    
    async def _stop_mcp_server(self, mcp_server: MCPServer):
        """Stop an MCP server subprocess."""
        if mcp_server.process:
            try:
                mcp_server.process.terminate()
                # Use asyncio to wait without blocking
                try:
                    await asyncio.wait_for(
                        asyncio.create_task(asyncio.to_thread(mcp_server.process.wait)),
                        timeout=2.0
                    )
                except asyncio.TimeoutError:
                    mcp_server.process.kill()
                    await asyncio.create_task(asyncio.to_thread(mcp_server.process.wait))
            except Exception as e:
                logger.error(f"Error stopping MCP server {mcp_server.name}: {e}")
            
            mcp_server.process = None
        
        mcp_server.status = ProcessStatus.STOPPED
    
    async def stop_all_agents(self):
        """Stop all agent processes gracefully."""
        if not self.running:
            return
        
        logger.info("Stopping all agent processes...")
        self.running = False
        
        # Cancel monitoring task
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        # Stop all agents
        stop_tasks = []
        for agent in self.agents.values():
            task = asyncio.create_task(self._stop_agent(agent))
            stop_tasks.append(task)
        
        # Wait for all agents to stop
        await asyncio.gather(*stop_tasks, return_exceptions=True)
        
        await self.send_system_log("info", "All agent processes stopped")
    
    def get_agent_status(self) -> Dict[str, Dict]:
        """Get status of all agents."""
        status = {}
        for agent_name, agent in self.agents.items():
            status[agent_name] = {
                "status": agent.status.value,
                "pid": agent.process.pid if agent.process else None,
                "restart_count": agent.restart_count,
                "connected": agent.connected_to_orchestrator,
                "mcp_servers": {
                    name: {
                        "status": server.status.value,
                        "pid": server.process.pid if server.process else None,
                        "restart_count": server.restart_count
                    }
                    for name, server in agent.mcp_servers.items()
                }
            }
        return status
    
    async def restart_agent(self, agent_name: str):
        """Manually restart an agent."""
        if agent_name not in self.agents:
            raise ValueError(f"Agent {agent_name} not found")
        
        agent = self.agents[agent_name]
        await self.send_system_log("info", f"Manual restart requested for agent {agent_name}", agent_name)
        
        await self._restart_agent(agent)