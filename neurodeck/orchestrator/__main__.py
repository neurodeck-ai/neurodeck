"""
NeuroDeck Orchestrator Daemon Entry Point

Usage:
    python -m neurodeck.orchestrator [--config CONFIG_FILE] [--port PORT] [--host HOST]
"""

import asyncio
import sys
import signal
import argparse
from pathlib import Path

from ..common.config import load_config
from ..common.logging import setup_logging
from .server import OrchestratorServer
from .process_manager import ProcessManager


class OrchestratorDaemon:
    """Main orchestrator daemon with signal handling and timeouts."""
    
    def __init__(self, config_path: str = "config/agents.ini"):
        self.config_path = config_path
        self.server = None
        self.process_manager = None
        self.running = False
    
    async def start(self, host: str = None, port: int = None):
        """Start the orchestrator daemon with timeout protection."""
        try:
            # Load configuration with timeout
            agents, orchestrator_config, mcp_tools, tool_configs, agent_tool_overrides = await asyncio.wait_for(
                asyncio.to_thread(load_config, self.config_path),
                timeout=10.0
            )
            
            # Override host/port if provided
            if host:
                orchestrator_config.host = host
            if port:
                orchestrator_config.port = port
            
            # Setup logging
            setup_logging(
                log_level=orchestrator_config.log_level,
                log_file="logs/orchestrator.log"
            )
            
            print(f"🚀 Starting NeuroDeck Orchestrator on {orchestrator_config.host}:{orchestrator_config.port}")
            print(f"📁 Config: {self.config_path}")
            print(f"🤖 Available agents: {len(agents)}")
            print(f"🔧 MCP tools: {len(mcp_tools)}")
            print(f"⚙️  Tool configurations: {len(tool_configs)}")
            
            # Create process manager
            self.process_manager = ProcessManager(
                orchestrator_config=orchestrator_config,
                agent_configs=agents,
                mcp_tool_configs=mcp_tools
            )
            
            # Create and configure server with tool configurations
            self.server = OrchestratorServer(
                orchestrator_config, 
                tool_configs, 
                agent_tool_overrides,
                mcp_tools
            )
            
            # Set up process manager callback for system logs
            self.process_manager.set_system_log_callback(
                self.server.broadcast_system_log
            )
            
            # Set up server callback for agent connections
            self.server.set_process_manager(self.process_manager)
            
            self.running = True
            
            # Start process manager first (this starts all agents)
            print("🤖 Starting agent processes...")
            await self.process_manager.start_all_agents()
            
            # Start server (this will accept connections from agents)
            print(f"🌐 Starting orchestrator server...")
            await asyncio.wait_for(
                self.server.start(),
                timeout=None  # Server runs indefinitely, no timeout for main loop
            )
            
        except asyncio.TimeoutError:
            print("❌ Timeout during orchestrator startup")
            sys.exit(1)
        except KeyboardInterrupt:
            print("\n🛑 Shutting down orchestrator...")
            await self.stop()
        except Exception as e:
            print(f"❌ Error starting orchestrator: {e}")
            sys.exit(1)
    
    async def stop(self):
        """Stop the orchestrator daemon with timeout protection."""
        if self.running:
            self.running = False
            
            # Stop process manager first (this stops all agents)
            if self.process_manager:
                try:
                    print("🛑 Stopping agent processes...")
                    await asyncio.wait_for(
                        self.process_manager.stop_all_agents(),
                        timeout=15.0
                    )
                except asyncio.TimeoutError:
                    print("⚠️  Timeout stopping agents, forcing shutdown")
                except Exception as e:
                    print(f"❌ Error stopping agents: {e}")
            
            # Stop server
            if self.server:
                try:
                    print("🛑 Stopping orchestrator server...")
                    await asyncio.wait_for(
                        self.server.stop(),
                        timeout=10.0
                    )
                    print("✅ Orchestrator stopped")
                except asyncio.TimeoutError:
                    print("⚠️  Timeout during server shutdown, forcing exit")
                except Exception as e:
                    print(f"❌ Error during server shutdown: {e}")
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            print(f"\n📡 Received signal {signum}")
            if self.running:
                # Create a new event loop task to handle shutdown
                asyncio.create_task(self.stop())
        
        # Handle SIGINT (Ctrl+C) and SIGTERM
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="NeuroDeck Orchestrator Daemon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m neurodeck.orchestrator
  python -m neurodeck.orchestrator --port 8888
  python -m neurodeck.orchestrator --config /path/to/config.ini
        """
    )
    
    parser.add_argument(
        "--config",
        default="config/agents.ini",
        help="Path to configuration file (default: config/agents.ini)"
    )
    
    parser.add_argument(
        "--host",
        help="Override host from config file"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        help="Override port from config file"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="NeuroDeck Orchestrator 0.1.0"
    )
    
    args = parser.parse_args()
    
    # Validate config file exists
    if not Path(args.config).exists():
        print(f"❌ Configuration file not found: {args.config}")
        sys.exit(1)
    
    # Create daemon instance
    daemon = OrchestratorDaemon(args.config)
    daemon.setup_signal_handlers()
    
    # Run daemon
    try:
        asyncio.run(daemon.start(host=args.host, port=args.port))
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except asyncio.CancelledError:
        # Expected during graceful shutdown
        pass
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()