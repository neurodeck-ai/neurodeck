"""
Chat introspection tool for NeuroDeck.

Provides agents with dynamic information about the chat environment,
including participant discovery and status information.
"""

import json
from typing import List, Dict, Any, Optional
from ..common.logging import get_logger
from ..common.config import ConfigManager

logger = get_logger(__name__)

class ChatInfoTool:
    """
    Chat introspection tool for dynamic participant discovery.
    
    Features:
    - Dynamic participant list discovery
    - Real-time connection status
    - Agent self-introspection
    - Flexible chat environment queries
    """
    
    def __init__(self, orchestrator_server=None):
        """
        Initialize chat info tool.
        
        Args:
            orchestrator_server: Reference to orchestrator server for live data
        """
        self.orchestrator_server = orchestrator_server
        logger.info("Chat info tool initialized")
    
    async def execute(self, action: str, **kwargs) -> Any:
        """
        Execute chat info operation.
        
        Args:
            action: Operation to perform (list_participants, get_chat_status, get_my_info)
            **kwargs: Additional arguments based on action
            
        Returns:
            Operation result
        """
        logger.info(f"Executing chat_info {action}")
        
        if action == "list_participants":
            return await self._list_participants()
        elif action == "get_chat_status":
            return await self._get_chat_status()
        elif action == "get_my_info":
            agent_name = kwargs.get("agent_name")
            return await self._get_my_info(agent_name)
        elif action == "get_participant_info":
            participant_name = kwargs.get("participant_name")
            return await self._get_participant_info(participant_name)
        else:
            raise ValueError(f"Unknown chat_info action: {action}")
    
    async def _list_participants(self) -> Dict[str, Any]:
        """Get list of all available participants with their status."""
        participants = []
        
        # Always include human user
        human_connected = self._is_user_connected()
        participants.append({
            "name": "human",
            "type": "user",
            "status": "connected" if human_connected else "unknown"
        })
        
        # Get configured agents
        try:
            config_manager = ConfigManager("config/agents.ini")
            agent_configs = config_manager.load_agent_configs()
            
            for agent_name, config in agent_configs.items():
                # Get connection status from orchestrator if available
                status = "unknown"
                if self.orchestrator_server:
                    status = self._get_agent_connection_status(agent_name)
                
                participants.append({
                    "name": agent_name,
                    "type": "agent",
                    "status": status,
                    "provider": config.provider,
                    "model": config.model
                })
                
        except Exception as e:
            logger.error(f"Error loading agent configs: {e}")
            # Fallback to connected agents only
            if self.orchestrator_server:
                connected_agents = self._get_connected_agents()
                for agent_name in connected_agents:
                    participants.append({
                        "name": agent_name,
                        "type": "agent", 
                        "status": "connected"
                    })
        
        logger.info(f"Found {len(participants)} participants")
        return {
            "participants": participants,
            "total_count": len(participants),
            "connected_count": len([p for p in participants if p["status"] == "connected"])
        }
    
    async def _get_chat_status(self) -> Dict[str, Any]:
        """Get overall chat environment status."""
        participants_info = await self._list_participants()
        
        # Calculate statistics
        total_participants = participants_info["total_count"]
        connected_participants = participants_info["connected_count"]
        agents = [p for p in participants_info["participants"] if p["type"] == "agent"]
        connected_agents = [p for p in agents if p["status"] == "connected"]
        
        return {
            "total_participants": total_participants,
            "connected_participants": connected_participants,
            "total_agents": len(agents),
            "connected_agents": len(connected_agents),
            "user_connected": self._is_user_connected(),
            "agent_list": [agent["name"] for agent in agents],
            "connected_agent_list": [agent["name"] for agent in connected_agents]
        }
    
    async def _get_my_info(self, agent_name: str) -> Dict[str, Any]:
        """Get information about the requesting agent."""
        if not agent_name:
            raise ValueError("Agent name is required for get_my_info")
        
        try:
            config_manager = ConfigManager("config/agents.ini")
            agent_configs = config_manager.load_agent_configs()
            
            if agent_name not in agent_configs:
                raise ValueError(f"Agent '{agent_name}' not found in configuration")
            
            config = agent_configs[agent_name]
            
            # Get connection status
            status = "unknown"
            if self.orchestrator_server:
                status = self._get_agent_connection_status(agent_name)
            
            # Get tool list
            tools = config.tools if config.tools else []
            
            return {
                "name": agent_name,
                "type": "agent",
                "status": status,
                "provider": config.provider,
                "model": config.model,
                "api_endpoint": config.api_endpoint,
                "tools": tools,
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
                "max_context_messages": config.max_context_messages
            }
            
        except Exception as e:
            logger.error(f"Error getting agent info for {agent_name}: {e}")
            raise ValueError(f"Failed to get agent info: {str(e)}")
    
    async def _get_participant_info(self, participant_name: str) -> Dict[str, Any]:
        """Get information about a specific participant."""
        if not participant_name:
            raise ValueError("Participant name is required")
        
        if participant_name == "human":
            return {
                "name": "human",
                "type": "user",
                "status": "connected" if self._is_user_connected() else "unknown"
            }
        else:
            # Assume it's an agent
            return await self._get_my_info(participant_name)
    
    def _is_user_connected(self) -> bool:
        """Check if human user is connected."""
        if not self.orchestrator_server:
            return False
        
        # Check for UI connections
        for client_id, connection in self.orchestrator_server.clients.items():
            if connection.client_type == "ui" and connection.authenticated:
                return True
        return False
    
    def _get_agent_connection_status(self, agent_name: str) -> str:
        """Get connection status for specific agent."""
        if not self.orchestrator_server:
            return "unknown"
        
        # Check for agent connections
        for client_id, connection in self.orchestrator_server.clients.items():
            if (connection.client_type == "agent" and 
                connection.agent_name == agent_name and 
                connection.authenticated):
                return "connected"
        return "disconnected"
    
    def _get_connected_agents(self) -> List[str]:
        """Get list of currently connected agent names."""
        if not self.orchestrator_server:
            return []
        
        connected_agents = []
        for client_id, connection in self.orchestrator_server.clients.items():
            if (connection.client_type == "agent" and 
                connection.authenticated and 
                connection.agent_name):
                connected_agents.append(connection.agent_name)
        
        return connected_agents