"""
TTS Configuration tool for NeuroDeck agents.

Allows agents to dynamically modify their text-to-speech parameters during conversations,
including speech speed, naturalness, and timing characteristics.
"""

import json
from typing import Dict, Any, Optional, Union
from pathlib import Path
from ..common.logging import get_logger
from ..common.config import ConfigManager

logger = get_logger(__name__)

class TTSConfigTool:
    """
    TTS configuration tool for dynamic voice parameter adjustment.
    
    Features:
    - Get current voice settings for the agent
    - Set voice parameters with validation
    - List available voice IDs
    - Reset to default configuration values
    - Real-time parameter updates
    """
    
    # Parameter validation ranges based on TTS_VITS_PARAMETERS_GUIDE.md
    PARAMETER_RANGES = {
        "length_scale": {"min": 0.5, "max": 2.0, "default": 1.0},
        "inference_noise_scale": {"min": 0.0, "max": 1.5, "default": 0.667},
        "inference_noise_scale_dp": {"min": 0.0, "max": 1.5, "default": 0.8}
    }
    
    # Available VCTK voices (from tts_service.py model config)
    AVAILABLE_VOICES = [
        "p225", "p226", "p227", "p228", "p229", "p230", "p231", "p232", 
        "p233", "p234", "p236", "p237", "p238", "p239", "p240", "p241", 
        "p243", "p244", "p245", "p246", "p247", "p248", "p249", "p250", 
        "p251", "p252", "p253", "p254", "p255", "p256", "p257", "p258", 
        "p259", "p260", "p261", "p262", "p263", "p264", "p265", "p266", 
        "p267", "p268", "p269", "p270", "p271", "p272", "p273", "p274", "p275"
    ]
    
    def __init__(self, tts_service=None):
        """
        Initialize TTS config tool.
        
        Args:
            tts_service: Optional reference to TTSService for real-time updates
        """
        self.tts_service = tts_service
        self.session_overrides = {}  # Store session-specific parameter overrides
        self.session_file = Path("config/tts_session_overrides.json")  # Shared state file
        self._load_session_overrides()
        logger.info("TTS config tool initialized")
    
    async def execute(self, action: str, agent_name: str = None, **kwargs) -> Any:
        """
        Execute TTS configuration operation.
        
        Args:
            action: Operation to perform
            agent_name: Name of the requesting agent
            **kwargs: Additional arguments based on action
            
        Returns:
            Operation result
        """
        logger.info(f"Executing tts_config action='{action}' for agent {agent_name}, kwargs={kwargs}")
        
        # Handle None or missing action
        if action is None:
            logger.error(f"TTS config called with action=None, kwargs={kwargs}")
            # Check if action is in kwargs (common mistake)
            if "action" in kwargs:
                action = kwargs.pop("action")
                logger.info(f"Found action in kwargs: '{action}'")
            else:
                raise ValueError("TTS config action parameter is required. Available actions: get_voice_settings, set_voice_settings, list_available_voices, reset_voice_settings, get_help")
        
        # Actions that don't require agent name
        if action == "get_help" or action == "help":
            return await self._get_help()
        elif action == "list_available_voices":
            return await self._list_available_voices()
        
        # Actions that require agent name
        if not agent_name:
            raise ValueError("Agent name is required for TTS configuration")
        
        if action == "get_voice_settings":
            return await self._get_voice_settings(agent_name)
        elif action == "set_voice_settings":
            return await self._set_voice_settings(agent_name, **kwargs)
        elif action == "reset_voice_settings":
            return await self._reset_voice_settings(agent_name)
        else:
            raise ValueError(f"Unknown tts_config action: {action}. Available actions: get_voice_settings, set_voice_settings, list_available_voices, reset_voice_settings, get_help")
    
    async def _get_voice_settings(self, agent_name: str) -> Dict[str, Any]:
        """Get current voice settings for the agent."""
        try:
            # Load agent configuration
            config_manager = ConfigManager("config/agents.ini")
            agent_configs = config_manager.load_agent_configs()
            
            if agent_name not in agent_configs:
                raise ValueError(f"Agent '{agent_name}' not found in configuration")
            
            config = agent_configs[agent_name]
            
            # Get current settings (session overrides take precedence)
            session_key = agent_name
            if session_key in self.session_overrides:
                current_settings = self.session_overrides[session_key].copy()
            else:
                current_settings = {}
            
            # Fill in values from config or defaults
            voice_settings = {
                "voice_id": getattr(config, "tts_voice", "p225") or "p225",
                "length_scale": current_settings.get("length_scale", 
                    float(getattr(config, "tts_length_scale", self.PARAMETER_RANGES["length_scale"]["default"]))),
                "inference_noise_scale": current_settings.get("inference_noise_scale",
                    float(getattr(config, "tts_inference_noise", self.PARAMETER_RANGES["inference_noise_scale"]["default"]))),
                "inference_noise_scale_dp": current_settings.get("inference_noise_scale_dp",
                    float(getattr(config, "tts_duration_noise", self.PARAMETER_RANGES["inference_noise_scale_dp"]["default"])))
            }
            
            return {
                "agent_name": agent_name,
                "current_settings": voice_settings,
                "parameter_ranges": self.PARAMETER_RANGES,
                "has_session_overrides": session_key in self.session_overrides,
                "parameter_descriptions": {
                    "length_scale": f"Speech speed: {voice_settings['length_scale']:.2f} (0.5=2x faster, 1.0=normal, 2.0=2x slower)",
                    "inference_noise_scale": f"Expressiveness: {voice_settings['inference_noise_scale']:.2f} (0.0=robotic, 0.667=natural, 1.5=very expressive)",
                    "inference_noise_scale_dp": f"Timing variation: {voice_settings['inference_noise_scale_dp']:.2f} (0.0=mechanical, 0.8=natural, 1.5=dramatic)",
                    "voice_id": f"Voice: {voice_settings['voice_id']} (use list_available_voices to see options)"
                },
                "quick_tips": [
                    "Use tts_config.execute('get_help') for comprehensive usage guide",
                    "Start with small changes (±0.2) to hear differences",
                    "Speed (length_scale) has the most noticeable effect"
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting voice settings for {agent_name}: {e}")
            raise ValueError(f"Failed to get voice settings: {str(e)}")
    
    async def _set_voice_settings(self, agent_name: str, **kwargs) -> Dict[str, Any]:
        """Set voice parameters with validation."""
        try:
            # Extract parameters from kwargs
            length_scale = kwargs.get("length_scale")
            inference_noise_scale = kwargs.get("inference_noise_scale")
            inference_noise_scale_dp = kwargs.get("inference_noise_scale_dp")
            voice_id = kwargs.get("voice_id")
            
            # Validate parameters
            updates = {}
            validation_errors = []
            
            if length_scale is not None:
                if self._validate_parameter("length_scale", length_scale):
                    updates["length_scale"] = float(length_scale)
                else:
                    validation_errors.append(f"length_scale must be between {self.PARAMETER_RANGES['length_scale']['min']} and {self.PARAMETER_RANGES['length_scale']['max']}")
            
            if inference_noise_scale is not None:
                if self._validate_parameter("inference_noise_scale", inference_noise_scale):
                    updates["inference_noise_scale"] = float(inference_noise_scale)
                else:
                    validation_errors.append(f"inference_noise_scale must be between {self.PARAMETER_RANGES['inference_noise_scale']['min']} and {self.PARAMETER_RANGES['inference_noise_scale']['max']}")
            
            if inference_noise_scale_dp is not None:
                if self._validate_parameter("inference_noise_scale_dp", inference_noise_scale_dp):
                    updates["inference_noise_scale_dp"] = float(inference_noise_scale_dp)
                else:
                    validation_errors.append(f"inference_noise_scale_dp must be between {self.PARAMETER_RANGES['inference_noise_scale_dp']['min']} and {self.PARAMETER_RANGES['inference_noise_scale_dp']['max']}")
            
            if voice_id is not None:
                if voice_id in self.AVAILABLE_VOICES:
                    updates["voice_id"] = voice_id
                else:
                    validation_errors.append(f"voice_id must be one of: {', '.join(self.AVAILABLE_VOICES[:10])}... (use list_available_voices for full list)")
            
            if validation_errors:
                return {
                    "success": False,
                    "error": "Validation failed",
                    "validation_errors": validation_errors,
                    "parameter_ranges": self.PARAMETER_RANGES
                }
            
            if not updates:
                return {
                    "success": False,
                    "error": "No valid parameters provided to update",
                    "available_parameters": ["length_scale", "inference_noise_scale", "inference_noise_scale_dp", "voice_id"]
                }
            
            # Store session overrides
            session_key = agent_name
            if session_key not in self.session_overrides:
                self.session_overrides[session_key] = {}
            
            self.session_overrides[session_key].update(updates)
            
            # Save session overrides to file for TTS service to pick up
            self._save_session_overrides()
            
            # Update TTS service if available
            if self.tts_service:
                await self._update_tts_service(agent_name, self.session_overrides[session_key])
            
            # Get updated settings to return
            updated_settings = await self._get_voice_settings(agent_name)
            
            logger.info(f"Updated TTS settings for {agent_name}: {updates}")
            
            return {
                "success": True,
                "message": f"Updated TTS settings for {agent_name}",
                "updated_parameters": updates,
                "current_settings": updated_settings["current_settings"]
            }
            
        except Exception as e:
            logger.error(f"Error setting voice settings for {agent_name}: {e}")
            raise ValueError(f"Failed to set voice settings: {str(e)}")
    
    async def _list_available_voices(self) -> Dict[str, Any]:
        """List all available voice IDs with descriptions."""
        return {
            "available_voices": self.AVAILABLE_VOICES,
            "total_count": len(self.AVAILABLE_VOICES),
            "voice_info": {
                "source": "VCTK Dataset",
                "description": "Multi-speaker English voices with different accents and characteristics",
                "note": "Each voice has unique personality traits. Switch voices to match your character or mood.",
                "popular_voices": {
                    "p225": "Female, English, Southern England - Clear, professional",
                    "p232": "Male, English - Deep, authoritative", 
                    "p251": "Male, Indian accent - Warm, expressive",
                    "p254": "Male, American - Friendly, conversational",
                    "p270": "Male, English - Smooth, articulate",
                    "p272": "Male, English - Educated, scholarly",
                    "p275": "Male, English - Young, energetic"
                },
                "choosing_tips": [
                    "Try different voices to find one that matches your personality",
                    "Female voices: p225, p226, p229, p230, p231, p233, p236, p244, p248, p250, p253, p256, p261, p262, p268, p275",
                    "Male voices: All others (majority are male voices)",
                    "Accents: Mix of British, American, Irish, and international accents",
                    "Use set_voice_settings with voice_id parameter to switch"
                ]
            },
            "usage_example": "tts_config.execute('set_voice_settings', voice_id='p272') # Switch to educated male voice"
        }
    
    async def _reset_voice_settings(self, agent_name: str) -> Dict[str, Any]:
        """Reset voice settings to configuration defaults."""
        try:
            session_key = agent_name
            
            # Clear session overrides
            if session_key in self.session_overrides:
                del self.session_overrides[session_key]
                logger.info(f"Cleared session overrides for {agent_name}")
            
            # Save session overrides to file
            self._save_session_overrides()
            
            # Update TTS service to use config defaults
            if self.tts_service:
                await self._update_tts_service(agent_name, {})
            
            # Return current settings (now using defaults)
            updated_settings = await self._get_voice_settings(agent_name)
            
            return {
                "success": True,
                "message": f"Reset TTS settings to defaults for {agent_name}",
                "current_settings": updated_settings["current_settings"]
            }
            
        except Exception as e:
            logger.error(f"Error resetting voice settings for {agent_name}: {e}")
            raise ValueError(f"Failed to reset voice settings: {str(e)}")
    
    async def _get_help(self) -> Dict[str, Any]:
        """Get comprehensive help and usage instructions for the TTS config tool."""
        return {
            "tool_name": "tts_config",
            "description": "Dynamically adjust your text-to-speech voice parameters during conversations to match your personality, mood, or speaking style.",
            
            "available_actions": {
                "get_voice_settings": "Get your current voice parameters and ranges",
                "set_voice_settings": "Modify your voice parameters with validation", 
                "list_available_voices": "See all available voice IDs you can switch to",
                "reset_voice_settings": "Reset all parameters to your default configuration",
                "get_help": "Show this comprehensive usage guide"
            },
            
            "voice_parameters": {
                "length_scale": {
                    "description": "Controls speech speed (how fast or slow you speak)",
                    "range": "0.5 to 2.0",
                    "default": 1.0,
                    "effects": {
                        "0.5": "2x faster speech - energetic, excited, rushed",
                        "0.8": "25% faster - brisk, confident, alert", 
                        "1.0": "Normal speed - natural, conversational",
                        "1.2": "20% slower - thoughtful, deliberate, wise",
                        "2.0": "2x slower - very dramatic, emphasizing importance"
                    },
                    "personality_suggestions": {
                        "energetic/excited": "0.8-0.9",
                        "confident/assertive": "0.9-1.0", 
                        "thoughtful/scholarly": "1.1-1.3",
                        "dramatic/emphatic": "1.3-1.5"
                    }
                },
                
                "inference_noise_scale": {
                    "description": "Controls naturalness and expressiveness (how human-like vs robotic)",
                    "range": "0.0 to 1.5", 
                    "default": 0.667,
                    "effects": {
                        "0.0": "Completely robotic - identical delivery every time",
                        "0.3": "Very consistent - corporate, professional, reliable",
                        "0.667": "Natural balance - human-like but stable",
                        "1.0": "Expressive - varied, emotional, engaging",
                        "1.5": "Very expressive - dramatic, artistic, potentially inconsistent"
                    },
                    "personality_suggestions": {
                        "professional/corporate": "0.3-0.5",
                        "friendly/conversational": "0.6-0.8",
                        "expressive/emotional": "0.9-1.2", 
                        "artistic/dramatic": "1.2-1.5"
                    }
                },
                
                "inference_noise_scale_dp": {
                    "description": "Controls timing and rhythm variation (pauses, pacing, emphasis)",
                    "range": "0.0 to 1.5",
                    "default": 0.8,
                    "effects": {
                        "0.0": "Mechanical timing - perfectly regular rhythm",
                        "0.5": "Slight variation - subtle natural pauses",
                        "0.8": "Natural timing - human-like rhythm and pauses", 
                        "1.2": "Dynamic timing - expressive pauses for emphasis",
                        "1.5": "Artistic timing - dramatic pauses, varied pacing"
                    },
                    "personality_suggestions": {
                        "academic/precise": "0.5-0.7",
                        "conversational/natural": "0.7-0.9",
                        "expressive/dynamic": "1.0-1.3",
                        "theatrical/dramatic": "1.3-1.5"
                    }
                },
                
                "voice_id": {
                    "description": "Your base voice characteristics (accent, gender, tone)",
                    "note": "Use list_available_voices to see all options with descriptions"
                }
            },
            
            "usage_examples": {
                "getting_started": {
                    "description": "Check your current settings first",
                    "example": "tts_config.execute('get_voice_settings')"
                },
                
                "become_more_energetic": {
                    "description": "Speed up and add expressiveness for exciting topics",
                    "example": "tts_config.execute('set_voice_settings', length_scale=0.8, inference_noise_scale=1.0)",
                    "effect": "25% faster speech with more natural variation"
                },
                
                "thoughtful_mode": {
                    "description": "Slower, more deliberate for serious discussions", 
                    "example": "tts_config.execute('set_voice_settings', length_scale=1.2, inference_noise_scale=0.5, inference_noise_scale_dp=0.6)",
                    "effect": "Slower, consistent, precise delivery"
                },
                
                "dramatic_emphasis": {
                    "description": "Make important points stand out with timing",
                    "example": "tts_config.execute('set_voice_settings', inference_noise_scale_dp=1.3)",
                    "effect": "Adds dramatic pauses and emphasis"
                },
                
                "reset_to_normal": {
                    "description": "Return to your default configuration",
                    "example": "tts_config.execute('reset_voice_settings')"
                }
            },
            
            "personality_presets": {
                "energetic_friend": {
                    "description": "Fast, expressive, engaging",
                    "settings": {"length_scale": 0.8, "inference_noise_scale": 1.0, "inference_noise_scale_dp": 1.0}
                },
                "wise_mentor": {
                    "description": "Slower, thoughtful, deliberate",
                    "settings": {"length_scale": 1.2, "inference_noise_scale": 0.6, "inference_noise_scale_dp": 0.7}
                },
                "professional_assistant": {
                    "description": "Consistent, reliable, clear",
                    "settings": {"length_scale": 1.0, "inference_noise_scale": 0.4, "inference_noise_scale_dp": 0.6}
                },
                "dramatic_storyteller": {
                    "description": "Expressive, varied timing, engaging",
                    "settings": {"length_scale": 1.1, "inference_noise_scale": 1.2, "inference_noise_scale_dp": 1.3}
                }
            },
            
            "tips": [
                "Start with small adjustments (±0.2 from current values) to hear the difference",
                "Speed changes (length_scale) have the most immediate noticeable effect",
                "Higher expressiveness (inference_noise_scale) makes you sound more human and emotional",
                "Timing variation (inference_noise_scale_dp) adds dramatic pauses and emphasis",
                "You can change parameters mid-conversation to match the topic or mood",
                "Use reset_voice_settings to return to your defaults anytime",
                "Extreme values (near min/max) may sound unnatural - test carefully"
            ],
            
            "when_to_adjust": {
                "exciting_news": "Increase speed (0.8) and expressiveness (1.0+)",
                "serious_topics": "Slow down (1.2+) and reduce variation for clarity", 
                "storytelling": "Add timing variation (1.2+) for dramatic effect",
                "technical_explanations": "Moderate speed (0.9-1.1) with clear timing",
                "emotional_moments": "Increase expressiveness (1.0+) to convey feeling",
                "jokes_or_humor": "Faster pace (0.8-0.9) with good timing variation"
            }
        }
    
    def _validate_parameter(self, param_name: str, value: Union[int, float]) -> bool:
        """Validate parameter value against defined ranges."""
        try:
            value = float(value)
            range_info = self.PARAMETER_RANGES.get(param_name)
            if not range_info:
                return False
            return range_info["min"] <= value <= range_info["max"]
        except (ValueError, TypeError):
            return False
    
    async def _update_tts_service(self, agent_name: str, overrides: Dict[str, Any]):
        """Update TTS service with new parameters."""
        if not self.tts_service:
            return
        
        try:
            # Update the agent's config in TTS service
            if hasattr(self.tts_service, 'update_agent_config'):
                await self.tts_service.update_agent_config(agent_name, overrides)
            else:
                # Fallback: directly update agent_configs if method doesn't exist
                if hasattr(self.tts_service, 'agent_configs') and agent_name in self.tts_service.agent_configs:
                    # Create a temporary config object with overrides
                    class TempConfig:
                        def __init__(self, base_config, overrides):
                            self.base_config = base_config
                            self.overrides = overrides
                        
                        def get(self, key, default=None):
                            # Check overrides first, then base config
                            if key in self.overrides:
                                return self.overrides[key]
                            return getattr(self.base_config, key, default)
                        
                        def __getattr__(self, name):
                            if name in self.overrides:
                                return self.overrides[name]
                            return getattr(self.base_config, name)
                    
                    # Store the override parameters for the TTS service to use
                    if not hasattr(self.tts_service, 'session_overrides'):
                        self.tts_service.session_overrides = {}
                    self.tts_service.session_overrides[agent_name] = overrides
            
            logger.info(f"Updated TTS service config for {agent_name}")
            
        except Exception as e:
            logger.warning(f"Could not update TTS service for {agent_name}: {e}")
    
    def _load_session_overrides(self):
        """Load session overrides from file."""
        try:
            if self.session_file.exists():
                with open(self.session_file, 'r') as f:
                    self.session_overrides = json.load(f)
                logger.info(f"Loaded {len(self.session_overrides)} session overrides from file")
            else:
                self.session_overrides = {}
        except Exception as e:
            logger.warning(f"Failed to load session overrides: {e}")
            self.session_overrides = {}
    
    def _save_session_overrides(self):
        """Save session overrides to file."""
        try:
            # Ensure config directory exists
            self.session_file.parent.mkdir(exist_ok=True)
            
            with open(self.session_file, 'w') as f:
                json.dump(self.session_overrides, f, indent=2)
            logger.debug(f"Saved {len(self.session_overrides)} session overrides to file")
        except Exception as e:
            logger.error(f"Failed to save session overrides: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get TTS config tool status."""
        return {
            "active_overrides": len(self.session_overrides),
            "agents_with_overrides": list(self.session_overrides.keys()),
            "tts_service_connected": self.tts_service is not None,
            "available_voices_count": len(self.AVAILABLE_VOICES),
            "session_file": str(self.session_file)
        }