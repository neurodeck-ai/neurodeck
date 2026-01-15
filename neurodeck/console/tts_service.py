"""
TTS Service for NeuroDeck Console
Provides text-to-speech functionality with unique voices per agent and FIFO message queuing.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Any, Callable
import numpy as np
from pathlib import Path
import sys
import io
import os
import json
from contextlib import redirect_stdout, redirect_stderr

from ..common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TTSMessage:
    """Data structure for TTS message queue."""
    agent_name: str
    content: str
    voice_id: str
    message_id: Optional[str] = None


class TTSService:
    """
    Text-to-Speech service with async queue processing and unique agent voices.
    """
    
    def __init__(self, agent_voices: Dict[str, str], agent_configs: Optional[Dict[str, Any]] = None, log_callback: Optional[Callable[[str, str, str], None]] = None):
        """
        Initialize TTS service.
        
        Args:
            agent_voices: Mapping of agent names to voice IDs (e.g., {"claudius": "p225"})
            agent_configs: Optional mapping of agent names to their full config objects
            log_callback: Optional callback function for logging to TUI debug panel
        """
        self.agent_voices = agent_voices
        self.agent_configs = agent_configs or {}
        self.enabled = False
        self.tts_model = None
        self.message_queue = asyncio.Queue()
        self.processing_task = None
        self.sample_rate = 22050
        self.log_callback = log_callback
        
        # Session-specific parameter overrides for dynamic TTS config
        self.session_overrides = {}
        self.session_file = Path("config/tts_session_overrides.json")
        
        # Audio playback
        self.sd = None
        
        # Note: Logging deferred until TTS actually starts to avoid UI timing issues
    
    def _log(self, sender: str, message: str, msg_type: str = "system-info"):
        """
        Log a message either to TUI debug panel or fallback to logger.
        
        Args:
            sender: Source of the log message
            message: Log message content  
            msg_type: Message type for styling (system-info, system-error, system-warning, system-success)
        """
        if self.log_callback:
            try:
                self.log_callback(sender, message, msg_type)
            except Exception:
                # If callback fails, fallback to logger to avoid losing critical info
                logger.warning(f"TTS log callback failed: {sender}: {message}")
        # For critical errors, always log to file regardless of callback
        elif msg_type == "system-error":
            logger.error(f"{sender}: {message}")
    
    def _initialize_sync(self) -> tuple:
        """
        Synchronous TTS initialization - runs in thread pool.

        Returns:
            tuple: (success, tts_model, sd_module, model_config, loaded_model, error_msg)
        """
        try:
            # Import TTS libraries (this alone can be slow)
            from TTS.api import TTS
            import sounddevice as sd

            # Configure TTS logging to prevent output interference
            import logging as tts_logging
            tts_logging.getLogger('TTS').setLevel(tts_logging.CRITICAL)
            tts_logging.getLogger('TTS.tts').setLevel(tts_logging.CRITICAL)
            tts_logging.getLogger('TTS.utils').setLevel(tts_logging.CRITICAL)
            tts_logging.getLogger('TTS.vocoder').setLevel(tts_logging.CRITICAL)

            # Define available models (VCTK first since we need multi-speaker support)
            model_configs = {
                "vctk": {
                    "model_name": "tts_models/en/vctk/vits",
                    "description": "VCTK - Multiple English speakers",
                    "speakers": ["p225", "p226", "p227", "p228", "p229", "p230", "p231", "p232", "p233", "p234", "p236", "p237", "p238", "p239", "p240", "p241", "p243", "p244", "p245", "p246", "p247", "p248", "p249", "p250", "p251", "p252", "p253", "p254", "p255", "p256", "p257", "p258", "p259", "p260", "p261", "p262", "p263", "p264", "p265", "p266", "p267", "p268", "p269", "p270", "p271", "p272", "p273", "p274", "p275"],
                    "sample_rate": 22050
                },
                "jenny": {
                    "model_name": "tts_models/en/ljspeech/tacotron2-DDC",
                    "description": "Jenny (LJSpeech) - High quality female English voice",
                    "speakers": ["jenny"],
                    "sample_rate": 22050
                },
                "blizzard": {
                    "model_name": "tts_models/en/blizzard2013/capacitron-t2-c50",
                    "description": "Blizzard 2013 - Natural sounding English voice",
                    "speakers": ["blizzard"],
                    "sample_rate": 22050
                }
            }

            # Try to load models in order of preference (VCTK first)
            loaded_model = None
            tts_model = None
            model_config = None

            for model_key, config in model_configs.items():
                try:
                    # Suppress stdout/stderr during model loading
                    original_stdout = sys.stdout
                    original_stderr = sys.stderr

                    try:
                        devnull_out = open(os.devnull, 'w')
                        devnull_err = open(os.devnull, 'w')
                        sys.stdout = devnull_out
                        sys.stderr = devnull_err

                        import builtins
                        original_print = builtins.print
                        builtins.print = lambda *args, **kwargs: None

                        try:
                            tts_model = TTS(config["model_name"])
                        finally:
                            builtins.print = original_print

                    finally:
                        sys.stdout = original_stdout
                        sys.stderr = original_stderr
                        devnull_out.close()
                        devnull_err.close()

                    model_config = config
                    loaded_model = model_key
                    break
                except Exception:
                    continue

            if not loaded_model:
                return (False, None, None, None, None, "No TTS models could be loaded")

            return (True, tts_model, sd, model_config, loaded_model, None)

        except ImportError as e:
            return (False, None, None, None, None, f"Dependencies not available: {e}")
        except Exception as e:
            return (False, None, None, None, None, f"Initialization failed: {e}")

    async def initialize(self) -> bool:
        """
        Initialize TTS models and audio system (non-blocking).

        Returns:
            bool: True if initialization successful
        """
        self._log("TTS", f"Service starting with voices: {self.agent_voices}", "system-info")
        self._log("TTS", "Loading TTS model in background thread...", "system-info")

        # Run the blocking initialization in a thread pool
        loop = asyncio.get_event_loop()
        success, tts_model, sd, model_config, loaded_model, error_msg = await loop.run_in_executor(
            None, self._initialize_sync
        )

        if not success:
            self._log("TTS", error_msg, "system-error")
            if "Dependencies" in error_msg:
                self._log("TTS", "Install with: pip install coqui-tts sounddevice", "system-info")
            return False

        # Store results
        self.tts_model = tts_model
        self.sd = sd
        self.model_config = model_config

        self._log("TTS", f"{loaded_model} model loaded successfully", "system-success")

        # Configure audio system
        try:
            self.sd.default.samplerate = self.sample_rate
            self.sd.default.blocksize = 2048
            self.sd.check_output_settings()
            self._log("TTS", "Audio system configured successfully", "system-success")
        except Exception as e:
            self._log("TTS", f"Audio system configuration warning: {e}", "system-warning")

        self._log("TTS", f"System initialized with {loaded_model}", "system-success")
        return True
    
    async def start_service(self) -> bool:
        """
        Start the TTS service and processing task.
        
        Returns:
            bool: True if service started successfully
        """
        if not await self.initialize():
            return False
        
        if self.processing_task is None or self.processing_task.done():
            self.processing_task = asyncio.create_task(self._process_queue())
            self._log("TTS", "Processing task started", "system-info")
        
        return True
    
    async def stop_service(self):
        """Stop the TTS service and cleanup."""
        self.enabled = False
        
        if self.processing_task and not self.processing_task.done():
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        
        # Clear any remaining messages in queue
        while not self.message_queue.empty():
            try:
                self.message_queue.get_nowait()
                self.message_queue.task_done()
            except asyncio.QueueEmpty:
                break
        
        self._log("TTS", "Service stopped", "system-info")
    
    def enable_tts(self):
        """Enable TTS processing."""
        self.enabled = True
        self._log("TTS", "Enabled", "system-info")
    
    def disable_tts(self):
        """Disable TTS processing."""
        self.enabled = False
        self._log("TTS", "Disabled", "system-info")
    
    def is_enabled(self) -> bool:
        """Check if TTS is enabled."""
        return self.enabled
    
    def is_ready(self) -> bool:
        """Check if TTS service is ready to process messages."""
        return (self.tts_model is not None and 
                self.processing_task is not None and 
                not self.processing_task.done())
    
    async def queue_message(self, agent_name: str, content: str, message_id: Optional[str] = None):
        """
        Queue a message for TTS processing.
        
        Args:
            agent_name: Name of the agent (must be in agent_voices mapping)
            content: Text content to speak
            message_id: Optional message identifier for logging
        """
        if not self.enabled or not self.is_ready():
            return
        
        # Get voice for agent
        voice_id = self.agent_voices.get(agent_name)
        if not voice_id:
            self._log("TTS", f"No voice configured for agent: {agent_name}", "system-warning")
            return
        
        # Clean content (remove markdown, thinking tags, etc.)
        cleaned_content = self._clean_content(content)
        
        if not cleaned_content.strip():
            # Skip debug logging for empty content to reduce noise
            return
        
        # Create TTS message
        tts_message = TTSMessage(
            agent_name=agent_name,
            content=cleaned_content,
            voice_id=voice_id,
            message_id=message_id
        )
        
        # Queue message for processing
        await self.message_queue.put(tts_message)
        # NOTE: Removed debug logging to prevent UI corruption from TTS threads
    
    def _clean_content(self, content: str) -> str:
        """
        Clean content for TTS by removing markdown, thinking tags, etc.
        Also adds natural emphasis and inflection.
        
        Args:
            content: Raw content string
            
        Returns:
            str: Cleaned content suitable for TTS with emphasis
        """
        import re
        
        # Remove <think> tags and their content
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        
        # Convert @mentions to natural speech (remove @ symbol)
        content = re.sub(r'@(\w+)', r'\1', content)  # @human -> human, @grok -> grok
        
        # Convert markdown formatting to emphasis
        content = re.sub(r'\*\*(.*?)\*\*', r'\1!', content)  # **bold** -> word!
        content = re.sub(r'\*(.*?)\*', r'\1', content)       # *italic* -> word
        content = re.sub(r'`(.*?)`', r'\1', content)         # `code` -> code
        content = re.sub(r'```.*?```', '', content, flags=re.DOTALL)  # Remove code blocks
        
        # Remove URLs but keep their context
        content = re.sub(r'https?://\S+', 'link', content)
        
        # Add emphasis for excitement and strong emotions
        excitement_words = ['amazing', 'awesome', 'fantastic', 'incredible', 'wow', 'cool', 'great', 'love', 'brilliant']
        for word in excitement_words:
            # Add exclamation for emphasis (case insensitive)
            content = re.sub(rf'\b({word})\b', r'\1!', content, flags=re.IGNORECASE)
        
        # Convert multiple punctuation to single for better speech
        content = re.sub(r'[!]{2,}', '!', content)
        content = re.sub(r'[?]{2,}', '?', content)
        content = re.sub(r'[.]{3,}', '...', content)
        
        # Add natural pauses
        content = re.sub(r'\.\.\.', ', ', content)  # Convert ... to comma pause
        content = re.sub(r' - ', ', ', content)  # Convert dashes to pauses
        
        # Clean up excessive whitespace and newlines
        content = re.sub(r'\n+', ' ', content)
        content = re.sub(r'\s+', ' ', content)
        
        return content.strip()
    
    async def _process_queue(self):
        """
        Main queue processing loop - runs in background task.
        Processes messages in FIFO order.
        """
        self._log("TTS", "Queue processing started", "system-info")
        
        while True:
            try:
                # Wait for next message
                tts_message = await self.message_queue.get()
                
                # Skip if TTS disabled
                if not self.enabled:
                    self.message_queue.task_done()
                    continue
                
                # Process the message
                await self._synthesize_and_play(tts_message)
                
                # Show status after successful playback
                remaining_count = self.message_queue.qsize()
                if remaining_count > 0:
                    # Peek at who's next without removing from queue
                    try:
                        next_msg = self.message_queue._queue[0] if self.message_queue._queue else None
                        next_agent = next_msg.agent_name if next_msg else "unknown"
                        status_msg = f"🎵 Played {tts_message.agent_name} | Queue: {remaining_count} remaining | Next: {next_agent}"
                    except (IndexError, AttributeError):
                        status_msg = f"🎵 Played {tts_message.agent_name} | Queue: {remaining_count} remaining"
                else:
                    status_msg = f"🎵 Played {tts_message.agent_name} | Queue empty"
                self._log("TTS", status_msg, "system-info")
                
                # Mark task as done
                self.message_queue.task_done()
                
            except asyncio.CancelledError:
                self._log("TTS", "Queue processing cancelled", "system-info")
                break
            except Exception as e:
                self._log("TTS", f"Error processing TTS message: {e}", "system-error")
                # Continue processing other messages
                if not self.message_queue.empty():
                    self.message_queue.task_done()
                continue
    
    async def _synthesize_and_play(self, tts_message: TTSMessage):
        """
        Synthesize audio and play it.
        
        Args:
            tts_message: TTS message to process
        """
        try:
            # Generate audio in executor to avoid blocking
            # NOTE: Removed debug logging to prevent UI corruption from executor threads
            loop = asyncio.get_event_loop()
            audio_data = await loop.run_in_executor(
                None, 
                self._generate_audio, 
                tts_message.content, 
                tts_message.voice_id,
                tts_message.agent_name
            )
            
            if audio_data is not None:
                # Play audio
                await self._play_audio(audio_data, tts_message.agent_name)
            
        except Exception as e:
            # Only log errors, not debug info, to prevent UI corruption
            self._log("TTS", f"Synthesis failed for {tts_message.agent_name}: {e}", "system-error")
    
    def _generate_audio(self, text: str, voice_id: str, agent_name: str = "") -> Optional[np.ndarray]:
        """
        Generate audio data (runs in executor thread).
        
        Args:
            text: Text to synthesize
            voice_id: Voice ID for VCTK model
            agent_name: Name of agent for parameter lookup
            
        Returns:
            numpy.ndarray: Audio waveform or None if failed
        """
        try:
            # Check if this is a multi-speaker model
            available_speakers = self.model_config["speakers"]
            
            # Capture TTS processing output to prevent console pollution
            # This runs in an executor thread, so it won't interfere with Textual UI
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            
            try:
                # More aggressive stdout/stderr redirection for TTS library
                # Some TTS operations bypass normal redirection, so we need to be more thorough
                devnull_out = open(os.devnull, 'w')
                devnull_err = open(os.devnull, 'w')
                sys.stdout = devnull_out
                sys.stderr = devnull_err
                
                # Also try to suppress print() calls more directly
                import builtins
                original_print = builtins.print
                builtins.print = lambda *args, **kwargs: None
                
                try:
                    # Load session overrides from file (for TTS config tool integration)
                    self._load_session_overrides()
                    
                    # Get agent-specific parameters with session override support
                    agent_config = self.agent_configs.get(agent_name, None)
                    session_override = self.session_overrides.get(agent_name, {})
                    
                    # Start with config defaults, then apply session overrides
                    if agent_config:
                        length_scale = getattr(agent_config, 'tts_length_scale', 1.0)
                        inference_noise = getattr(agent_config, 'tts_inference_noise', 0.667)
                        duration_noise = getattr(agent_config, 'tts_duration_noise', 0.8)
                    else:
                        # Fallback values if agent config not found
                        length_scale = 1.0
                        inference_noise = 0.667
                        duration_noise = 0.8
                    
                    # Apply session overrides if they exist
                    length_scale = session_override.get('length_scale', length_scale)
                    inference_noise = session_override.get('inference_noise_scale', inference_noise)
                    duration_noise = session_override.get('inference_noise_scale_dp', duration_noise)
                    
                    # DEBUG: Use callback to safely log TTS parameters to UI debug panel
                    if self.log_callback:
                        try:
                            self.log_callback("TTS", f"Setting params for {agent_name}: length={length_scale:.1f}, noise={inference_noise:.2f}, duration={duration_noise:.2f}", "system-info")
                        except Exception:
                            pass  # Ignore callback errors to prevent TTS failure
                    
                    # CRITICAL FIX: Set parameters on the model instance, not as function arguments
                    # VITS models only read these parameters from instance variables, not from tts() call
                    if hasattr(self.tts_model, 'synthesizer') and hasattr(self.tts_model.synthesizer, 'tts_model'):
                        # Access the actual VITS model instance
                        vits_model = self.tts_model.synthesizer.tts_model
                        vits_model.length_scale = length_scale
                        vits_model.inference_noise_scale = inference_noise
                        vits_model.inference_noise_scale_dp = duration_noise
                    
                    if len(available_speakers) == 1:
                        # Single-speaker model (like jenny) - don't pass parameters to tts() call
                        audio = self.tts_model.tts(text)
                    else:
                        # Multi-speaker model (like VCTK) - only pass speaker parameter
                        if voice_id not in available_speakers:
                            # Can't log warning here as it would corrupt UI
                            voice_id = available_speakers[0]
                        audio = self.tts_model.tts(text, speaker=voice_id)
                finally:
                    # Restore print function
                    builtins.print = original_print
                    
            finally:
                # Always restore stdout/stderr in executor thread
                sys.stdout = original_stdout
                sys.stderr = original_stderr
                devnull_out.close()
                devnull_err.close()
            
            # Convert to numpy array if needed
            if isinstance(audio, list):
                audio = np.array(audio, dtype=np.float32)
            elif not isinstance(audio, np.ndarray):
                audio = np.array(audio, dtype=np.float32)
            
            # Ensure proper data type
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # Normalize audio to prevent clipping and distortion
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                # Leave more headroom and use gentler normalization
                audio = audio / max_val * 0.85
            
            # Remove any potential NaN or inf values
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.85, neginf=-0.85)
            
            return audio
            
        except Exception as e:
            logger.error(f"Audio generation failed for voice {voice_id}: {e}")
            return None
    
    async def _play_audio(self, audio_data: np.ndarray, agent_name: str):
        """
        Play audio data.
        
        Args:
            audio_data: Audio waveform to play
            agent_name: Name of agent for logging
        """
        try:
            # Play audio (blocking) - removed debug logging to prevent UI corruption
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._play_audio_blocking, audio_data)
            
        except Exception as e:
            logger.error(f"Audio playback failed for {agent_name}: {e}")
    
    def _play_audio_blocking(self, audio_data: np.ndarray):
        """
        Play audio in blocking mode (runs in executor).
        
        Args:
            audio_data: Audio waveform to play
        """
        # Ensure proper audio format and prevent clipping
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Normalize to prevent distortion
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val * 0.8  # Leave some headroom
        
        # Use larger buffer size to prevent stuttering
        self.sd.play(audio_data, samplerate=self.sample_rate, blocksize=2048)
        self.sd.wait()  # Wait for playback to complete
    
    def get_queue_size(self) -> int:
        """Get current queue size."""
        return self.message_queue.qsize()
    
    async def clear_queue(self) -> int:
        """
        Clear all pending messages from the TTS queue.
        
        Returns:
            int: Number of messages that were cleared from the queue
        """
        cleared_count = 0
        
        # Clear all pending messages
        while not self.message_queue.empty():
            try:
                self.message_queue.get_nowait()
                self.message_queue.task_done()
                cleared_count += 1
            except asyncio.QueueEmpty:
                break
        
        if cleared_count > 0:
            self._log("TTS", f"Cleared {cleared_count} messages from queue", "system-info")
        
        return cleared_count
    
    async def update_agent_config(self, agent_name: str, parameter_overrides: Dict[str, Any]):
        """
        Update agent TTS configuration with session-specific overrides.
        
        Args:
            agent_name: Name of the agent to update
            parameter_overrides: Dictionary of parameter overrides
        """
        if parameter_overrides:
            self.session_overrides[agent_name] = parameter_overrides.copy()
            self._log("TTS", f"Updated session overrides for {agent_name}: {parameter_overrides}", "system-info")
        elif agent_name in self.session_overrides:
            # Clear overrides if empty dict provided
            del self.session_overrides[agent_name]
            self._log("TTS", f"Cleared session overrides for {agent_name}", "system-info")
    
    def get_agent_parameters(self, agent_name: str) -> Dict[str, Any]:
        """
        Get current effective parameters for an agent (config + session overrides).
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Dict with current effective parameters
        """
        agent_config = self.agent_configs.get(agent_name, None)
        session_override = self.session_overrides.get(agent_name, {})
        
        # Start with config defaults
        if agent_config:
            length_scale = getattr(agent_config, 'tts_length_scale', 1.0)
            inference_noise = getattr(agent_config, 'tts_inference_noise', 0.667)
            duration_noise = getattr(agent_config, 'tts_duration_noise', 0.8)
            voice_id = getattr(agent_config, 'tts_voice', 'p225')
        else:
            # Fallback values
            length_scale = 1.0
            inference_noise = 0.667
            duration_noise = 0.8
            voice_id = 'p225'
        
        # Apply session overrides
        return {
            "voice_id": session_override.get('voice_id', voice_id),
            "length_scale": session_override.get('length_scale', length_scale),
            "inference_noise_scale": session_override.get('inference_noise_scale', inference_noise),
            "inference_noise_scale_dp": session_override.get('inference_noise_scale_dp', duration_noise)
        }
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get TTS service status.
        
        Returns:
            Dict with status information
        """
        return {
            "enabled": self.enabled,
            "ready": self.is_ready(),
            "queue_size": self.get_queue_size(),
            "agent_voices": self.agent_voices,
            "model_loaded": self.tts_model is not None,
            "session_overrides": self.session_overrides
        }
    
    def _load_session_overrides(self):
        """Load session overrides from file (updated by TTS config tool)."""
        try:
            if self.session_file.exists():
                with open(self.session_file, 'r') as f:
                    file_overrides = json.load(f)
                # Only update if file has newer data
                if file_overrides != self.session_overrides:
                    self.session_overrides.update(file_overrides)
                    # Note: Don't log here as it runs in TTS executor thread
        except Exception:
            # Silently fail to avoid TTS corruption
            pass