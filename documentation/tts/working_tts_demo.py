#!/usr/bin/env python3
"""
Complete Working Coqui TTS Demo with Real-time Audio
====================================================

This script demonstrates a fully functional text-to-speech system using Coqui TTS
with multiple voice models and real-time audio playback.

Features:
- Multiple TTS models (LJSpeech, VCTK, Multilingual models)
- Real-time audio playback
- Batch processing
- Interactive voice selection
- Error handling and fallbacks

Requirements:
- coqui-tts
- sounddevice
- torch
- numpy

Installation:
pip install coqui-tts sounddevice torch torchaudio

Author: Claude Code Assistant
Date: 2025-01-29
"""

import os
import time
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WorkingTTSDemo:
    """
    Complete TTS demonstration class with multiple models and features
    """
    
    def __init__(self, use_gpu: bool = False):
        """
        Initialize the TTS demo with available models
        
        Args:
            use_gpu (bool): Whether to use GPU acceleration (requires CUDA)
        """
        self.device = "cuda" if use_gpu and self._check_cuda() else "cpu"
        self.models = {}
        self.current_model = None
        
        # Import required libraries
        try:
            from TTS.api import TTS
            import sounddevice as sd
            import torch
            
            self.TTS = TTS
            self.sd = sd
            self.torch = torch
            
            logger.info("✅ All required libraries imported successfully")
            
        except ImportError as e:
            logger.error(f"❌ Failed to import required libraries: {e}")
            raise
        
        # Initialize available models
        self._initialize_models()
        
        logger.info(f"🚀 TTS Demo initialized on {self.device}")
        logger.info(f"📋 Available models: {list(self.models.keys())}")
    
    def _check_cuda(self) -> bool:
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def _initialize_models(self):
        """Initialize available TTS models"""
        
        # Model definitions with descriptions
        model_configs = {
            "jenny": {
                "model_name": "tts_models/en/ljspeech/tacotron2-DDC",
                "description": "Jenny (LJSpeech) - High quality female English voice",
                "language": "en",
                "speakers": ["jenny"],
                "sample_rate": 22050
            },
            "vctk": {
                "model_name": "tts_models/en/vctk/vits", 
                "description": "VCTK - Multiple English speakers (100+ voices)",
                "language": "en",
                "speakers": ["p225", "p226", "p227", "p228", "p229", "p230", 
                           "p231", "p232", "p233", "p234", "p236", "p237"],
                "sample_rate": 22050
            },
            "blizzard": {
                "model_name": "tts_models/en/blizzard2013/capacitron-t2-c50",
                "description": "Blizzard 2013 - Natural sounding English voice",
                "language": "en", 
                "speakers": ["blizzard"],
                "sample_rate": 22050
            }
        }
        
        # Try to load each model
        for model_key, config in model_configs.items():
            try:
                logger.info(f"🔄 Loading {config['description']}...")
                
                tts_model = self.TTS(config["model_name"]).to(self.device)
                
                self.models[model_key] = {
                    "model": tts_model,
                    "config": config
                }
                
                logger.info(f"✅ {model_key} model loaded successfully")
                
                # Set first successful model as default
                if self.current_model is None:
                    self.current_model = model_key
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to load {model_key}: {e}")
                continue
        
        if not self.models:
            raise RuntimeError("❌ No TTS models could be loaded!")
        
        logger.info(f"✅ Successfully loaded {len(self.models)} models")
    
    def list_models(self) -> Dict[str, str]:
        """
        Get list of available models and their descriptions
        
        Returns:
            Dict mapping model keys to descriptions
        """
        return {key: model["config"]["description"] 
                for key, model in self.models.items()}
    
    def list_speakers(self, model_key: Optional[str] = None) -> List[str]:
        """
        Get available speakers for a model
        
        Args:
            model_key (str): Model to check speakers for (uses current if None)
            
        Returns:
            List of available speaker names/IDs
        """
        model_key = model_key or self.current_model
        
        if model_key not in self.models:
            raise ValueError(f"Model '{model_key}' not available")
        
        return self.models[model_key]["config"]["speakers"]
    
    def set_model(self, model_key: str):
        """
        Switch to a different TTS model
        
        Args:
            model_key (str): Key of the model to switch to
        """
        if model_key not in self.models:
            available = list(self.models.keys())
            raise ValueError(f"Model '{model_key}' not available. Options: {available}")
        
        self.current_model = model_key
        logger.info(f"🔄 Switched to model: {self.models[model_key]['config']['description']}")
    
    def synthesize(self, text: str, speaker: Optional[str] = None, 
                  model_key: Optional[str] = None) -> np.ndarray:
        """
        Generate audio from text
        
        Args:
            text (str): Text to synthesize
            speaker (str): Speaker name/ID (uses default if None)
            model_key (str): Model to use (uses current if None)
            
        Returns:
            numpy.ndarray: Audio waveform data
        """
        model_key = model_key or self.current_model
        
        if model_key not in self.models:
            raise ValueError(f"Model '{model_key}' not available")
        
        model_info = self.models[model_key]
        tts_model = model_info["model"]
        config = model_info["config"]
        
        # Use provided speaker or default to first available
        if speaker is None:
            speaker = config["speakers"][0]
        elif speaker not in config["speakers"]:
            logger.warning(f"Speaker '{speaker}' not available for {model_key}, using default")
            speaker = config["speakers"][0]
        
        logger.info(f"🎙️ Synthesizing with {model_key} - Speaker: {speaker}")
        
        try:
            # Generate audio based on model capabilities
            if len(config["speakers"]) == 1:
                # Single-speaker models (like jenny)
                wav = tts_model.tts(text)
            else:
                # Multi-speaker models (like VCTK)
                wav = tts_model.tts(text, speaker=speaker)
            
            logger.info(f"✅ Audio generated - Duration: {len(wav)/config['sample_rate']:.2f}s")
            return wav
            
        except Exception as e:
            logger.error(f"❌ Synthesis failed: {e}")
            raise
    
    def play_audio(self, audio_data: np.ndarray, sample_rate: int = 22050, 
                   blocking: bool = True) -> bool:
        """
        Play audio data through speakers
        
        Args:
            audio_data (np.ndarray): Audio waveform to play
            sample_rate (int): Sample rate of the audio
            blocking (bool): Whether to wait for playback to finish
            
        Returns:
            bool: True if playback was successful
        """
        try:
            # Ensure audio is in correct format
            if isinstance(audio_data, list):
                audio_data = np.array(audio_data)
            
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Normalize audio if needed
            if np.max(np.abs(audio_data)) > 1.0:
                audio_data = audio_data / np.max(np.abs(audio_data)) * 0.95
            
            logger.info("🔊 Playing audio...")
            
            self.sd.play(audio_data, samplerate=sample_rate)
            
            if blocking:
                self.sd.wait()  # Wait for playback to finish
                logger.info("✅ Audio playback complete")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Audio playback failed: {e}")
            return False
    
    def speak(self, text: str, speaker: Optional[str] = None, 
              model_key: Optional[str] = None, play_immediately: bool = True) -> np.ndarray:
        """
        Generate and optionally play speech
        
        Args:
            text (str): Text to speak
            speaker (str): Speaker to use
            model_key (str): Model to use  
            play_immediately (bool): Whether to play audio immediately
            
        Returns:
            numpy.ndarray: Generated audio data
        """
        # Generate audio
        audio_data = self.synthesize(text, speaker, model_key)
        
        # Play if requested
        if play_immediately:
            model_key = model_key or self.current_model
            sample_rate = self.models[model_key]["config"]["sample_rate"]
            self.play_audio(audio_data, sample_rate)
        
        return audio_data
    
    def batch_speak(self, texts: List[str], speakers: Optional[List[str]] = None,
                   model_key: Optional[str] = None, delay_between: float = 0.5) -> List[np.ndarray]:
        """
        Speak multiple texts in sequence
        
        Args:
            texts (List[str]): List of texts to speak
            speakers (List[str]): List of speakers (same length as texts)
            model_key (str): Model to use
            delay_between (float): Seconds to wait between speeches
            
        Returns:
            List of generated audio arrays
        """
        if speakers and len(speakers) != len(texts):
            raise ValueError("Number of speakers must match number of texts")
        
        model_key = model_key or self.current_model
        available_speakers = self.list_speakers(model_key)
        
        if not speakers:
            # Cycle through available speakers
            speakers = [available_speakers[i % len(available_speakers)] 
                       for i in range(len(texts))]
        
        logger.info(f"🎵 Speaking {len(texts)} texts in sequence...")
        
        audio_results = []
        
        for i, (text, speaker) in enumerate(zip(texts, speakers)):
            logger.info(f"📢 Speaking {i+1}/{len(texts)}: '{text[:50]}...' with {speaker}")
            
            audio_data = self.speak(text, speaker, model_key, play_immediately=True)
            audio_results.append(audio_data)
            
            # Wait between speeches (except for last one)
            if i < len(texts) - 1 and delay_between > 0:
                time.sleep(delay_between)
        
        logger.info("🎉 Batch speaking complete!")
        return audio_results
    
    def interactive_demo(self):
        """Run an interactive demo session"""
        
        print("\n" + "="*60)
        print("🎤 INTERACTIVE TTS DEMO")
        print("="*60)
        
        while True:
            print(f"\nCurrent model: {self.models[self.current_model]['config']['description']}")
            print("\nOptions:")
            print("1. Speak text with current model")
            print("2. Change TTS model")
            print("3. List available speakers")
            print("4. Batch conversation demo")
            print("5. Model comparison demo")
            print("6. Quit")
            
            choice = input("\nSelect option (1-6): ").strip()
            
            if choice == "1":
                text = input("Enter text to speak: ").strip()
                if text:
                    speakers = self.list_speakers()
                    print(f"Available speakers: {', '.join(speakers[:5])}...")
                    speaker = input(f"Speaker (default: {speakers[0]}): ").strip()
                    speaker = speaker if speaker else speakers[0]
                    
                    try:
                        self.speak(text, speaker)
                    except Exception as e:
                        print(f"❌ Error: {e}")
            
            elif choice == "2":
                models = self.list_models()
                print("\nAvailable models:")
                for i, (key, desc) in enumerate(models.items(), 1):
                    print(f"{i}. {key}: {desc}")
                
                try:
                    model_choice = input("Enter model number or name: ").strip()
                    if model_choice.isdigit():
                        model_key = list(models.keys())[int(model_choice) - 1]
                    else:
                        model_key = model_choice
                    
                    self.set_model(model_key)
                except (ValueError, IndexError):
                    print("❌ Invalid model selection")
            
            elif choice == "3":
                speakers = self.list_speakers()
                print(f"\nAvailable speakers for {self.current_model}:")
                for i, speaker in enumerate(speakers, 1):
                    print(f"{i}. {speaker}")
            
            elif choice == "4":
                self._conversation_demo()
            
            elif choice == "5":
                self._model_comparison_demo()
            
            elif choice == "6":
                print("👋 Goodbye!")
                break
            
            else:
                print("❌ Invalid choice. Try again.")
    
    def _conversation_demo(self):
        """Demo with multiple speakers having a conversation"""
        
        conversation = [
            "Hello there! How are you doing today?",
            "I'm doing fantastic, thank you for asking!",
            "That's wonderful to hear. What have you been working on lately?",
            "I've been experimenting with text-to-speech technology!",
            "That sounds fascinating. How is it working out?",
            "It's amazing! The voices sound so natural now."
        ]
        
        print("\n🗣️ Starting conversation demo...")
        print("Press Enter to continue, or 'q' to skip...")
        
        if input().lower() == 'q':
            return
        
        try:
            self.batch_speak(conversation, delay_between=1.0)
        except Exception as e:
            print(f"❌ Conversation demo failed: {e}")
    
    def _model_comparison_demo(self):
        """Demo comparing different models with the same text"""
        
        test_text = "This is a demonstration of different text-to-speech models."
        
        print(f"\n🔄 Comparing models with text: '{test_text}'")
        print("Press Enter to continue, or 'q' to skip...")
        
        if input().lower() == 'q':
            return
        
        original_model = self.current_model
        
        for model_key in self.models.keys():
            try:
                print(f"\n🎙️ {self.models[model_key]['config']['description']}")
                input("Press Enter to play...")
                
                self.set_model(model_key)
                self.speak(test_text)
                
                time.sleep(0.5)  # Brief pause between models
                
            except Exception as e:
                print(f"❌ Error with {model_key}: {e}")
        
        # Restore original model
        self.set_model(original_model)
    
    def save_audio(self, audio_data: np.ndarray, filename: str, 
                   sample_rate: int = 22050) -> bool:
        """
        Save audio data to a file
        
        Args:
            audio_data (np.ndarray): Audio waveform to save
            filename (str): Output filename
            sample_rate (int): Sample rate of the audio
            
        Returns:
            bool: True if save was successful
        """
        try:
            import soundfile as sf
            
            # Ensure audio is in correct format
            if isinstance(audio_data, list):
                audio_data = np.array(audio_data)
            
            sf.write(filename, audio_data, sample_rate)
            logger.info(f"💾 Audio saved to: {filename}")
            return True
            
        except ImportError:
            logger.warning("soundfile not installed - cannot save audio files")
            logger.info("Install with: pip install soundfile")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to save audio: {e}")
            return False


def main():
    """Main demo function"""
    
    print("🚀 Initializing Complete TTS Demo...")
    
    try:
        # Initialize TTS demo
        tts_demo = WorkingTTSDemo(use_gpu=False)  # Set to True if you have CUDA
        
        # Quick test
        print("\n🔊 Quick test - generating welcome message...")
        tts_demo.speak("Welcome to the complete text-to-speech demonstration!")
        
        # Show available models
        models = tts_demo.list_models()
        print("\n📋 Available models:")
        for key, description in models.items():
            print(f"  • {key}: {description}")
        
        # Demo different features
        print("\n🎭 Feature demonstrations:")
        
        # 1. Different speakers
        print("\n1. Testing different speakers...")
        speakers = tts_demo.list_speakers()[:3]  # Use first 3 speakers
        for speaker in speakers:
            print(f"   🎙️ Speaker: {speaker}")
            tts_demo.speak(f"Hello, this is speaker {speaker}", speaker=speaker)
            time.sleep(0.5)
        
        # 2. Batch conversation
        print("\n2. Batch conversation demo...")
        conversation = [
            "Good morning! How can I help you today?",
            "I'd like to learn about text-to-speech technology.",
            "Excellent choice! TTS can convert text into natural speech.",
            "That's amazing! The voices sound very realistic."
        ]
        tts_demo.batch_speak(conversation, delay_between=1.0)
        
        # 3. Save audio example
        print("\n3. Saving audio to file...")
        audio_data = tts_demo.synthesize("This audio will be saved to a file.")
        tts_demo.save_audio(audio_data, "demo_output.wav")
        
        # Start interactive demo
        print("\n🎮 Starting interactive demo...")
        tts_demo.interactive_demo()
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        print("\n🔧 Troubleshooting tips:")
        print("1. Ensure all dependencies are installed")
        print("2. Check your audio system is working")
        print("3. Try running with different models")


if __name__ == "__main__":
    main()