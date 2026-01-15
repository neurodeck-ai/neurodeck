#!/usr/bin/env python3
"""
Simple TTS test with basic models that avoid the PyTorch weights_only issue
"""

import os
import torch
import numpy as np
import logging

# Try to fix PyTorch loading issue by setting environment
os.environ["TORCH_SERIALIZATION_USE_SAFE_GLOBALS"] = "0"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_simple_tts():
    """Test with simpler TTS models that don't have the PyTorch loading issue"""
    
    try:
        from TTS.api import TTS
        import sounddevice as sd
        
        print("✅ TTS and sounddevice imports successful")
        
        # Try a simpler model first - Jenny (LJSpeech)
        print("🔊 Testing with Jenny (LJSpeech) model...")
        
        try:
            tts = TTS("tts_models/en/ljspeech/tacotron2-DDC")
            
            # Generate audio
            text = "Hello! This is a test of the simple TTS system."
            print(f"🎙️ Generating audio for: '{text}'")
            
            wav = tts.tts(text)
            
            print(f"✅ Audio generated successfully! Shape: {np.array(wav).shape}")
            
            # Play the audio
            print("🔊 Playing audio...")
            sd.play(wav, samplerate=22050)
            sd.wait()
            print("✅ Audio playback complete!")
            
            return True
            
        except Exception as e:
            print(f"❌ LJSpeech model failed: {e}")
            
        # Try another simple model - VCTK
        print("\n🔊 Testing with VCTK model...")
        
        try:
            tts = TTS("tts_models/en/vctk/vits")
            
            # Generate audio with a specific speaker
            text = "This is a test with a different voice model."
            print(f"🎙️ Generating audio for: '{text}'")
            
            wav = tts.tts(text, speaker="p230")
            
            print(f"✅ Audio generated successfully! Shape: {np.array(wav).shape}")
            
            # Play the audio
            print("🔊 Playing audio...")
            sd.play(wav, samplerate=22050)
            sd.wait()
            print("✅ Audio playback complete!")
            
            return True
            
        except Exception as e:
            print(f"❌ VCTK model failed: {e}")
            
        return False
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_alternative_tts():
    """Test alternative TTS libraries that don't have PyTorch issues"""
    
    print("\n🔧 Testing alternative TTS libraries...")
    
    # Test pyttsx3 (offline TTS)
    try:
        import pyttsx3
        
        print("✅ pyttsx3 available - testing offline TTS...")
        
        engine = pyttsx3.init()
        
        # Set properties
        voices = engine.getProperty('voices')
        if voices:
            engine.setProperty('voice', voices[0].id)  # Use first available voice
        
        engine.setProperty('rate', 150)    # Speed
        engine.setProperty('volume', 0.8)  # Volume
        
        text = "Hello! This is pyttsx3 text to speech working offline."
        print(f"🎙️ Speaking with pyttsx3: '{text}'")
        
        engine.say(text)
        engine.runAndWait()
        
        print("✅ pyttsx3 TTS working!")
        return True
        
    except ImportError:
        print("⚠️ pyttsx3 not installed. Install with: pip install pyttsx3")
    except Exception as e:
        print(f"❌ pyttsx3 error: {e}")
    
    return False

if __name__ == "__main__":
    print("🚀 Starting Simple TTS Tests")
    print("=" * 50)
    
    # Test simple Coqui TTS models first
    success = test_simple_tts()
    
    if not success:
        print("\n🔄 Coqui TTS failed, trying alternatives...")
        test_alternative_tts()
    
    print("\n🏁 TTS testing complete!")