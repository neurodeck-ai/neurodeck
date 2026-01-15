# Coqui TTS Complete Setup and Troubleshooting Guide

## Overview
This guide documents all the issues encountered while setting up Coqui TTS with real-time audio playback, and provides complete solutions for each problem.

## Table of Contents
1. [Initial Setup](#initial-setup)
2. [Major Issues Encountered](#major-issues-encountered)
3. [Step-by-Step Solutions](#step-by-step-solutions)
4. [Working Configuration](#working-configuration)
5. [Alternative Solutions](#alternative-solutions)
6. [Final Working Code](#final-working-code)

---

## Initial Setup

### Required Dependencies
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install core dependencies
pip install coqui-tts sounddevice torch torchaudio numpy
```

### Basic Test Script
Create a simple test to verify installation:
```python
from TTS.api import TTS
import sounddevice as sd
import numpy as np

# Test basic functionality
tts = TTS("tts_models/en/ljspeech/tacotron2-DDC")
wav = tts.tts("Hello world")
sd.play(wav, samplerate=22050)
sd.wait()
```

---

## Major Issues Encountered

### Issue 1: LogitsWarper Import Error ❌
**Error Message:**
```
cannot import name 'LogitsWarper' from 'transformers'
```

**Root Cause:** 
- LogitsWarper was removed from transformers v4.41+
- Coqui TTS (especially XTTS models) still depends on it
- Version incompatibility between transformers and coqui-tts

**Solution:**
```bash
# Downgrade transformers to compatible version
pip uninstall transformers
pip install transformers==4.40.2
```

**Why This Works:**
- transformers 4.40.2 still includes LogitsWarper
- Pre-built wheels available (no compilation issues)
- Compatible with current coqui-tts versions

### Issue 2: PyTorch Weights Loading Security Error ❌
**Error Message:**
```
Weights only load failed. Re-running `torch.load` with `weights_only` set to `False` 
will likely succeed, but it can result in arbitrary code execution.
WeightsUnpickler error: Unsupported class TTS.tts.configs.xtts_config.XttsConfig
```

**Root Cause:**
- PyTorch 2.6+ changed default `weights_only=True` for security
- XTTS models contain custom classes not in PyTorch's allowlist
- Affects specifically XTTS-v2 and other advanced models

**Attempted Solutions (Failed):**
```bash
# These didn't work:
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TORCH_SERIALIZATION_USE_SAFE_GLOBALS=0

# Code-based fix (also failed):
import torch
from TTS.tts.configs.xtts_config import XttsConfig
torch.serialization.add_safe_globals([XttsConfig])
```

**Working Solution:**
Use simpler TTS models that don't have this issue:
- `tts_models/en/ljspeech/tacotron2-DDC` (Jenny voice) ✅
- `tts_models/en/vctk/vits` (Multiple English speakers) ✅
- `tts_models/en/blizzard2013/capacitron-t2-c50` ✅

### Issue 3: Tokenizers Compilation Failure ❌
**Error Message:**
```
error: `cargo rustc --lib --message-format=json-render-diagnostics`
Building wheel for tokenizers (pyproject.toml): finished with status 'error'
```

**Root Cause:**
- tokenizers package needs Rust compilation
- macOS Xcode tools or Rust toolchain issues
- Happens when trying to install transformers==4.33.0

**Solution:**
Use transformers 4.40.2 which has pre-built tokenizers wheels:
```bash
pip install transformers==4.40.2  # Has pre-built tokenizers==0.19.1
```

### Issue 4: Audio System Not Working ❌
**Error Message:**
```
sounddevice not installed - real-time audio disabled
```

**Solution:**
```bash
pip install sounddevice
```

**Additional macOS Audio Setup:**
```bash
# If you get audio errors on macOS:
brew install portaudio  # Required for sounddevice
```

---

## Step-by-Step Solutions

### Step 1: Clean Environment Setup
```bash
# Remove any existing problematic installations
pip uninstall coqui-tts transformers tokenizers torch torchaudio -y

# Create fresh virtual environment
python3 -m venv venv_tts
source venv_tts/bin/activate

# Install in correct order
pip install torch==2.2.0 torchaudio==2.2.0
pip install transformers==4.40.2
pip install coqui-tts sounddevice numpy
```

### Step 2: Test Basic Functionality
```python
#!/usr/bin/env python3
"""Test script to verify TTS is working"""

def test_tts():
    try:
        from TTS.api import TTS
        import sounddevice as sd
        print("✅ Imports successful")
        
        # Use simple model that works
        tts = TTS("tts_models/en/ljspeech/tacotron2-DDC")
        print("✅ Model loaded")
        
        wav = tts.tts("Testing text to speech")
        print("✅ Audio generated")
        
        sd.play(wav, samplerate=22050)
        sd.wait()
        print("✅ Audio played successfully")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_tts()
```

### Step 3: Identify Working Models
After testing, these models work reliably:

```python
WORKING_MODELS = {
    "jenny": "tts_models/en/ljspeech/tacotron2-DDC",  # Single female voice
    "vctk": "tts_models/en/vctk/vits",                # Multiple English speakers  
    "blizzard": "tts_models/en/blizzard2013/capacitron-t2-c50"  # Natural English
}

# These DON'T work due to PyTorch security:
PROBLEMATIC_MODELS = [
    "tts_models/multilingual/multi-dataset/xtts_v2",  # XTTS-v2
    "tts_models/multilingual/multi-dataset/your_tts"  # YourTTS
]
```

---

## Working Configuration

### Final Dependencies That Work:
```
coqui-tts==0.24.3
sounddevice==0.5.1  
torch==2.2.0
torchaudio==2.2.0
transformers==4.40.2
tokenizers==0.19.1  # Installed automatically with transformers 4.40.2
numpy==1.26.4
```

### Requirements.txt File:
Create a `requirements.txt` file with these exact versions:
```txt
# Core TTS and audio dependencies
coqui-tts==0.24.3
sounddevice==0.5.1
numpy==1.26.4

# PyTorch (CPU version)
torch==2.2.0
torchaudio==2.2.0

# Transformers with LogitsWarper support
transformers==4.40.2

# Additional dependencies (installed automatically)
# tokenizers==0.19.1
# scipy>=1.13.0
# librosa>=0.11.0
# matplotlib>=3.8.4
```

Install with:
```bash
pip install -r requirements.txt
```

### Environment Variables (Optional):
```bash
# For CPU-only inference (default)
export CUDA_VISIBLE_DEVICES=""

# For better audio on macOS
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

---

## Alternative Solutions

### Option 1: Use Different TTS Libraries
If Coqui TTS continues to have issues:

```python
# pyttsx3 - Simple offline TTS
import pyttsx3
engine = pyttsx3.init()
engine.say("Hello world")
engine.runAndWait()

# gTTS - Google Text-to-Speech (requires internet)
from gtts import gTTS
import pygame
tts = gTTS("Hello world", lang='en')
tts.save("output.mp3")
pygame.mixer.init()
pygame.mixer.music.load("output.mp3")
pygame.mixer.music.play()
```

### Option 2: Use Older Coqui TTS Version
```bash
pip install coqui-tts==0.22.0 transformers==4.33.0
```

### Option 3: Use Community Forks
```bash
pip install git+https://github.com/idiap/coqui-ai-TTS.git
```

---

## Final Working Code

The complete working demo is in `working_tts_demo.py`. Key features:

### Simple Usage:
```python
from working_tts_demo import WorkingTTSDemo

# Initialize
tts = WorkingTTSDemo()

# Speak text
tts.speak("Hello, this is working text to speech!")

# Use different speakers
speakers = tts.list_speakers()
tts.speak("This is a different voice", speaker=speakers[1])

# Batch conversation
conversation = [
    "Hello there!",
    "Hi, how are you?", 
    "I'm doing great, thanks!"
]
tts.batch_speak(conversation)
```

### Advanced Features:
```python
# Switch models
tts.set_model("vctk")  # Multiple speakers
tts.speak("VCTK model with multiple voices", speaker="p230")

# Save audio to file
audio_data = tts.synthesize("Text to save")
tts.save_audio(audio_data, "output.wav")

# Interactive demo
tts.interactive_demo()  # Full interactive interface
```

---

## Troubleshooting Checklist

### ✅ Before You Start:
- [ ] Python 3.8+ installed
- [ ] Virtual environment created
- [ ] Audio system working (try playing music)

### ✅ Installation Issues:
- [ ] Use transformers==4.40.2 (not 4.33.0 or latest)
- [ ] Install torch before transformers
- [ ] Use `pip install` not `conda install` for TTS

### ✅ Runtime Issues:
- [ ] Avoid XTTS-v2 model (use ljspeech/vctk instead)
- [ ] Check speakers list before using specific speaker
- [ ] Use sample_rate=22050 for audio playback

### ✅ Audio Issues:
- [ ] Install sounddevice: `pip install sounddevice`
- [ ] On macOS: `brew install portaudio`
- [ ] Test system audio with music first
- [ ] Check volume levels

### ✅ Import Errors:
- [ ] Activate virtual environment
- [ ] Check transformers version: `pip list | grep transformers`
- [ ] Reinstall in clean environment if needed

---

## Quick Fix Commands

### Complete Reset:
```bash
# Nuclear option - start completely fresh
rm -rf venv/
python3 -m venv venv
source venv/bin/activate
pip install torch==2.2.0 torchaudio==2.2.0
pip install transformers==4.40.2
pip install coqui-tts sounddevice
python working_tts_demo.py
```

### Version Check:
```bash
python -c "
import torch, transformers
from TTS.api import TTS
print(f'PyTorch: {torch.__version__}')
print(f'Transformers: {transformers.__version__}')
print('✅ All imports successful')
"
```

---

## Summary

The key insights from resolving all these issues:

1. **Use transformers 4.40.2** - Perfect balance of features and compatibility
2. **Avoid XTTS-v2** - Use simpler models like LJSpeech/VCTK 
3. **Install in correct order** - torch → transformers → coqui-tts
4. **Test incrementally** - Verify each step works before proceeding
5. **Use virtual environments** - Avoid dependency conflicts

The final working solution provides high-quality TTS with multiple voices, real-time playback, and all the features needed for most applications.

**Result: Complete working TTS system with multiple voices and real-time audio! 🎉**