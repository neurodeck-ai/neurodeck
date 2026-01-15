# Coqui TTS Implementation Guide

A complete step-by-step guide to implement real-time text-to-speech with multiple voices using Coqui TTS.

## Table of Contents
1. [Quick Start](#quick-start)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Basic Implementation](#basic-implementation)
5. [Advanced Features](#advanced-features)
6. [Production Setup](#production-setup)
7. [Common Use Cases](#common-use-cases)
8. [Performance Optimization](#performance-optimization)

---

## Quick Start

### 5-Minute Setup
```bash
# 1. Create and activate virtual environment
python3 -m venv tts_env
source tts_env/bin/activate  # On Windows: tts_env\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Test basic functionality
python -c "
from TTS.api import TTS
import sounddevice as sd
tts = TTS('tts_models/en/ljspeech/tacotron2-DDC')
audio = tts.tts('Hello world!')
sd.play(audio, samplerate=22050)
sd.wait()
print('✅ TTS working!')
"
```

---

## Prerequisites

### System Requirements
- **Python**: 3.8+ (3.9-3.11 recommended)
- **OS**: Windows, macOS, or Linux
- **RAM**: 4GB minimum, 8GB+ recommended
- **Storage**: 2GB for models
- **Audio**: Working speakers/headphones

### Development Environment
- Virtual environment (required)
- Code editor (VS Code, PyCharm, etc.)
- Terminal/Command Prompt

---

## Installation

### Step 1: Environment Setup
```bash
# Create isolated environment
python3 -m venv tts_project
cd tts_project
source bin/activate  # Linux/macOS
# OR
Scripts\activate  # Windows

# Upgrade pip
pip install --upgrade pip
```

### Step 2: Install Dependencies
Create `requirements.txt`:
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
```

Install:
```bash
pip install -r requirements.txt
```

### Step 3: Verify Installation
```python
# test_installation.py
def test_imports():
    try:
        from TTS.api import TTS
        import sounddevice as sd
        import torch
        import transformers
        print("✅ All imports successful")
        print(f"PyTorch: {torch.__version__}")
        print(f"Transformers: {transformers.__version__}")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

if __name__ == "__main__":
    test_imports()
```

---

## Basic Implementation

### Simple Text-to-Speech
```python
# basic_tts.py
from TTS.api import TTS
import sounddevice as sd

class SimpleTTS:
    def __init__(self, model_name="tts_models/en/ljspeech/tacotron2-DDC"):
        """Initialize TTS with a specific model"""
        self.tts = TTS(model_name)
        self.sample_rate = 22050
    
    def speak(self, text):
        """Convert text to speech and play it"""
        try:
            # Generate audio
            audio = self.tts.tts(text)
            
            # Play audio
            sd.play(audio, samplerate=self.sample_rate)
            sd.wait()  # Wait for playback to finish
            
            return True
        except Exception as e:
            print(f"Error: {e}")
            return False
    
    def save_audio(self, text, filename):
        """Save audio to file"""
        self.tts.tts_to_file(text=text, file_path=filename)

# Usage
if __name__ == "__main__":
    tts = SimpleTTS()
    tts.speak("Hello! This is your text to speech system.")
    tts.save_audio("This will be saved to a file.", "output.wav")
```

### Multiple Voice Models
```python
# multi_voice_tts.py
from TTS.api import TTS
import sounddevice as sd

class MultiVoiceTTS:
    def __init__(self):
        """Initialize with multiple TTS models"""
        self.models = {}
        self.current_model = None
        self.sample_rate = 22050
        
        # Define available models
        model_configs = {
            "jenny": {
                "model": "tts_models/en/ljspeech/tacotron2-DDC",
                "description": "Jenny - High quality female voice",
                "speakers": ["jenny"]
            },
            "vctk": {
                "model": "tts_models/en/vctk/vits",
                "description": "VCTK - Multiple English speakers",
                "speakers": ["p225", "p226", "p227", "p228", "p229", "p230"]
            }
        }
        
        # Load models
        for key, config in model_configs.items():
            try:
                print(f"Loading {config['description']}...")
                self.models[key] = {
                    "tts": TTS(config["model"]),
                    "config": config
                }
                if self.current_model is None:
                    self.current_model = key
                print(f"✅ {key} loaded successfully")
            except Exception as e:
                print(f"❌ Failed to load {key}: {e}")
    
    def list_voices(self):
        """List all available voices"""
        voices = {}
        for model_key, model_info in self.models.items():
            voices[model_key] = model_info["config"]["speakers"]
        return voices
    
    def speak(self, text, model_key=None, speaker=None):
        """Speak text with specified model and speaker"""
        model_key = model_key or self.current_model
        
        if model_key not in self.models:
            raise ValueError(f"Model {model_key} not available")
        
        model_info = self.models[model_key]
        tts_engine = model_info["tts"]
        available_speakers = model_info["config"]["speakers"]
        
        # Use specified speaker or default
        if speaker is None:
            speaker = available_speakers[0]
        elif speaker not in available_speakers:
            print(f"Speaker {speaker} not available, using {available_speakers[0]}")
            speaker = available_speakers[0]
        
        try:
            # Generate audio
            if len(available_speakers) == 1:
                audio = tts_engine.tts(text)
            else:
                audio = tts_engine.tts(text, speaker=speaker)
            
            # Play audio
            sd.play(audio, samplerate=self.sample_rate)
            sd.wait()
            
            print(f"✅ Played with {model_key} - {speaker}")
            return True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def batch_speak(self, texts, speakers=None, model_key=None, delay=0.5):
        """Speak multiple texts in sequence"""
        import time
        
        model_key = model_key or self.current_model
        available_speakers = self.models[model_key]["config"]["speakers"]
        
        if speakers is None:
            # Cycle through available speakers
            speakers = [available_speakers[i % len(available_speakers)] 
                       for i in range(len(texts))]
        
        for i, (text, speaker) in enumerate(zip(texts, speakers)):
            print(f"Speaking {i+1}/{len(texts)}: {speaker}")
            self.speak(text, model_key, speaker)
            if i < len(texts) - 1:
                time.sleep(delay)

# Usage
if __name__ == "__main__":
    tts = MultiVoiceTTS()
    
    # List available voices
    voices = tts.list_voices()
    print("Available voices:", voices)
    
    # Test different speakers
    tts.speak("Hello from Jenny!", "jenny")
    tts.speak("Hello from VCTK speaker p225!", "vctk", "p225")
    tts.speak("Hello from VCTK speaker p230!", "vctk", "p230")
    
    # Batch conversation
    conversation = [
        "Hello there!",
        "Hi, how are you?",
        "I'm doing great, thanks!"
    ]
    tts.batch_speak(conversation, model_key="vctk")
```

---

## Advanced Features

### Interactive Voice Selection
```python
# interactive_tts.py
class InteractiveTTS:
    def __init__(self):
        self.tts_system = MultiVoiceTTS()
    
    def run_demo(self):
        """Run interactive demo"""
        print("\n🎤 Interactive TTS Demo")
        print("=" * 40)
        
        while True:
            print("\nOptions:")
            print("1. Speak text")
            print("2. List voices")
            print("3. Voice comparison")
            print("4. Batch conversation")
            print("5. Save audio")
            print("6. Quit")
            
            choice = input("Select (1-6): ").strip()
            
            if choice == "1":
                self._single_speech()
            elif choice == "2":
                self._list_voices()
            elif choice == "3":
                self._voice_comparison()
            elif choice == "4":
                self._batch_conversation()
            elif choice == "5":
                self._save_audio()
            elif choice == "6":
                print("Goodbye!")
                break
            else:
                print("Invalid choice")
    
    def _single_speech(self):
        """Handle single speech input"""
        text = input("Enter text: ").strip()
        if not text:
            return
        
        voices = self.tts_system.list_voices()
        print("Available models:", list(voices.keys()))
        
        model = input("Model (or Enter for default): ").strip()
        if model and model in voices:
            speakers = voices[model]
            print(f"Speakers: {speakers[:5]}...")
            speaker = input("Speaker (or Enter for default): ").strip()
            self.tts_system.speak(text, model, speaker if speaker else None)
        else:
            self.tts_system.speak(text)
    
    def _list_voices(self):
        """List all available voices"""
        voices = self.tts_system.list_voices()
        for model, speakers in voices.items():
            print(f"\n{model.upper()}:")
            for speaker in speakers[:10]:  # Show first 10
                print(f"  - {speaker}")
            if len(speakers) > 10:
                print(f"  ... and {len(speakers)-10} more")
    
    def _voice_comparison(self):
        """Compare different voices with same text"""
        text = input("Enter text for comparison: ").strip()
        if not text:
            return
        
        voices = self.tts_system.list_voices()
        for model in voices.keys():
            print(f"\n🎙️ {model.upper()}")
            input("Press Enter to play...")
            self.tts_system.speak(text, model)
    
    def _batch_conversation(self):
        """Create batch conversation"""
        print("Enter conversation lines (empty line to finish):")
        texts = []
        while True:
            line = input(f"Line {len(texts)+1}: ").strip()
            if not line:
                break
            texts.append(line)
        
        if texts:
            model = input("Model (vctk for multiple speakers): ").strip() or "vctk"
            self.tts_system.batch_speak(texts, model_key=model)
    
    def _save_audio(self):
        """Save audio to file"""
        text = input("Text to save: ").strip()
        filename = input("Filename (with .wav): ").strip()
        
        if text and filename:
            # Use basic model for file saving
            from TTS.api import TTS
            tts = TTS("tts_models/en/ljspeech/tacotron2-DDC")
            tts.tts_to_file(text=text, file_path=filename)
            print(f"✅ Saved to {filename}")

if __name__ == "__main__":
    demo = InteractiveTTS()
    demo.run_demo()
```

### Real-time Audio Processing
```python
# realtime_tts.py
import threading
import queue
import time

class RealtimeTTS:
    def __init__(self):
        self.tts_system = MultiVoiceTTS()
        self.audio_queue = queue.Queue()
        self.is_playing = False
        self.playback_thread = None
        
    def start_playback_thread(self):
        """Start background audio playback thread"""
        if self.playback_thread is None or not self.playback_thread.is_alive():
            self.playback_thread = threading.Thread(target=self._audio_worker)
            self.playback_thread.daemon = True
            self.playback_thread.start()
    
    def _audio_worker(self):
        """Background worker for audio playback"""
        while True:
            try:
                audio_data = self.audio_queue.get(timeout=1)
                if audio_data is None:  # Shutdown signal
                    break
                
                self.is_playing = True
                sd.play(audio_data, samplerate=22050)
                sd.wait()
                self.is_playing = False
                
                self.audio_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Audio playback error: {e}")
                self.is_playing = False
    
    def queue_speech(self, text, model_key=None, speaker=None):
        """Queue text for speech without blocking"""
        def generate_audio():
            try:
                model_key_to_use = model_key or self.tts_system.current_model
                model_info = self.tts_system.models[model_key_to_use]
                tts_engine = model_info["tts"]
                available_speakers = model_info["config"]["speakers"]
                
                speaker_to_use = speaker or available_speakers[0]
                
                if len(available_speakers) == 1:
                    audio = tts_engine.tts(text)
                else:
                    audio = tts_engine.tts(text, speaker=speaker_to_use)
                
                self.audio_queue.put(audio)
                print(f"✅ Queued: '{text[:30]}...' with {speaker_to_use}")
                
            except Exception as e:
                print(f"❌ Generation error: {e}")
        
        # Generate audio in background thread
        gen_thread = threading.Thread(target=generate_audio)
        gen_thread.daemon = True
        gen_thread.start()
        
        # Start playback thread if not running
        self.start_playback_thread()
    
    def is_busy(self):
        """Check if system is currently playing audio"""
        return self.is_playing or not self.audio_queue.empty()
    
    def wait_for_completion(self):
        """Wait for all queued audio to finish playing"""
        self.audio_queue.join()
        while self.is_playing:
            time.sleep(0.1)
    
    def clear_queue(self):
        """Clear all queued audio"""
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.task_done()
            except queue.Empty:
                break
    
    def shutdown(self):
        """Shutdown the realtime system"""
        self.clear_queue()
        self.audio_queue.put(None)  # Shutdown signal
        if self.playback_thread:
            self.playback_thread.join(timeout=2)

# Usage
if __name__ == "__main__":
    rt_tts = RealtimeTTS()
    
    # Queue multiple speeches
    rt_tts.queue_speech("First message", "vctk", "p225")
    rt_tts.queue_speech("Second message", "vctk", "p226")
    rt_tts.queue_speech("Third message", "vctk", "p227")
    
    print("Audio queued, will play in sequence...")
    
    # Do other work while audio plays
    while rt_tts.is_busy():
        print("Audio playing... doing other work")
        time.sleep(1)
    
    print("All audio finished!")
    rt_tts.shutdown()
```

---

## Production Setup

### Configuration Management
```python
# config.py
import os
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class TTSConfig:
    """TTS system configuration"""
    
    # Model settings
    default_model: str = "tts_models/en/ljspeech/tacotron2-DDC"
    cache_dir: str = os.path.expanduser("~/.cache/tts")
    use_gpu: bool = False
    
    # Audio settings
    sample_rate: int = 22050
    audio_format: str = "wav"
    volume: float = 1.0
    
    # Performance settings
    max_text_length: int = 1000
    batch_size: int = 1
    num_threads: int = 2
    
    # Available models
    models: Dict[str, Dict] = None
    
    def __post_init__(self):
        if self.models is None:
            self.models = {
                "jenny": {
                    "model": "tts_models/en/ljspeech/tacotron2-DDC",
                    "description": "Jenny - High quality female voice",
                    "speakers": ["jenny"],
                    "languages": ["en"]
                },
                "vctk": {
                    "model": "tts_models/en/vctk/vits",
                    "description": "VCTK - Multiple English speakers",
                    "speakers": ["p225", "p226", "p227", "p228", "p229", "p230"],
                    "languages": ["en"]
                }
            }
    
    @classmethod
    def from_file(cls, config_path: str):
        """Load configuration from file"""
        import json
        with open(config_path, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    def save_to_file(self, config_path: str):
        """Save configuration to file"""
        import json
        with open(config_path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)

# production_tts.py
import logging
from pathlib import Path

class ProductionTTS:
    def __init__(self, config: TTSConfig):
        self.config = config
        self.models = {}
        self.setup_logging()
        self.setup_cache_directory()
        self.load_models()
    
    def setup_logging(self):
        """Setup logging for production"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('tts_production.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('ProductionTTS')
    
    def setup_cache_directory(self):
        """Create cache directory if it doesn't exist"""
        cache_path = Path(self.config.cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Cache directory: {cache_path}")
    
    def load_models(self):
        """Load TTS models with error handling"""
        for model_key, model_config in self.config.models.items():
            try:
                self.logger.info(f"Loading {model_config['description']}...")
                
                tts = TTS(model_config["model"])
                if self.config.use_gpu:
                    tts = tts.to("cuda")
                
                self.models[model_key] = {
                    "tts": tts,
                    "config": model_config
                }
                
                self.logger.info(f"✅ {model_key} loaded successfully")
                
            except Exception as e:
                self.logger.error(f"❌ Failed to load {model_key}: {e}")
    
    def validate_input(self, text: str) -> bool:
        """Validate input text"""
        if not text or not isinstance(text, str):
            return False
        
        if len(text) > self.config.max_text_length:
            self.logger.warning(f"Text too long: {len(text)} > {self.config.max_text_length}")
            return False
        
        return True
    
    def synthesize(self, text: str, model_key: str = None, speaker: str = None):
        """Production-ready synthesis with error handling"""
        if not self.validate_input(text):
            raise ValueError("Invalid input text")
        
        model_key = model_key or list(self.models.keys())[0]
        
        if model_key not in self.models:
            raise ValueError(f"Model {model_key} not available")
        
        try:
            model_info = self.models[model_key]
            tts_engine = model_info["tts"]
            available_speakers = model_info["config"]["speakers"]
            
            speaker = speaker or available_speakers[0]
            
            self.logger.info(f"Synthesizing: '{text[:50]}...' with {model_key}:{speaker}")
            
            start_time = time.time()
            
            if len(available_speakers) == 1:
                audio = tts_engine.tts(text)
            else:
                audio = tts_engine.tts(text, speaker=speaker)
            
            duration = time.time() - start_time
            self.logger.info(f"Synthesis completed in {duration:.2f}s")
            
            return audio
            
        except Exception as e:
            self.logger.error(f"Synthesis failed: {e}")
            raise
    
    def health_check(self) -> Dict[str, bool]:
        """Check system health"""
        health = {
            "models_loaded": len(self.models) > 0,
            "audio_system": True,  # Test audio system
            "cache_writable": True  # Test cache directory
        }
        
        try:
            # Test audio system
            import sounddevice as sd
            sd.check_output_settings()
        except:
            health["audio_system"] = False
        
        try:
            # Test cache directory
            test_file = Path(self.config.cache_dir) / "test"
            test_file.touch()
            test_file.unlink()
        except:
            health["cache_writable"] = False
        
        return health

# Usage
if __name__ == "__main__":
    config = TTSConfig(use_gpu=False, max_text_length=500)
    tts = ProductionTTS(config)
    
    # Health check
    health = tts.health_check()
    print("System health:", health)
    
    # Test synthesis
    if all(health.values()):
        audio = tts.synthesize("Production TTS system is working correctly!")
        print("✅ Production system ready")
    else:
        print("❌ System not healthy")
```

---

## Common Use Cases

### 1. Chatbot Integration
```python
# chatbot_tts.py
class ChatbotTTS:
    def __init__(self):
        self.tts = ProductionTTS(TTSConfig())
        self.personality_voices = {
            "friendly": ("vctk", "p225"),
            "professional": ("jenny", "jenny"),
            "casual": ("vctk", "p226"),
            "authoritative": ("vctk", "p230")
        }
    
    def respond_with_voice(self, message: str, personality: str = "friendly"):
        """Respond to chat message with appropriate voice"""
        if personality in self.personality_voices:
            model, speaker = self.personality_voices[personality]
            audio = self.tts.synthesize(message, model, speaker)
            sd.play(audio, samplerate=22050)
            sd.wait()
        else:
            print(f"Unknown personality: {personality}")
```

### 2. Accessibility Tool
```python
# accessibility_tts.py
class AccessibilityTTS:
    def __init__(self):
        self.tts = ProductionTTS(TTSConfig())
        self.reading_speed = "normal"  # slow, normal, fast
        
    def read_text(self, text: str, speed: str = None):
        """Read text with accessibility features"""
        speed = speed or self.reading_speed
        
        # Adjust text for reading speed
        if speed == "slow":
            # Add pauses for slower reading
            text = text.replace(".", ". ... ").replace(",", ", .. ")
        elif speed == "fast":
            # Remove extra punctuation for faster reading
            text = text.replace("...", ".").replace(",", "")
        
        audio = self.tts.synthesize(text)
        sd.play(audio, samplerate=22050)
        sd.wait()
    
    def read_file(self, file_path: str):
        """Read entire file aloud"""
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Split into chunks for better processing
        chunks = self.split_text(content, max_length=500)
        
        for i, chunk in enumerate(chunks):
            print(f"Reading chunk {i+1}/{len(chunks)}")
            self.read_text(chunk)
    
    def split_text(self, text: str, max_length: int = 500):
        """Split text into readable chunks"""
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) < max_length:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
```

### 3. Educational Content
```python
# educational_tts.py
class EducationalTTS:
    def __init__(self):
        self.tts = ProductionTTS(TTSConfig())
        self.teacher_voice = ("vctk", "p225")
        self.student_voices = [("vctk", "p226"), ("vctk", "p227"), ("vctk", "p228")]
    
    def create_dialogue(self, dialogue_script: List[Dict]):
        """Create educational dialogue with multiple voices"""
        for line in dialogue_script:
            role = line.get("role", "teacher")
            text = line.get("text", "")
            
            if role == "teacher":
                model, speaker = self.teacher_voice
            else:
                # Assign student voices cyclically
                student_index = hash(role) % len(self.student_voices)
                model, speaker = self.student_voices[student_index]
            
            print(f"{role.title()}: {text}")
            audio = self.tts.synthesize(text, model, speaker)
            sd.play(audio, samplerate=22050)
            sd.wait()
            
            # Pause between speakers
            time.sleep(0.5)

# Usage
dialogue = [
    {"role": "teacher", "text": "Today we'll learn about text-to-speech technology."},
    {"role": "student1", "text": "How does it work?"},
    {"role": "teacher", "text": "It converts written text into natural-sounding speech."},
    {"role": "student2", "text": "That's amazing! Can it speak in different voices?"},
    {"role": "teacher", "text": "Yes, as you can hear right now!"}
]

edu_tts = EducationalTTS()
edu_tts.create_dialogue(dialogue)
```

---

## Performance Optimization

### Memory Management
```python
# optimized_tts.py
import gc
import psutil
import time

class OptimizedTTS:
    def __init__(self, config: TTSConfig):
        self.config = config
        self.models = {}
        self.model_cache = {}
        self.memory_threshold = 80  # Percentage
        
    def load_model_on_demand(self, model_key: str):
        """Load model only when needed"""
        if model_key not in self.models:
            if self.check_memory_usage() > self.memory_threshold:
                self.cleanup_unused_models()
            
            print(f"Loading {model_key} on demand...")
            model_config = self.config.models[model_key]
            self.models[model_key] = TTS(model_config["model"])
            
        return self.models[model_key]
    
    def check_memory_usage(self) -> float:
        """Check current memory usage percentage"""
        return psutil.virtual_memory().percent
    
    def cleanup_unused_models(self):
        """Remove unused models from memory"""
        if len(self.models) > 1:  # Keep at least one model
            # Remove oldest model
            oldest_key = next(iter(self.models))
            del self.models[oldest_key]
            gc.collect()
            print(f"Cleaned up {oldest_key} model from memory")
    
    def batch_synthesize(self, texts: List[str], model_key: str = None):
        """Synthesize multiple texts efficiently"""
        model_key = model_key or list(self.config.models.keys())[0]
        tts_engine = self.load_model_on_demand(model_key)
        
        audio_results = []
        start_time = time.time()
        
        for i, text in enumerate(texts):
            print(f"Processing {i+1}/{len(texts)}: {text[:30]}...")
            audio = tts_engine.tts(text)
            audio_results.append(audio)
            
            # Memory check every 10 items
            if i % 10 == 0 and self.check_memory_usage() > self.memory_threshold:
                gc.collect()
        
        duration = time.time() - start_time
        print(f"Batch synthesis completed in {duration:.2f}s")
        
        return audio_results
```

### Caching System
```python
# caching_tts.py
import hashlib
import pickle
import os
from pathlib import Path

class CachingTTS:
    def __init__(self, config: TTSConfig):
        self.config = config
        self.tts = ProductionTTS(config)
        self.cache_dir = Path(config.cache_dir) / "audio_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def get_cache_key(self, text: str, model: str, speaker: str) -> str:
        """Generate cache key for text/model/speaker combination"""
        content = f"{text}:{model}:{speaker}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_cached_audio(self, cache_key: str):
        """Retrieve audio from cache"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                # Remove corrupted cache file
                cache_file.unlink()
        return None
    
    def cache_audio(self, cache_key: str, audio_data):
        """Store audio in cache"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(audio_data, f)
        except Exception as e:
            print(f"Failed to cache audio: {e}")
    
    def synthesize_with_cache(self, text: str, model: str = None, speaker: str = None):
        """Synthesize with caching"""
        cache_key = self.get_cache_key(text, model or "default", speaker or "default")
        
        # Try cache first
        cached_audio = self.get_cached_audio(cache_key)
        if cached_audio is not None:
            print(f"✅ Using cached audio for: {text[:30]}...")
            return cached_audio
        
        # Generate new audio
        print(f"🔄 Generating new audio for: {text[:30]}...")
        audio = self.tts.synthesize(text, model, speaker)
        
        # Cache the result
        self.cache_audio(cache_key, audio)
        
        return audio
    
    def clear_cache(self):
        """Clear all cached audio"""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
        print("✅ Cache cleared")
    
    def get_cache_stats(self):
        """Get cache statistics"""
        cache_files = list(self.cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            "files": len(cache_files),
            "total_size_mb": total_size / (1024 * 1024),
            "cache_dir": str(self.cache_dir)
        }
```

---

## Getting Started Checklist

### ✅ Setup Checklist
- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] Dependencies installed from requirements.txt
- [ ] Audio system tested (can play music)
- [ ] Basic TTS test passed

### ✅ Implementation Checklist  
- [ ] Basic TTS class implemented
- [ ] Multiple voice models configured
- [ ] Error handling added
- [ ] Logging configured
- [ ] Configuration management setup

### ✅ Production Checklist
- [ ] Health checks implemented
- [ ] Memory management optimized
- [ ] Caching system configured
- [ ] Performance monitoring added
- [ ] Documentation completed

### ✅ Testing Checklist
- [ ] Unit tests for core functionality
- [ ] Integration tests for voice models
- [ ] Performance tests for batch processing
- [ ] Memory usage tests
- [ ] Audio quality validation

---

## Next Steps

1. **Start with Basic Implementation** - Get simple TTS working first
2. **Add Multiple Voices** - Implement VCTK multi-speaker support
3. **Build Interactive Features** - Create voice selection and demos
4. **Optimize for Production** - Add caching, logging, error handling
5. **Create Your Application** - Build your specific use case

## Support and Resources

- **GitHub Repository**: Save all code examples
- **Documentation**: Keep this guide handy for reference
- **Community**: Join TTS communities for help and updates
- **Testing**: Always test with your specific use case

**Happy coding! 🎉**