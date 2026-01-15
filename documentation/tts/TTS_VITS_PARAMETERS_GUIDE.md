# VITS TTS Parameters Guide

## Overview

This guide explains the advanced parameters available in Coqui TTS VITS models for controlling speech characteristics. While VITS doesn't support direct emphasis control like SSML, these parameters allow fine-tuning of speech speed, naturalness, and timing to create distinct AI agent personalities.

## Parameter Categories

VITS parameters fall into two categories:
- **Training Parameters**: Only affect model training (not useful for pre-trained models)
- **Inference Parameters**: Control speech synthesis in real-time (useful for customization)

---

## 1. `length_scale` - Speech Speed Control

### Purpose
Directly controls the speaking rate by scaling the predicted phoneme durations.

### How It Works
- The VITS model predicts how long each phoneme should last
- `length_scale` multiplies these duration predictions
- **Smaller values = faster speech** (durations shortened)
- **Larger values = slower speech** (durations lengthened)

### Practical Ranges
- `0.5` = 2x faster (chipmunk-like, potentially unintelligible)
- `0.8` = 25% faster (brisk pace, energetic)
- `1.0` = normal speed (default, natural pace)
- `1.2` = 20% slower (deliberate pace, thoughtful)
- `2.0` = 2x slower (very slow, potentially unnatural)

### Use Cases
- **Fast speakers**: Grok (witty/energetic) = `0.9`
- **Normal speakers**: ChatGPT (helpful) = `1.0`
- **Slow speakers**: Claudius (scholarly/measured) = `1.1`

### Code Example
```python
audio = self.tts_model.tts(
    text="Hello world", 
    speaker=voice_id,
    length_scale=0.9  # 10% faster speech
)
```

---

## 2. `noise_scale` - Training Variability ⚠️ (Training Only)

### Purpose
Controls randomness during model training, not synthesis.

### How It Works
- Adds noise to the latent space during training
- Higher values make the model learn more diverse speech patterns
- **Only affects training** - not relevant for inference
- Pre-trained models already have this "baked in"

### Practical Impact
❌ **Cannot be changed** for pre-trained VCTK models. This parameter is fixed during training.

### Note
This parameter is included for completeness but has no effect on speech synthesis with existing models.

---

## 3. `inference_noise_scale` - Speech Naturalness/Variability

### Purpose
Controls how much randomness is added during speech generation to create natural variation.

### How It Works
- Adds controlled noise to the latent representation during synthesis
- **Higher values = more expressive/variable speech** (but potentially less stable)
- **Lower values = more consistent/robotic speech** (but potentially more predictable)
- Default: `0.667`

### Practical Ranges
- `0.0` = deterministic/robotic (same text always produces identical audio)
- `0.3` = very consistent (minimal variation, corporate-like)
- `0.667` = default balance (natural but stable)
- `1.0` = more expressive/variable (natural human-like variation)
- `1.5` = very expressive (may sound unstable or inconsistent)

### Use Cases
- **Consistent speakers**: ChatGPT (reliable assistant) = `0.4`
- **Balanced speakers**: Claudius (professional but human) = `0.667`
- **Expressive speakers**: Grok (dynamic personality) = `1.0`
- **Creative speakers**: Kimi (artistic/varied) = `1.2`

### Code Example
```python
audio = self.tts_model.tts(
    text="This is amazing!", 
    speaker=voice_id,
    inference_noise_scale=1.0  # More expressive delivery
)
```

---

## 4. `noise_scale_dp` - Duration Predictor Training Noise ⚠️ (Training Only)

### Purpose
Controls randomness in duration prediction during training.

### How It Works
- Affects how the model learns to predict phoneme durations during training
- **Training-only parameter** - irrelevant for inference
- Pre-trained models have this already determined

### Practical Impact
❌ **Cannot be modified** for existing models. Fixed during training phase.

### Note
Included for technical completeness but not applicable to inference with pre-trained models.

---

## 5. `inference_noise_scale_dp` - Duration Prediction Variability

### Purpose
Controls randomness in phoneme duration prediction during synthesis, affecting speech rhythm and timing.

### How It Works
- The "Stochastic Duration Predictor" adds controlled randomness to timing
- **Higher values = more varied rhythm/timing** (natural pauses, varied pacing)
- **Lower values = more mechanical timing** (consistent, predictable rhythm)
- Default: `0.8`

### Practical Ranges
- `0.0` = mechanical timing (perfectly regular, robotic rhythm)
- `0.5` = slightly varied timing (subtle natural variation)
- `0.8` = default natural variation (balanced human-like timing)
- `1.2` = more dramatic timing variations (expressive pauses and pacing)
- `1.5` = very expressive timing (potentially irregular, artistic)

### Use Cases
- **Regular speakers**: Claudius (academic precision) = `0.6`
- **Natural speakers**: ChatGPT (conversational flow) = `0.8`
- **Dynamic speakers**: Grok (varied pacing for emphasis) = `1.0`
- **Expressive speakers**: Kimi (artistic timing variations) = `1.2`

### Code Example
```python
audio = self.tts_model.tts(
    text="Well... I think that's interesting.", 
    speaker=voice_id,
    inference_noise_scale_dp=1.0  # More varied pause timing
)
```

---

## Practical Implementation for NeuroDeck

### Agent Configuration Example

```ini
[agent:claudius]
tts_voice = p272
tts_length_scale = 1.1          # Slower, more deliberate
tts_inference_noise = 0.5       # More consistent delivery
tts_duration_noise = 0.6        # Regular, academic timing

[agent:grok]
tts_voice = p251  
tts_length_scale = 0.9          # Faster, more energetic
tts_inference_noise = 1.0       # More expressive
tts_duration_noise = 1.0        # Varied timing for wit

[agent:chatgpt]
tts_voice = p254
tts_length_scale = 1.0          # Normal helpful pace
tts_inference_noise = 0.4       # Consistent assistant
tts_duration_noise = 0.8        # Default natural timing

[agent:kimi]
tts_voice = p270
tts_length_scale = 1.05         # Slightly slower, thoughtful
tts_inference_noise = 1.2       # Creative expressiveness  
tts_duration_noise = 1.1        # Artistic timing variations
```

### Code Implementation

```python
def _generate_audio(self, text: str, voice_id: str, agent_config: dict) -> Optional[np.ndarray]:
    """Generate audio with agent-specific parameters."""
    
    # Get agent-specific parameters or use defaults
    length_scale = getattr(agent_config, 'tts_length_scale', 1.0)
    inference_noise = getattr(agent_config, 'tts_inference_noise', 0.667)
    duration_noise = getattr(agent_config, 'tts_duration_noise', 0.8)
    
    # Generate audio with custom parameters
    audio = self.tts_model.tts(
        text,
        speaker=voice_id,
        length_scale=length_scale,
        inference_noise_scale=inference_noise,
        inference_noise_scale_dp=duration_noise
    )
    
    return audio
```

---

## Important Notes

### Parameter Interactions
- Parameters work together to create overall speaking style
- Start with small adjustments (±0.1-0.2 from defaults)
- Test combinations to avoid unnatural speech

### Performance Considerations
- More variability (higher noise scales) may increase generation time slightly
- Extreme values can produce unstable or unnatural speech
- Always test parameter changes with actual speech samples

### Model Limitations
- These parameters work with VCTK/VITS models specifically
- Other TTS models (XTTS, Tacotron) may have different parameter sets
- Direct emphasis/pitch control still requires text-based methods (punctuation, capitalization)

### Best Practices
1. **Start Conservative**: Begin with small parameter changes
2. **Test Thoroughly**: Generate samples to verify naturalness
3. **Consider Context**: Match parameters to agent personality
4. **Document Settings**: Keep track of successful parameter combinations
5. **User Testing**: Get feedback on which voices sound most natural

---

## Troubleshooting

### Common Issues
- **Robotic Speech**: Increase `inference_noise_scale` (try 0.8-1.0)
- **Inconsistent Speech**: Decrease `inference_noise_scale` (try 0.4-0.6)
- **Too Fast/Slow**: Adjust `length_scale` (0.8-1.2 range usually works)
- **Mechanical Timing**: Increase `inference_noise_scale_dp` (try 0.9-1.1)
- **Erratic Timing**: Decrease `inference_noise_scale_dp` (try 0.6-0.8)

### Parameter Reset
If speech becomes unnatural, return to defaults:
```python
length_scale=1.0
inference_noise_scale=0.667  
inference_noise_scale_dp=0.8
```

---

## Conclusion

While VITS doesn't offer direct emphasis control, these parameters provide powerful tools for creating distinct AI agent personalities through speech characteristics. The combination of unique voices (speaker IDs) and customized speech parameters can create truly differentiated agent experiences.

For best results, combine these parameters with effective text preprocessing (punctuation, formatting) to achieve natural, expressive AI agent speech that matches each character's personality.