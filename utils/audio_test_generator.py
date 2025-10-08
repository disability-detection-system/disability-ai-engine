import numpy as np
import soundfile as sf
import os
import random

def create_sample_audio_files():
    """Create sample audio files for testing"""
    os.makedirs('samples/speech', exist_ok=True)
    
    # Sample 1: Good speech simulation
    duration = 5.0  # 5 seconds
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create a speech-like signal with multiple frequencies
    frequencies = [200, 400, 800, 1600]  # Formant-like frequencies
    signal = np.zeros_like(t)
    
    for freq in frequencies:
        signal += 0.25 * np.sin(2 * np.pi * freq * t)
    
    # Add some pauses (silent segments)
    pause_times = [(1.5, 1.7), (3.2, 3.4)]  # (start, end) in seconds
    for start, end in pause_times:
        start_idx = int(start * sample_rate)
        end_idx = int(end * sample_rate)
        signal[start_idx:end_idx] = 0
    
    # Add noise for realism
    noise = np.random.normal(0, 0.01, len(signal))
    signal += noise
    
    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.8
    
    sf.write('samples/speech/sample_good.wav', signal, sample_rate)
    
    # Sample 2: Poor speech simulation (many pauses, inconsistent volume)
    signal2 = np.zeros_like(t)
    
    # Create choppy speech with many pauses
    speech_segments = [(0, 0.5), (1.0, 1.3), (2.0, 2.2), (3.5, 4.0)]
    
    for start, end in speech_segments:
        start_idx = int(start * sample_rate)
        end_idx = int(end * sample_rate)
        segment_t = t[start_idx:end_idx]
        
        # Variable volume
        volume = random.uniform(0.2, 0.8)
        segment_signal = volume * 0.5 * np.sin(2 * np.pi * 300 * segment_t)
        signal2[start_idx:end_idx] = segment_signal
    
    # Add more noise
    noise2 = np.random.normal(0, 0.05, len(signal2))
    signal2 += noise2
    
    sf.write('samples/speech/sample_poor.wav', signal2, sample_rate)
    
    print("Sample audio files created in samples/speech/")

if __name__ == "__main__":
    create_sample_audio_files()
