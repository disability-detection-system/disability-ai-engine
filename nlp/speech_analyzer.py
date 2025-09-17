import speech_recognition as sr
import numpy as np
import librosa
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import tempfile
import os
import wave
import webrtcvad
import collections
import contextlib
import sys
import time
from datetime import datetime

@dataclass
class SpeechFeatures:
    """Data class to store extracted speech features"""
    transcript: str
    reading_speed_wpm: float
    pause_frequency: float
    average_pause_duration: float
    pronunciation_score: float
    fluency_score: float
    volume_consistency: float
    pitch_variation: float
    speech_clarity: float
    confidence_score: float
    total_duration: float
    word_count: int

class SpeechAnalyzer:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.features = None
        
    def analyze_speech_file(self, audio_file_path: str, 
                          reference_text: str = None) -> SpeechFeatures:
        """
        Analyze speech from uploaded audio file
        """
        try:
            # Load audio file with better error handling
            try:
                audio_data, sample_rate = librosa.load(audio_file_path, sr=16000)
            except Exception as e:
                print(f"Error loading audio with librosa: {e}")
                # Try with different parameters
                audio_data, sample_rate = librosa.load(audio_file_path, sr=None)
                # Resample to 16kHz if needed
                if sample_rate != 16000:
                    audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
                    sample_rate = 16000
            
            # Extract basic features
            print(f"Starting speech analysis for file: {audio_file_path}")
            transcript = self._speech_to_text(audio_file_path)
            duration = len(audio_data) / sample_rate
            print(f"Audio duration: {duration:.2f} seconds")
            
            # Calculate reading speed
            word_count = len(transcript.split()) if transcript else 0
            reading_speed = (word_count / duration) * 60 if duration > 0 else 0
            print(f"Word count: {word_count}, Reading speed: {reading_speed:.1f} WPM")
            
            # Extract advanced features
            pause_metrics = self._analyze_pauses(audio_data, sample_rate)
            pronunciation_score = self._calculate_pronunciation_score(
                transcript, reference_text
            )
            fluency_score = self._calculate_fluency_score(
                audio_data, sample_rate, pause_metrics
            )
            volume_consistency = self._analyze_volume_consistency(audio_data)
            pitch_variation = self._analyze_pitch_variation(audio_data, sample_rate)
            speech_clarity = self._analyze_speech_clarity(audio_data, sample_rate)
            
            features = SpeechFeatures(
                transcript=transcript,
                reading_speed_wpm=reading_speed,
                pause_frequency=pause_metrics['frequency'],
                average_pause_duration=pause_metrics['avg_duration'],
                pronunciation_score=pronunciation_score,
                fluency_score=fluency_score,
                volume_consistency=volume_consistency,
                pitch_variation=pitch_variation,
                speech_clarity=speech_clarity,
                confidence_score=self._calculate_overall_confidence(
                    pronunciation_score, fluency_score, speech_clarity
                ),
                total_duration=duration,
                word_count=word_count
            )
            
            self.features = features
            return features
            
        except Exception as e:
            print(f"Error analyzing speech: {str(e)}")
            print(f"Audio file path: {audio_file_path}")
            print(f"File exists: {os.path.exists(audio_file_path) if audio_file_path else 'N/A'}")
            return self._get_default_features()
    
    def _speech_to_text(self, audio_file_path: str) -> str:
        """Convert speech to text using Google Speech API"""
        try:
            # Check if file exists and is not empty
            if not os.path.exists(audio_file_path):
                return "Audio file not found"
            
            file_size = os.path.getsize(audio_file_path)
            if file_size == 0:
                return "Audio file is empty"
            
            print(f"Attempting speech-to-text for file: {audio_file_path} (size: {file_size} bytes)")
            
            with sr.AudioFile(audio_file_path) as source:
                audio_data = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio_data)
                print(f"Transcription successful: {text[:50]}...")
                return text
        except sr.UnknownValueError:
            print("Speech recognition could not understand audio")
            return "Could not understand audio"
        except sr.RequestError as e:
            print(f"Speech recognition request error: {e}")
            return f"Error with speech recognition: {e}"
        except Exception as e:
            print(f"Transcription error: {str(e)}")
            return f"Transcription error: {str(e)}"
    
    def _analyze_pauses(self, audio_data: np.ndarray, 
                       sample_rate: int) -> Dict[str, float]:
        """Analyze pause frequency and duration"""
        # Use Voice Activity Detection
        frame_duration_ms = 30  # ms
        frame_length = int(sample_rate * frame_duration_ms / 1000)
        
        # Simple energy-based VAD
        energy_threshold = np.mean(audio_data**2) * 0.1
        frames = []
        
        for i in range(0, len(audio_data) - frame_length, frame_length):
            frame = audio_data[i:i + frame_length]
            energy = np.mean(frame**2)
            frames.append(energy > energy_threshold)
        
        # Find pauses (consecutive silent frames)
        pauses = []
        current_pause = 0
        
        for is_speech in frames:
            if not is_speech:
                current_pause += 1
            else:
                if current_pause > 0:
                    pause_duration = current_pause * frame_duration_ms / 1000
                    if pause_duration > 0.1:  # Only count pauses > 100ms
                        pauses.append(pause_duration)
                current_pause = 0
        
        if current_pause > 0:
            pauses.append(current_pause * frame_duration_ms / 1000)
        
        return {
            'frequency': len(pauses),
            'avg_duration': np.mean(pauses) if pauses else 0,
            'total_pause_time': sum(pauses) if pauses else 0
        }
    
    def _calculate_pronunciation_score(self, transcript: str, 
                                     reference_text: str = None) -> float:
        """Calculate pronunciation accuracy score"""
        if not transcript or transcript.startswith("Could not"):
            return 0.0
        
        if reference_text:
            # Compare with reference text using simple word matching
            transcript_words = set(transcript.lower().split())
            reference_words = set(reference_text.lower().split())
            
            if reference_words:
                common_words = transcript_words.intersection(reference_words)
                accuracy = len(common_words) / len(reference_words) * 100
                return min(100, accuracy)
        
        # If no reference, score based on transcript quality
        word_count = len(transcript.split())
        if word_count == 0:
            return 0.0
        
        # Basic heuristics for pronunciation quality
        score = 50  # Base score
        
        # Bonus for longer, coherent speech
        if word_count > 10:
            score += 20
        elif word_count > 5:
            score += 10
        
        # Penalty for recognition errors
        if "could not understand" in transcript.lower():
            score -= 30
        
        return max(0, min(100, score))
    
    def _calculate_fluency_score(self, audio_data: np.ndarray, 
                               sample_rate: int, 
                               pause_metrics: Dict[str, float]) -> float:
        """Calculate speech fluency score"""
        duration = len(audio_data) / sample_rate
        
        if duration == 0:
            return 0.0
        
        # Base fluency score
        fluency_score = 70
        
        # Adjust based on pause frequency
        pause_rate = pause_metrics['frequency'] / duration
        if pause_rate < 0.5:  # Less than 0.5 pauses per second is good
            fluency_score += 20
        elif pause_rate > 1.5:  # More than 1.5 pauses per second is poor
            fluency_score -= 30
        
        # Adjust based on average pause duration
        if pause_metrics['avg_duration'] < 0.5:  # Short pauses are better
            fluency_score += 10
        elif pause_metrics['avg_duration'] > 1.0:  # Long pauses reduce fluency
            fluency_score -= 20
        
        return max(0, min(100, fluency_score))
    
    def _analyze_volume_consistency(self, audio_data: np.ndarray) -> float:
        """Analyze volume consistency throughout speech"""
        # Calculate RMS energy in chunks
        chunk_size = len(audio_data) // 10  # Divide into 10 chunks
        if chunk_size == 0:
            return 0.0
        
        rms_values = []
        for i in range(0, len(audio_data) - chunk_size, chunk_size):
            chunk = audio_data[i:i + chunk_size]
            rms = np.sqrt(np.mean(chunk**2))
            rms_values.append(rms)
        
        if not rms_values:
            return 0.0
        
        # Calculate consistency (lower std deviation = more consistent)
        mean_rms = np.mean(rms_values)
        std_rms = np.std(rms_values)
        
        if mean_rms == 0:
            return 0.0
        
        # Convert to 0-100 scale (higher = more consistent)
        consistency = max(0, 100 - (std_rms / mean_rms * 100))
        return consistency
    
    def _analyze_pitch_variation(self, audio_data: np.ndarray, 
                               sample_rate: int) -> float:
        """Analyze pitch variation (prosody)"""
        try:
            # Extract pitch using librosa
            pitches, magnitudes = librosa.piptrack(
                y=audio_data, sr=sample_rate, threshold=0.1
            )
            
            # Get the most prominent pitch at each time frame
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if not pitch_values:
                return 50.0  # Default moderate score
            
            # Calculate pitch variation statistics
            pitch_mean = np.mean(pitch_values)
            pitch_std = np.std(pitch_values)
            
            # Good variation is typically 10-30% of mean frequency
            variation_ratio = pitch_std / pitch_mean if pitch_mean > 0 else 0
            
            if 0.1 <= variation_ratio <= 0.3:
                return 85.0  # Good prosody
            elif 0.05 <= variation_ratio <= 0.5:
                return 70.0  # Acceptable prosody
            else:
                return 40.0  # Poor prosody (too flat or too variable)
            
        except Exception:
            return 50.0  # Default score if analysis fails
    
    def _analyze_speech_clarity(self, audio_data: np.ndarray, 
                              sample_rate: int) -> float:
        """Analyze speech clarity based on spectral features"""
        try:
            # Calculate spectral features
            spectral_centroids = librosa.feature.spectral_centroid(
                y=audio_data, sr=sample_rate
            )
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=audio_data, sr=sample_rate
            )
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_data)
            
            # Calculate clarity metrics
            avg_centroid = np.mean(spectral_centroids)
            avg_rolloff = np.mean(spectral_rolloff)
            avg_zcr = np.mean(zero_crossing_rate)
            
            # Heuristic scoring based on typical speech characteristics
            clarity_score = 50  # Base score
            
            # Good speech typically has centroid around 1000-3000 Hz
            if 1000 <= avg_centroid <= 3000:
                clarity_score += 25
            
            # Reasonable rolloff indicates clear formants
            if 2000 <= avg_rolloff <= 8000:
                clarity_score += 15
            
            # Moderate ZCR indicates clear articulation
            if 0.02 <= avg_zcr <= 0.15:
                clarity_score += 10
            
            return max(0, min(100, clarity_score))
            
        except Exception:
            return 50.0  # Default score if analysis fails
    
    def _calculate_overall_confidence(self, pronunciation: float, 
                                    fluency: float, clarity: float) -> float:
        """Calculate overall confidence score"""
        # Weighted average of key metrics
        weights = {'pronunciation': 0.4, 'fluency': 0.35, 'clarity': 0.25}
        
        confidence = (pronunciation * weights['pronunciation'] + 
                     fluency * weights['fluency'] + 
                     clarity * weights['clarity'])
        
        return max(0, min(100, confidence))
    
    def _get_default_features(self) -> SpeechFeatures:
        """Return default features when analysis fails"""
        return SpeechFeatures(
            transcript="Analysis failed",
            reading_speed_wpm=0,
            pause_frequency=0,
            average_pause_duration=0,
            pronunciation_score=0,
            fluency_score=0,
            volume_consistency=0,
            pitch_variation=0,
            speech_clarity=0,
            confidence_score=0,
            total_duration=0,
            word_count=0
        )
    
    def visualize_speech_analysis(self, audio_file_path: str):
        """Visualize speech analysis results"""
        try:
            # Load audio file with better error handling
            try:
                audio_data, sample_rate = librosa.load(audio_file_path, sr=16000)
            except Exception as e:
                print(f"Error loading audio with librosa: {e}")
                # Try with different parameters
                audio_data, sample_rate = librosa.load(audio_file_path, sr=None)
                # Resample to 16kHz if needed
                if sample_rate != 16000:
                    audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
                    sample_rate = 16000
            
            # Create time axis
            time_frames = librosa.frames_to_time(
                np.arange(len(audio_data)), sr=sample_rate
            )
            
            fig, axes = plt.subplots(3, 1, figsize=(12, 10))
            
            # Plot waveform
            axes.plot(time_frames, audio_data)
            axes.set_title('Speech Waveform')
            axes.set_xlabel('Time (s)')
            axes.set_ylabel('Amplitude')
            
            # Plot spectrogram
            D = librosa.amplitude_to_db(
                np.abs(librosa.stft(audio_data)), ref=np.max
            )
            img = librosa.display.specshow(
                D, sr=sample_rate, x_axis='time', y_axis='hz', ax=axes[1]
            )
            axes[1].set_title('Spectrogram')
            fig.colorbar(img, ax=axes[1])
            
            # Plot pitch
            pitches, magnitudes = librosa.piptrack(
                y=audio_data, sr=sample_rate, threshold=0.1
            )
            pitch_values = []
            times = []
            
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
                    times.append(librosa.frames_to_time(t, sr=sample_rate))
            
            if times and pitch_values:
                axes[2].plot(times, pitch_values)
                axes[2].set_title('Pitch Contour')
                axes[2].set_xlabel('Time (s)')
                axes[2].set_ylabel('Frequency (Hz)')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Visualization failed: {str(e)}")

# Test the speech analyzer
if __name__ == "__main__":
    analyzer = SpeechAnalyzer()
    print("Speech analyzer initialized successfully!")
