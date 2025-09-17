import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlp.speech_analyzer import SpeechAnalyzer, SpeechFeatures
from utils.audio_test_generator import create_sample_audio_files

class TestSpeechAnalyzer(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = SpeechAnalyzer()
        create_sample_audio_files()
    
    def test_speech_analysis(self):
        """Test complete speech analysis"""
        sample_path = 'samples/speech/sample_good.wav'
        if os.path.exists(sample_path):
            features = self.analyzer.analyze_speech_file(sample_path)
            
            self.assertIsInstance(features, SpeechFeatures)
            self.assertGreaterEqual(features.total_duration, 0)
            self.assertGreaterEqual(features.fluency_score, 0)
            self.assertLessEqual(features.fluency_score, 100)
            self.assertGreaterEqual(features.confidence_score, 0)
    
    def test_pause_analysis(self):
        """Test pause detection"""
        sample_path = 'samples/speech/sample_good.wav'
        if os.path.exists(sample_path):
            features = self.analyzer.analyze_speech_file(sample_path)
            self.assertGreaterEqual(features.pause_frequency, 0)
            self.assertGreaterEqual(features.average_pause_duration, 0)
    
    def test_invalid_audio(self):
        """Test handling of invalid audio files"""
        features = self.analyzer.analyze_speech_file('nonexistent.wav')
        self.assertEqual(features.transcript, "Analysis failed")

if __name__ == '__main__':
    unittest.main()
