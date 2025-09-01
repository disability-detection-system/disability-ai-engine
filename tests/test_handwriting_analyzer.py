import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cv.handwriting_analyzer import HandwritingAnalyzer, HandwritingFeatures
from utils.test_data_generator import create_sample_handwriting_images

class TestHandwritingAnalyzer(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = HandwritingAnalyzer()
        create_sample_handwriting_images()
    
    def test_preprocessing(self):
        """Test image preprocessing"""
        sample_path = 'samples/handwriting/sample_good.png'
        if os.path.exists(sample_path):
            processed = self.analyzer.preprocess_image(sample_path)
            self.assertIsNotNone(processed)
            self.assertEqual(len(processed.shape), 2)  # Should be grayscale
    
    def test_contour_detection(self):
        """Test contour detection"""
        sample_path = 'samples/handwriting/sample_good.png'
        if os.path.exists(sample_path):
            binary = self.analyzer.preprocess_image(sample_path)
            contours = self.analyzer.detect_contours(binary)
            self.assertIsInstance(contours, list)
            self.assertGreaterEqual(len(contours), 0)
    
    def test_feature_extraction(self):
        """Test complete feature extraction"""
        sample_path = 'samples/handwriting/sample_good.png'
        if os.path.exists(sample_path):
            features = self.analyzer.extract_features(sample_path)
            
            self.assertIsInstance(features, HandwritingFeatures)
            self.assertGreaterEqual(features.avg_letter_size, 0)
            self.assertGreaterEqual(features.line_straightness, 0)
            self.assertLessEqual(features.line_straightness, 100)
            self.assertGreaterEqual(features.writing_pressure, 0)
            self.assertLessEqual(features.writing_pressure, 100)
    
    def test_invalid_image(self):
        """Test handling of invalid image paths"""
        with self.assertRaises(ValueError):
            self.analyzer.preprocess_image('nonexistent.png')

if __name__ == '__main__':
    unittest.main()
