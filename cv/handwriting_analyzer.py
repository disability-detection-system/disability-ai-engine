import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import os
from dataclasses import dataclass
import math

@dataclass
class HandwritingFeatures:
    """Data class to store extracted handwriting features"""
    avg_letter_size: float
    line_straightness: float
    letter_spacing: float
    word_spacing: float
    writing_pressure: float
    letter_formation_quality: float
    slant_angle: float
    consistency_score: float
    contour_count: int
    aspect_ratio: float

class HandwritingAnalyzer:
    def __init__(self):
        self.features = None
        
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess handwriting image"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def detect_contours(self, binary_image: np.ndarray) -> List[np.ndarray]:
        """Detect and filter contours representing letters/characters"""
        contours, _ = cv2.findContours(
            binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter contours by area to remove noise
        min_area = 50
        max_area = 10000
        filtered_contours = [
            c for c in contours 
            if min_area < cv2.contourArea(c) < max_area
        ]
        
        return filtered_contours
    
    def extract_features(self, image_path: str) -> HandwritingFeatures:
        """Extract comprehensive handwriting features"""
        # Preprocess image
        binary_image = self.preprocess_image(image_path)
        original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Detect contours
        contours = self.detect_contours(binary_image)
        
        if not contours:
            # Return default values if no contours found
            return HandwritingFeatures(
                avg_letter_size=0, line_straightness=0, letter_spacing=0,
                word_spacing=0, writing_pressure=0, letter_formation_quality=0,
                slant_angle=0, consistency_score=0, contour_count=0, aspect_ratio=0
            )
        
        # Feature 1: Average letter size
        areas = [cv2.contourArea(c) for c in contours]
        avg_letter_size = np.mean(areas) if areas else 0
        
        # Feature 2: Letter spacing
        bounding_boxes = [cv2.boundingRect(c) for c in contours]
        bounding_boxes.sort(key=lambda x: x[0])  # Sort by x-coordinate
        
        spacings = []
        for i in range(len(bounding_boxes) - 1):
            current_right = bounding_boxes[i][0] + bounding_boxes[i][2]
            next_left = bounding_boxes[i + 1][0]
            spacing = max(0, next_left - current_right)
            spacings.append(spacing)
        
        avg_letter_spacing = np.mean(spacings) if spacings else 0
        
        # Feature 3: Writing pressure (based on stroke thickness)
        writing_pressure = self._calculate_writing_pressure(binary_image)
        
        # Feature 4: Line straightness
        line_straightness = self._calculate_line_straightness(contours)
        
        # Feature 5: Slant angle
        slant_angle = self._calculate_slant_angle(contours)
        
        # Feature 6: Letter formation quality (contour complexity)
        letter_formation_quality = self._calculate_formation_quality(contours)
        
        # Feature 7: Consistency score
        consistency_score = self._calculate_consistency(areas, spacings)
        
        # Feature 8: Aspect ratio
        if bounding_boxes:
            heights = [box[3] for box in bounding_boxes]
            widths = [box[2] for box in bounding_boxes]
            avg_aspect_ratio = np.mean([h/w for h, w in zip(heights, widths) if w > 0])
        else:
            avg_aspect_ratio = 0
        
        features = HandwritingFeatures(
            avg_letter_size=avg_letter_size,
            line_straightness=line_straightness,
            letter_spacing=avg_letter_spacing,
            word_spacing=avg_letter_spacing * 2,  # Approximation
            writing_pressure=writing_pressure,
            letter_formation_quality=letter_formation_quality,
            slant_angle=slant_angle,
            consistency_score=consistency_score,
            contour_count=len(contours),
            aspect_ratio=avg_aspect_ratio
        )
        
        self.features = features
        return features
    
    def _calculate_writing_pressure(self, binary_image: np.ndarray) -> float:
        """Estimate writing pressure based on stroke thickness"""
        # Find connected components to measure thickness
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_image, connectivity=8
        )
        
        if num_labels <= 1:  # Only background
            return 0
        
        # Calculate average component area (excluding background)
        component_areas = stats[1:, cv2.CC_STAT_AREA]  # Skip background
        avg_thickness = np.mean(component_areas) if len(component_areas) > 0 else 0
        
        # Normalize to 0-100 scale
        return min(100, avg_thickness / 10)
    
    def _calculate_line_straightness(self, contours: List[np.ndarray]) -> float:
        """Calculate how straight the writing lines are"""
        if not contours:
            return 0
        
        straightness_scores = []
        
        for contour in contours:
            if len(contour) < 5:
                continue
                
            # Fit line to contour points
            [vx, vy, x, y] = cv2.fitLine(
                contour, cv2.DIST_L2, 0, 0.01, 0.01
            )
            
            # Calculate how well points fit the line
            distances = []
            for point in contour:
                px, py = point[0]
                # Distance from point to line
                distance = abs(vy * (px - x) - vx * (py - y)) / np.sqrt(vx*vx + vy*vy)
                distances.append(distance)
            
            # Lower average distance = more straight
            avg_distance = np.mean(distances)
            straightness = max(0, 100 - avg_distance)
            straightness_scores.append(straightness)
        
        return np.mean(straightness_scores) if straightness_scores else 0
    
    def _calculate_slant_angle(self, contours: List[np.ndarray]) -> float:
        """Calculate the average slant angle of letters"""
        if not contours:
            return 0
        
        angles = []
        
        for contour in contours:
            if len(contour) < 5:
                continue
                
            # Find the minimal area rectangle
            rect = cv2.minAreaRect(contour)
            angle = rect[2]
            
            # Normalize angle to -45 to 45 degrees
            if angle < -45:
                angle += 90
            elif angle > 45:
                angle -= 90
                
            angles.append(angle)
        
        return np.mean(angles) if angles else 0
    
    def _calculate_formation_quality(self, contours: List[np.ndarray]) -> float:
        """Calculate letter formation quality based on contour smoothness"""
        if not contours:
            return 0
        
        quality_scores = []
        
        for contour in contours:
            if len(contour) < 5:
                continue
            
            # Calculate contour perimeter and area
            perimeter = cv2.arcLength(contour, True)
            area = cv2.contourArea(contour)
            
            if area == 0:
                continue
            
            # Circularity: measure how close to a perfect shape
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # Higher circularity often indicates better formation
            quality = min(100, circularity * 100)
            quality_scores.append(quality)
        
        return np.mean(quality_scores) if quality_scores else 0
    
    def _calculate_consistency(self, areas: List[float], spacings: List[float]) -> float:
        """Calculate consistency of letter sizes and spacing"""
        consistency_scores = []
        
        # Size consistency
        if areas:
            size_std = np.std(areas)
            size_mean = np.mean(areas)
            if size_mean > 0:
                size_consistency = max(0, 100 - (size_std / size_mean * 100))
                consistency_scores.append(size_consistency)
        
        # Spacing consistency
        if spacings:
            spacing_std = np.std(spacings)
            spacing_mean = np.mean(spacings)
            if spacing_mean > 0:
                spacing_consistency = max(0, 100 - (spacing_std / spacing_mean * 100))
                consistency_scores.append(spacing_consistency)
        
        return np.mean(consistency_scores) if consistency_scores else 0
    
    def visualize_analysis(self, image_path: str):
        """Visualize the analysis process"""
        original = cv2.imread(image_path)
        binary = self.preprocess_image(image_path)
        contours = self.detect_contours(binary)
        
        # Draw contours on original image
        contour_image = original.copy()
        cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
        
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(binary, cmap='gray')
        plt.title('Preprocessed Binary')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
        plt.title(f'Detected Contours ({len(contours)})')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

# Test the feature extraction
if __name__ == "__main__":
    # First create sample images
    import sys
    import os
    
    # Add the parent directory to the path so we can import from utils module
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from utils.test_data_generator import create_sample_handwriting_images
    create_sample_handwriting_images()
    
    analyzer = HandwritingAnalyzer()
    
    # Test with sample images
    sample_files = ['samples/handwriting/sample_good.png', 
                   'samples/handwriting/sample_poor.png']
    
    for sample_file in sample_files:
        if os.path.exists(sample_file):
            print(f"\nAnalyzing {sample_file}:")
            features = analyzer.extract_features(sample_file)
            
            print(f"Average letter size: {features.avg_letter_size:.2f}")
            print(f"Line straightness: {features.line_straightness:.2f}")
            print(f"Letter spacing: {features.letter_spacing:.2f}")
            print(f"Word spacing: {features.word_spacing:.2f}")
            print(f"Writing pressure: {features.writing_pressure:.2f}")
            print(f"Formation quality: {features.letter_formation_quality:.2f}")
            print(f"Slant angle: {features.slant_angle:.2f}°")
            print(f"Consistency score: {features.consistency_score:.2f}")
            print(f"Contour count: {features.contour_count}")
            print(f"Aspect ratio: {features.aspect_ratio:.2f}")
            
            # Visualize the analysis
            analyzer.visualize_analysis(sample_file)
