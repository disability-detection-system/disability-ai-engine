import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import os
from dataclasses import dataclass


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
        
    def preprocess_image(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Enhanced preprocessing with original image return"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding for better text extraction
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Morphological operations to clean up the image
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return cleaned, gray
    
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
    
    def extract_features(self, image_path: str, age: int = 8) -> HandwritingFeatures:
        """Extract comprehensive handwriting features with age adjustment"""
        # Preprocess image (now returns both binary and original grayscale)
        binary_image, original_grayscale = self.preprocess_image(image_path)
        
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
        
        # Feature 3: Enhanced writing pressure
        writing_pressure = self._calculate_writing_pressure_advanced(binary_image, original_grayscale)
        
        # Feature 4: Advanced line straightness
        line_straightness = self._calculate_line_straightness_advanced(contours, binary_image.shape)
        
        # Feature 5: Slant angle
        slant_angle = self._calculate_slant_angle(contours)
        
        # Feature 6: Advanced letter formation quality
        letter_formation_quality = self._calculate_letter_formation_advanced(contours)
        
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
        
        # Apply age-appropriate scoring adjustments
        adjusted_features = self.calculate_age_appropriate_scoring(features, age)
        self.features = adjusted_features
        return adjusted_features

    def calculate_overall_score(self, features: HandwritingFeatures) -> float:
        """Calculate a more realistic overall score with weighted features"""
        
        scores = {}
        
        # Normalize and score important features
        # Line straightness: 0-100, higher is better
        scores['straightness'] = min(100, max(0, features.line_straightness))
        
        # Consistency: 0-100, higher is better  
        scores['consistency'] = min(100, max(0, features.consistency_score))
        
        # Letter formation: 0-100, higher is better
        scores['formation'] = min(100, max(0, features.letter_formation_quality))
        
        # Letter size: optimal range scoring
        if 500 <= features.avg_letter_size <= 2000:
            scores['size'] = 100
        elif features.avg_letter_size < 500 and features.avg_letter_size > 0:
            scores['size'] = (features.avg_letter_size / 500) * 100
        elif features.avg_letter_size > 2000:
            scores['size'] = max(0, 100 - ((features.avg_letter_size - 2000) / 10))
        else:
            scores['size'] = 0
        
        # Letter spacing: optimal range
        if 10 <= features.letter_spacing <= 50:
            scores['spacing'] = 100
        else:
            scores['spacing'] = max(0, 100 - abs(features.letter_spacing - 30) * 2)
        
        # Writing pressure: moderate is best
        if 20 <= features.writing_pressure <= 80:
            scores['pressure'] = 100
        else:
            scores['pressure'] = max(0, 100 - abs(features.writing_pressure - 50))
        
        # Slant angle: close to 0 is better (vertical)
        scores['slant'] = max(0, 100 - abs(features.slant_angle) * 2)
        
        # Weighted average (some features matter more)
        weights = {
            'straightness': 0.2,    # 20% - Line quality
            'consistency': 0.25,    # 25% - Most important for readability
            'formation': 0.2,       # 20% - Letter shape quality
            'size': 0.1,           # 10% - Appropriate sizing
            'spacing': 0.15,       # 15% - Good spacing
            'pressure': 0.05,      # 5%  - Pressure control
            'slant': 0.05          # 5%  - Slant consistency
        }
        
        weighted_score = sum(scores[key] * weights[key] for key in scores)
        return min(100, max(0, weighted_score))
    
    # ADVANCED FEATURE METHODS
    def _calculate_writing_pressure_advanced(self, binary_image: np.ndarray, 
                                           original_image: np.ndarray) -> float:
        """Enhanced writing pressure estimation using gradient analysis"""
        try:
            # Convert original to grayscale if needed
            if len(original_image.shape) == 3:
                gray_original = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
            else:
                gray_original = original_image.copy()
            
            # Invert so that writing is white on black
            inverted = 255 - gray_original
            
            # Find writing pixels using the binary mask
            writing_pixels = inverted[binary_image > 0]
            
            if len(writing_pixels) == 0:
                return 0.0
            
            # Calculate average intensity of writing pixels
            avg_intensity = np.mean(writing_pixels)
            
            # Calculate intensity variation for pressure consistency
            intensity_std = np.std(writing_pixels)
            consistency_factor = max(0.5, 1 - (intensity_std / avg_intensity)) if avg_intensity > 0 else 0.5
            
            # Normalize to 0-100 scale
            pressure_score = (avg_intensity / 255.0) * 100 * consistency_factor
            
            return min(100, max(0, pressure_score))
            
        except Exception as e:
            print(f"Advanced pressure calculation error: {e}")
            return 50.0

    def _calculate_letter_formation_advanced(self, contours: List[np.ndarray]) -> float:
        """Advanced letter formation quality assessment"""
        if not contours:
            return 0.0
        
        formation_scores = []
        
        for contour in contours:
            if len(contour) < 10:
                continue
            
            # Calculate multiple shape descriptors
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if area == 0 or perimeter == 0:
                continue
            
            # Compactness (how close to a circle)
            compactness = (4 * np.pi * area) / (perimeter ** 2)
            
            # Convexity
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            convexity = area / hull_area if hull_area > 0 else 0
            
            # Aspect ratio
            rect = cv2.minAreaRect(contour)
            width, height = rect[1]  # rect[1] contains (width, height)
            aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else 1
            
            # Solidity (how filled the shape is)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # Extent (ratio of contour area to bounding rectangle area)
            x, y, w, h = cv2.boundingRect(contour)
            extent = area / (w * h) if (w * h) > 0 else 0
            
            # Combine metrics for formation quality
            quality_score = 0
            
            # Compactness scoring (0.2-0.8 is good for letters)
            if 0.2 <= compactness <= 0.8:
                quality_score += 25
            elif 0.1 <= compactness <= 0.9:
                quality_score += 15
            
            # Convexity scoring (higher is better)
            quality_score += convexity * 25
            
            # Aspect ratio scoring (1-4 is reasonable for letters)
            if 1 <= aspect_ratio <= 4:
                quality_score += 25
            elif aspect_ratio <= 6:
                quality_score += 15
            else:
                quality_score += 5
            
            # Solidity scoring
            quality_score += solidity * 15
            
            # Extent scoring
            if extent > 0.3:
                quality_score += 10
            
            formation_scores.append(min(100, quality_score))
        
        return np.mean(formation_scores) if formation_scores else 0

    def _calculate_line_straightness_advanced(self, contours: List[np.ndarray], 
                                            image_shape: Tuple[int, int]) -> float:
        """Advanced line straightness measurement - FIXED VERSION"""
        if not contours:
            return 0.0
        
        height, width = image_shape
        
        # Group contours by approximate y-coordinate (text lines)
        contour_centers = []
        
        for contour in contours:
            if cv2.contourArea(contour) < 50:
                continue
            
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                contour_centers.append((cx, cy))
        
        if not contour_centers:
            return 0.0
        
        # Sort by y-coordinate and group into lines
        contour_centers.sort(key=lambda x: x[1])  # Sort by y-coordinate
        
        line_groups = []
        current_line = [contour_centers[0]]
        line_threshold = height * 0.05  # 5% of image height
        
        for cx, cy in contour_centers[1:]:
            if abs(cy - current_line[-1][1]) < line_threshold:
                current_line.append((cx, cy))
            else:
                if len(current_line) >= 2:
                    line_groups.append(current_line)
                current_line = [(cx, cy)]
        
        if len(current_line) >= 2:
            line_groups.append(current_line)
        
        # Calculate straightness for each line
        straightness_scores = []
        
        for line in line_groups:
            if len(line) < 3:
                continue
            
            # Extract coordinates
            x_coords = [point[0] for point in line]
            y_coords = [point[1] for point in line]
            
            # Fit a line and calculate deviation
            if len(set(x_coords)) > 1:  # Ensure we don't have vertical line
                try:
                    z = np.polyfit(x_coords, y_coords, 1)
                    fitted_line = np.poly1d(z)
                    
                    # Calculate deviations
                    deviations = []
                    for x, y in zip(x_coords, y_coords):
                        expected_y = fitted_line(x)
                        deviation = abs(y - expected_y)
                        deviations.append(deviation)
                    
                    avg_deviation = np.mean(deviations)
                    # Convert to straightness score (lower deviation = higher straightness)
                    max_acceptable_deviation = height * 0.02  # 2% of image height
                    straightness = max(0, 100 - (avg_deviation / max_acceptable_deviation * 100))
                    straightness_scores.append(min(100, straightness))
                    
                except (np.RankWarning, ValueError, TypeError):
                    # If polyfit fails, assume moderate straightness
                    straightness_scores.append(50)
        
        return np.mean(straightness_scores) if straightness_scores else 50.0

    def calculate_age_appropriate_scoring(self, features: HandwritingFeatures, 
                                        age: int = 8) -> HandwritingFeatures:
        """Adjust scoring based on age-appropriate expectations"""
        # Age-based adjustment factors
        age_factors = {
            5: {'pressure': 0.8, 'consistency': 0.7, 'formation': 0.8, 'straightness': 0.75},
            6: {'pressure': 0.85, 'consistency': 0.75, 'formation': 0.85, 'straightness': 0.8},
            7: {'pressure': 0.9, 'consistency': 0.8, 'formation': 0.9, 'straightness': 0.85},
            8: {'pressure': 1.0, 'consistency': 0.9, 'formation': 0.95, 'straightness': 0.9},
            9: {'pressure': 1.0, 'consistency': 0.95, 'formation': 1.0, 'straightness': 0.95},
            10: {'pressure': 1.0, 'consistency': 1.0, 'formation': 1.0, 'straightness': 1.0},
            11: {'pressure': 1.0, 'consistency': 1.0, 'formation': 1.0, 'straightness': 1.0},
        }
        
        # Use closest age if exact age not found
        closest_age = min(age_factors.keys(), key=lambda x: abs(x - age))
        factors = age_factors[closest_age]
        
        # Adjust scores - divide by factor to make expectations age-appropriate
        adjusted_features = HandwritingFeatures(
            avg_letter_size=features.avg_letter_size,
            line_straightness=min(100, features.line_straightness / factors['straightness']),
            letter_spacing=features.letter_spacing,
            word_spacing=features.word_spacing,
            writing_pressure=min(100, features.writing_pressure / factors['pressure']),
            letter_formation_quality=min(100, features.letter_formation_quality / factors['formation']),
            slant_angle=features.slant_angle,
            consistency_score=min(100, features.consistency_score / factors['consistency']),
            contour_count=features.contour_count,
            aspect_ratio=features.aspect_ratio
        )
        
        return adjusted_features

    # ORIGINAL METHODS (for compatibility)
    def _calculate_slant_angle(self, contours: List[np.ndarray]) -> float:
        """Calculate the average slant angle of letters - FIXED VERSION"""
        if not contours:
            return 0
        
        angles = []
        
        for contour in contours:
            if len(contour) < 5:
                continue
                
            try:
                # Find the minimal area rectangle
                rect = cv2.minAreaRect(contour)
                angle = rect[2]  # rect[2] is the angle
                
                # Normalize angle to -45 to 45 degrees
                if angle < -45:
                    angle += 90
                elif angle > 45:
                    angle -= 90
                    
                angles.append(angle)
            except Exception:
                # If minAreaRect fails, skip this contour
                continue
        
        return np.mean(angles) if angles else 0
    
    def _calculate_consistency(self, areas: List[float], spacings: List[float]) -> float:
        """Calculate consistency of letter sizes and spacing"""
        consistency_scores = []
        
        # Size consistency
        if areas and len(areas) > 1:
            size_std = np.std(areas)
            size_mean = np.mean(areas)
            if size_mean > 0:
                size_consistency = max(0, 100 - (size_std / size_mean * 100))
                consistency_scores.append(size_consistency)
        
        # Spacing consistency
        if spacings and len(spacings) > 1:
            spacing_std = np.std(spacings)
            spacing_mean = np.mean(spacings)
            if spacing_mean > 0:
                spacing_consistency = max(0, 100 - (spacing_std / spacing_mean * 100))
                consistency_scores.append(spacing_consistency)
        
        return np.mean(consistency_scores) if consistency_scores else 50.0

    def visualize_analysis(self, image_path: str):
        """Basic visualization method"""
        try:
            original = cv2.imread(image_path)
            binary, gray = self.preprocess_image(image_path)
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
        except Exception as e:
            print(f"Visualization error: {e}")


# Test the enhanced analyzer
if __name__ == "__main__":
    # First create sample images
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    try:
        from utils.test_data_generator import create_sample_handwriting_images
        create_sample_handwriting_images()
    except ImportError:
        print("Could not import test_data_generator, skipping sample creation")
    
    analyzer = HandwritingAnalyzer()
    
    # Test with sample images
    sample_files = ['samples/handwriting/sample_good.png', 
                   'samples/handwriting/sample_poor.png']
    
    for sample_file in sample_files:
        if os.path.exists(sample_file):
            print(f"\n{'='*60}")
            print(f"ANALYZING: {sample_file}")
            print(f"{'='*60}")
            
            try:
                features = analyzer.extract_features(sample_file, age=8)
                overall_score = analyzer.calculate_overall_score(features)
                
                print(f"Overall Score: {overall_score:.1f}%")
                print(f"Line Straightness: {features.line_straightness:.1f}%")
                print(f"Letter Formation: {features.letter_formation_quality:.1f}%")
                print(f"Writing Pressure: {features.writing_pressure:.1f}%")
                print(f"Consistency: {features.consistency_score:.1f}%")
                print(f"Contours: {features.contour_count}")
                
                # Show visualization
                analyzer.visualize_analysis(sample_file)
                
            except Exception as e:
                print(f"Error analyzing {sample_file}: {e}")
