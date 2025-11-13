import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import json
import os
from datetime import datetime

import os
import sys

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')


class SyntheticDataGenerator:
    """
    Generate synthetic datasets for dyslexia and dysgraphia detection
    """
    
    def __init__(self, seed=42):
        np.random.seed(seed)
        self.seed = seed
    
    def generate_handwriting_features(self, condition: str, n_samples: int) -> pd.DataFrame:
        """
        Generate synthetic handwriting features based on condition
        Features extracted from handwriting_analyzer.py
        """
        data = []
        
        for _ in range(n_samples):
            if condition == 'dysgraphia':
                # Dysgraphia characteristics: poor motor control, inconsistent writing
                features = {
                    'avg_letter_size': np.random.normal(1800, 600),  # Larger, more variable
                    'line_straightness': np.random.normal(35, 15),  # Poor straightness
                    'letter_spacing': np.random.normal(45, 20),  # Inconsistent spacing
                    'word_spacing': np.random.normal(90, 40),
                    'writing_pressure': np.random.normal(35, 20),  # Inconsistent pressure
                    'letter_formation_quality': np.random.normal(30, 15),  # Poor formation
                    'slant_angle': np.random.normal(25, 15),  # Inconsistent slant
                    'consistency_score': np.random.normal(25, 12),  # Low consistency
                    'contour_count': int(np.random.normal(45, 15)),  # Fewer recognizable letters
                    'aspect_ratio': np.random.normal(2.5, 0.8),
                    'condition': 'dysgraphia',
                    'disability_category': 'learning_disability'
                }
            
            elif condition == 'dyslexia':
                # Dyslexia: handwriting may be affected but less than dysgraphia
                features = {
                    'avg_letter_size': np.random.normal(1200, 400),
                    'line_straightness': np.random.normal(55, 12),  # Moderate quality
                    'letter_spacing': np.random.normal(28, 12),
                    'word_spacing': np.random.normal(56, 24),
                    'writing_pressure': np.random.normal(58, 15),
                    'letter_formation_quality': np.random.normal(52, 15),
                    'slant_angle': np.random.normal(12, 10),
                    'consistency_score': np.random.normal(48, 15),
                    'contour_count': int(np.random.normal(65, 12)),
                    'aspect_ratio': np.random.normal(1.8, 0.5),
                    'condition': 'dyslexia',
                    'disability_category': 'learning_disability'
                }
            
            else:  # normal/control
                # Normal development: good motor control, consistent writing
                features = {
                    'avg_letter_size': np.random.normal(1000, 200),
                    'line_straightness': np.random.normal(78, 10),  # Good straightness
                    'letter_spacing': np.random.normal(25, 8),
                    'word_spacing': np.random.normal(50, 15),
                    'writing_pressure': np.random.normal(72, 12),
                    'letter_formation_quality': np.random.normal(75, 12),
                    'slant_angle': np.random.normal(5, 8),
                    'consistency_score': np.random.normal(72, 12),
                    'contour_count': int(np.random.normal(85, 10)),
                    'aspect_ratio': np.random.normal(1.5, 0.3),
                    'condition': 'normal',
                    'disability_category': 'none'
                }
            
            # Add age factor (6-12 years)
            age = np.random.randint(6, 13)
            features['age'] = age
            
            # Clip values to realistic ranges
            features['avg_letter_size'] = np.clip(features['avg_letter_size'], 200, 3000)
            features['line_straightness'] = np.clip(features['line_straightness'], 0, 100)
            features['letter_spacing'] = np.clip(features['letter_spacing'], 5, 100)
            features['writing_pressure'] = np.clip(features['writing_pressure'], 0, 100)
            features['letter_formation_quality'] = np.clip(features['letter_formation_quality'], 0, 100)
            features['consistency_score'] = np.clip(features['consistency_score'], 0, 100)
            
            data.append(features)
        
        return pd.DataFrame(data)
    
    def generate_speech_features(self, condition: str, n_samples: int) -> pd.DataFrame:
        """
        Generate synthetic speech features based on condition
        Features extracted from speech_analyzer.py
        """
        data = []
        
        for _ in range(n_samples):
            if condition == 'dyslexia':
                # Dyslexia characteristics: reading difficulties, slow reading, pauses
                features = {
                    'reading_speed_wpm': np.random.normal(45, 15),  # Slow reading
                    'pause_frequency': np.random.normal(2.5, 0.8),  # Frequent pauses
                    'average_pause_duration': np.random.normal(1.2, 0.4),  # Long pauses
                    'pronunciation_score': np.random.normal(45, 15),  # Poor pronunciation
                    'fluency_score': np.random.normal(40, 15),  # Low fluency
                    'volume_consistency': np.random.normal(55, 12),
                    'pitch_variation': np.random.normal(48, 15),
                    'speech_clarity': np.random.normal(42, 15),
                    'word_count': int(np.random.normal(35, 12)),  # Fewer words
                    'total_duration': np.random.normal(55, 15),  # Takes longer
                    'condition': 'dyslexia',
                    'disability_category': 'learning_disability'
                }
            
            elif condition == 'dysgraphia':
                # Dysgraphia: speech usually not affected much
                features = {
                    'reading_speed_wpm': np.random.normal(78, 12),
                    'pause_frequency': np.random.normal(0.8, 0.3),
                    'average_pause_duration': np.random.normal(0.4, 0.15),
                    'pronunciation_score': np.random.normal(72, 12),
                    'fluency_score': np.random.normal(70, 12),
                    'volume_consistency': np.random.normal(68, 12),
                    'pitch_variation': np.random.normal(72, 12),
                    'speech_clarity': np.random.normal(70, 12),
                    'word_count': int(np.random.normal(65, 10)),
                    'total_duration': np.random.normal(35, 8),
                    'condition': 'dysgraphia',
                    'disability_category': 'learning_disability'
                }
            
            else:  # normal/control
                # Normal speech development
                features = {
                    'reading_speed_wpm': np.random.normal(95, 15),  # Good reading speed
                    'pause_frequency': np.random.normal(0.5, 0.2),  # Natural pauses
                    'average_pause_duration': np.random.normal(0.3, 0.1),  # Short pauses
                    'pronunciation_score': np.random.normal(80, 10),
                    'fluency_score': np.random.normal(82, 10),
                    'volume_consistency': np.random.normal(75, 10),
                    'pitch_variation': np.random.normal(78, 10),
                    'speech_clarity': np.random.normal(80, 10),
                    'word_count': int(np.random.normal(75, 12)),
                    'total_duration': np.random.normal(30, 6),
                    'condition': 'normal',
                    'disability_category': 'none'
                }
            
            # Add age factor
            age = np.random.randint(6, 13)
            features['age'] = age
            
            # Clip values
            features['reading_speed_wpm'] = np.clip(features['reading_speed_wpm'], 10, 150)
            features['pronunciation_score'] = np.clip(features['pronunciation_score'], 0, 100)
            features['fluency_score'] = np.clip(features['fluency_score'], 0, 100)
            features['speech_clarity'] = np.clip(features['speech_clarity'], 0, 100)
            
            data.append(features)
        
        return pd.DataFrame(data)
    
    def generate_combined_dataset(self, n_samples_per_condition: int = 1000) -> pd.DataFrame:
        """
        Generate combined dataset with handwriting and speech features
        """
        all_data = []
        conditions = ['normal', 'dyslexia', 'dysgraphia']
        
        for condition in conditions:
            print(f"Generating {n_samples_per_condition} samples for {condition}...")
            
            # Generate handwriting features
            hw_data = self.generate_handwriting_features(condition, n_samples_per_condition)
            
            # Generate speech features
            speech_data = self.generate_speech_features(condition, n_samples_per_condition)
            
            # Combine (same sample ID gets both handwriting and speech data)
            combined = pd.concat([hw_data.reset_index(drop=True), 
                                 speech_data.drop(['condition', 'disability_category', 'age'], axis=1).reset_index(drop=True)], 
                                axis=1)
            
            # Add sample ID
            combined['sample_id'] = [f"{condition}_{i:04d}" for i in range(n_samples_per_condition)]
            
            all_data.append(combined)
        
        # Combine all conditions
        final_dataset = pd.concat(all_data, ignore_index=True)
        
        # Shuffle dataset
        final_dataset = final_dataset.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        
        print(f"\nTotal samples generated: {len(final_dataset)}")
        print(f"Distribution:\n{final_dataset['condition'].value_counts()}")
        
        return final_dataset
    
    def save_dataset(self, dataset: pd.DataFrame, output_dir: str = None):
        """
        Save dataset to CSV and metadata to JSON
        """
        if output_dir is None:
            output_dir = DATA_DIR
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save main dataset
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = os.path.join(output_dir, f'synthetic_dataset_{timestamp}.csv')
        dataset.to_csv(csv_path, index=False)
        print(f"\nDataset saved to: {csv_path}")
        
        # Save metadata
        metadata = {
            'generation_date': timestamp,
            'total_samples': len(dataset),
            'conditions': dataset['condition'].value_counts().to_dict(),
            'features': {
                'handwriting': [
                    'avg_letter_size', 'line_straightness', 'letter_spacing',
                    'word_spacing', 'writing_pressure', 'letter_formation_quality',
                    'slant_angle', 'consistency_score', 'contour_count', 'aspect_ratio'
                ],
                'speech': [
                    'reading_speed_wpm', 'pause_frequency', 'average_pause_duration',
                    'pronunciation_score', 'fluency_score', 'volume_consistency',
                    'pitch_variation', 'speech_clarity', 'word_count', 'total_duration'
                ]
            },
            'seed': self.seed
        }
        
        json_path = os.path.join(output_dir, f'metadata_{timestamp}.json')
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"Metadata saved to: {json_path}")
        
        return csv_path, json_path



if __name__ == "__main__":
    # Generate synthetic dataset
    generator = SyntheticDataGenerator(seed=42)
    dataset = generator.generate_combined_dataset(n_samples_per_condition=1000)
    
    # Save dataset
    csv_path, json_path = generator.save_dataset(dataset)
    
    # Display sample
    print("\n" + "="*60)
    print("Sample data (first 3 rows):")
    print("="*60)
    print(dataset.head(3))
    
    # Display statistics
    print("\n" + "="*60)
    print("Dataset Statistics:")
    print("="*60)
    print(dataset.describe())
