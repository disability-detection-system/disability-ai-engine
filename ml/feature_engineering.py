import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """
    Feature engineering and preprocessing for disability detection
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_selector = None
        self.selected_features = None
        self.feature_columns = None
    
    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features from existing ones
        """
        df_copy = df.copy()
        
        # Handwriting derived features
        if 'avg_letter_size' in df.columns and 'contour_count' in df.columns:
            df_copy['size_per_contour'] = df['avg_letter_size'] / (df['contour_count'] + 1)
        
        if 'letter_spacing' in df.columns and 'word_spacing' in df.columns:
            df_copy['spacing_ratio'] = df['word_spacing'] / (df['letter_spacing'] + 1)
        
        if 'letter_formation_quality' in df.columns and 'consistency_score' in df.columns:
            df_copy['quality_consistency_product'] = (
                df['letter_formation_quality'] * df['consistency_score'] / 100
            )
        
        # Speech derived features
        if 'reading_speed_wpm' in df.columns and 'pause_frequency' in df.columns:
            df_copy['reading_efficiency'] = (
                df['reading_speed_wpm'] / (df['pause_frequency'] + 0.1)
            )
        
        if 'pronunciation_score' in df.columns and 'fluency_score' in df.columns:
            df_copy['speech_quality'] = (
                df['pronunciation_score'] + df['fluency_score']
            ) / 2
        
        if 'word_count' in df.columns and 'total_duration' in df.columns:
            df_copy['words_per_second'] = df['word_count'] / (df['total_duration'] + 1)
        
        # Combined features
        if 'letter_formation_quality' in df.columns and 'pronunciation_score' in df.columns:
            df_copy['overall_motor_skill'] = (
                df['letter_formation_quality'] * 0.5 + df['pronunciation_score'] * 0.5
            )
        
        if 'consistency_score' in df.columns and 'fluency_score' in df.columns:
            df_copy['overall_consistency'] = (
                df['consistency_score'] * 0.5 + df['fluency_score'] * 0.5
            )
        
        # Age-adjusted features
        if 'age' in df.columns:
            age_normalized = (df['age'] - 6) / 6
            
            if 'writing_pressure' in df.columns:
                df_copy['age_adjusted_pressure'] = df['writing_pressure'] * (1 + age_normalized * 0.3)
            
            if 'reading_speed_wpm' in df.columns:
                df_copy['age_adjusted_reading'] = df['reading_speed_wpm'] / (1 + age_normalized * 0.5)
        
        return df_copy
    
    def prepare_features(self, df: pd.DataFrame, is_training: bool = True):
        """
        Prepare features for model training/prediction
        """
        # First create derived features
        df_processed = self.create_derived_features(df)
        
        if is_training:
            # Define feature columns (exclude target and metadata)
            feature_cols = [col for col in df_processed.columns 
                           if col not in ['sample_id', 'condition', 'disability_category']]
            self.feature_columns = feature_cols
            
            X = df_processed[feature_cols].copy()
            
            # Handle missing values
            X = X.fillna(X.mean())
            
            # Convert to numpy array to avoid feature name issues
            X_values = X.values
            
            # Fit and transform
            X_scaled = self.scaler.fit_transform(X_values)
            
            # Encode labels
            y = self.label_encoder.fit_transform(df_processed['condition'])
            
            # Feature selection
            self.feature_selector = SelectKBest(score_func=f_classif, k=min(20, X.shape[1]))
            X_selected = self.feature_selector.fit_transform(X_scaled, y)
            
            # Store selected feature names
            selected_indices = self.feature_selector.get_support(indices=True)
            self.selected_features = [feature_cols[i] for i in selected_indices]
            
            return X_selected, y, self.selected_features
        
        else:
            # For prediction, ensure all required columns exist
            if self.feature_columns is None:
                raise ValueError("Model must be trained first or feature_columns must be loaded")
            
            # Add missing columns with 0 values
            for col in self.feature_columns:
                if col not in df_processed.columns:
                    df_processed[col] = 0
            
            # Select only the feature columns in the correct order
            X = df_processed[self.feature_columns].copy()
            
            # Handle missing values (use 0 for prediction)
            X = X.fillna(0)
            
            # Convert to numpy array
            X_values = X.values
            
            # Transform
            X_scaled = self.scaler.transform(X_values)
            X_selected = self.feature_selector.transform(X_scaled)
            
            return X_selected
    
    def save_preprocessors(self, output_dir: str = 'models'):
        """
        Save scaler and encoders
        """
        os.makedirs(output_dir, exist_ok=True)
        
        joblib.dump(self.scaler, os.path.join(output_dir, 'scaler.pkl'))
        joblib.dump(self.label_encoder, os.path.join(output_dir, 'label_encoder.pkl'))
        joblib.dump(self.feature_selector, os.path.join(output_dir, 'feature_selector.pkl'))
        
        # Save selected features list
        with open(os.path.join(output_dir, 'selected_features.txt'), 'w') as f:
            f.write('\n'.join(self.selected_features))
        
        # Save feature columns list
        with open(os.path.join(output_dir, 'feature_columns.txt'), 'w') as f:
            f.write('\n'.join(self.feature_columns))
        
        print(f"Preprocessors saved to {output_dir}/")
    
    def load_preprocessors(self, model_dir: str = 'models'):
        """
        Load saved preprocessors
        """
        self.scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
        self.label_encoder = joblib.load(os.path.join(model_dir, 'label_encoder.pkl'))
        self.feature_selector = joblib.load(os.path.join(model_dir, 'feature_selector.pkl'))
        
        with open(os.path.join(model_dir, 'selected_features.txt'), 'r') as f:
            self.selected_features = [line.strip() for line in f]
        
        with open(os.path.join(model_dir, 'feature_columns.txt'), 'r') as f:
            self.feature_columns = [line.strip() for line in f]
        
        print("Preprocessors loaded successfully")


if __name__ == "__main__":
    # Test feature engineering
    from data_generator import SyntheticDataGenerator
    
    # Generate sample data
    generator = SyntheticDataGenerator()
    dataset = generator.generate_combined_dataset(n_samples_per_condition=100)
    
    # Feature engineering
    engineer = FeatureEngineer()
    X, y, selected_features = engineer.prepare_features(dataset, is_training=True)
    
    print(f"\nOriginal features: {dataset.shape[1] - 3}")
    print(f"After feature engineering: {X.shape[1]}")
    print(f"\nSelected features ({len(selected_features)}):")
    for i, feature in enumerate(selected_features, 1):
        print(f"{i}. {feature}")
    
    # Save preprocessors
    engineer.save_preprocessors()
    
    # Test prediction mode
    print("\n" + "="*60)
    print("Testing prediction mode")
    print("="*60)
    
    test_sample = dataset.iloc[[0]].copy()
    test_sample = test_sample.drop(['condition', 'disability_category', 'sample_id'], axis=1, errors='ignore')
    
    X_test = engineer.prepare_features(test_sample, is_training=False)
    print(f"Test sample shape: {X_test.shape}")
    print("✓ Prediction mode works!")
