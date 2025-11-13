import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (classification_report, confusion_matrix, 
                            accuracy_score, precision_recall_fscore_support)
import joblib
import os
import json
from datetime import datetime
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

import os
import sys

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')


class DisabilityPredictor:
    """
    Main ML class for learning disability prediction
    Supports multiple models: Random Forest, SVM, Gradient Boosting, Neural Network
    """
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize predictor with specified model type
        
        Args:
            model_type: 'random_forest', 'svm', 'gradient_boosting', or 'neural_network'
        """
        self.model_type = model_type
        self.model = self._initialize_model()
        self.is_trained = False
        self.feature_importance = None
        self.training_history = {}
    
    def _initialize_model(self):
        """Initialize model based on type"""
        if self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        
        elif self.model_type == 'svm':
            return SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42
            )
        
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        
        elif self.model_type == 'neural_network':
            return MLPClassifier(
                hidden_layer_sizes=(64, 32, 16),
                activation='relu',
                solver='adam',
                learning_rate='adaptive',
                max_iter=500,
                random_state=42
            )
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
             X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict:
        """
        Train the model
        
        Returns:
            Dictionary with training metrics
        """
        print(f"\nTraining {self.model_type.upper()} model...")
        print(f"Training samples: {len(X_train)}")
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Training metrics
        train_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_pred)
        
        metrics = {
            'train_accuracy': train_accuracy,
            'train_samples': len(X_train)
        }
        
        # Validation metrics if provided
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_accuracy = accuracy_score(y_val, val_pred)
            metrics['val_accuracy'] = val_accuracy
            metrics['val_samples'] = len(X_val)
            
            print(f"Training Accuracy: {train_accuracy:.4f}")
            print(f"Validation Accuracy: {val_accuracy:.4f}")
        else:
            print(f"Training Accuracy: {train_accuracy:.4f}")
        
        # Store feature importance for tree-based models
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = self.model.feature_importances_
        
        self.training_history = metrics
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise ValueError(f"{self.model_type} does not support probability predictions")
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, 
                label_names: List[str] = None) -> Dict:
        """
        Comprehensive model evaluation
        
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        # Predictions
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, average='weighted'
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Classification report
        if label_names is None:
            label_names = [f"Class_{i}" for i in range(len(np.unique(y_test)))]
        
        report = classification_report(y_test, y_pred, target_names=label_names, 
                                      output_dict=True)
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, X_test, y_test, cv=5)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'test_samples': len(X_test)
        }
        
        # Print results
        print(f"\n{'='*60}")
        print(f"EVALUATION RESULTS - {self.model_type.upper()}")
        print(f"{'='*60}")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Cross-Validation Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print(f"\nConfusion Matrix:")
        print(cm)
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=label_names))
        
        return metrics
    
    def hyperparameter_tuning(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """
        Perform hyperparameter tuning using GridSearchCV
        """
        print(f"\nPerforming hyperparameter tuning for {self.model_type}...")
        
        param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'svm': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto', 0.01, 0.1],
                'kernel': ['rbf', 'linear']
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'neural_network': {
                'hidden_layer_sizes': [(64, 32), (128, 64, 32), (64, 32, 16)],
                'learning_rate': ['constant', 'adaptive'],
                'alpha': [0.0001, 0.001, 0.01]
            }
        }
        
        param_grid = param_grids.get(self.model_type, {})
        
        if not param_grid:
            print(f"No parameter grid defined for {self.model_type}")
            return {}
        
        # Grid search
        grid_search = GridSearchCV(
            self.model, 
            param_grid, 
            cv=5, 
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        self.is_trained = True
        
        print(f"\nBest Parameters: {grid_search.best_params_}")
        print(f"Best Cross-Validation Score: {grid_search.best_score_:.4f}")
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
    
    def get_feature_importance(self, feature_names: List[str] = None) -> pd.DataFrame:
        """
        Get feature importance (for tree-based models)
        """
        if self.feature_importance is None:
            print("Feature importance not available for this model type")
            return None
        
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(self.feature_importance))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, output_dir: str = None, model_name: str = None):
        """
        Save trained model and metadata
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        if output_dir is None:
            output_dir = MODELS_DIR
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate model name
        if model_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_name = f"{self.model_type}_{timestamp}"
        
        # Save model
        model_path = os.path.join(output_dir, f"{model_name}.pkl")
        joblib.dump(self.model, model_path)
        
        # Save metadata
        metadata = {
            'model_type': self.model_type,
            'model_name': model_name,
            'training_history': self.training_history,
            'is_trained': self.is_trained,
            'save_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        metadata_path = os.path.join(output_dir, f"{model_name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f"\nModel saved to: {model_path}")
        print(f"Metadata saved to: {metadata_path}")
        
        return model_path, metadata_path

    
    def load_model(self, model_path: str):
        """
        Load trained model
        """
        self.model = joblib.load(model_path)
        self.is_trained = True
        
        # Try to load metadata
        metadata_path = model_path.replace('.pkl', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.model_type = metadata.get('model_type', self.model_type)
                self.training_history = metadata.get('training_history', {})
        
        print(f"Model loaded from: {model_path}")


if __name__ == "__main__":
    # Test the predictor with dummy data
    from sklearn.datasets import make_classification
    
    # Generate synthetic data for testing
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=3,
                               n_informative=15, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Test Random Forest
    print("\n" + "="*60)
    print("Testing Random Forest Classifier")
    print("="*60)
    
    rf_predictor = DisabilityPredictor(model_type='random_forest')
    rf_predictor.train(X_train, y_train, X_test, y_test)
    metrics = rf_predictor.evaluate(X_test, y_test, 
                                    label_names=['Normal', 'Dyslexia', 'Dysgraphia'])
    
    # Feature importance
    importance_df = rf_predictor.get_feature_importance()
    print("\nTop 10 Important Features:")
    print(importance_df.head(10))
    
    # Save model
    rf_predictor.save_model()
