"""
Training script for learning disability detection models
Run from disability-ai-engine root: python train_models.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from ml.data_generator import SyntheticDataGenerator
from ml.feature_engineering import FeatureEngineer
from ml.disability_predictor import DisabilityPredictor

# Define paths
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
VIZ_DIR = os.path.join(MODELS_DIR, 'visualizations')


def plot_confusion_matrix(cm, labels, model_name):
    """Plot and save confusion matrix"""
    os.makedirs(VIZ_DIR, exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    filepath = os.path.join(VIZ_DIR, f'confusion_matrix_{model_name}.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {filepath}")


def plot_feature_importance(importance_df, model_name, top_n=15):
    """Plot and save feature importance"""
    os.makedirs(VIZ_DIR, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    top_features = importance_df.head(top_n)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance Score')
    plt.title(f'Top {top_n} Feature Importances - {model_name}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    filepath = os.path.join(VIZ_DIR, f'feature_importance_{model_name}.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Feature importance plot saved to: {filepath}")


def compare_models(results):
    """Compare model performances"""
    os.makedirs(VIZ_DIR, exist_ok=True)
    
    # Extract metrics
    model_names = list(results.keys())
    accuracies = [results[m]['metrics']['accuracy'] for m in model_names]
    f1_scores = [results[m]['metrics']['f1_score'] for m in model_names]
    precisions = [results[m]['metrics']['precision'] for m in model_names]
    recalls = [results[m]['metrics']['recall'] for m in model_names]
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Accuracy
    axes[0, 0].bar(model_names, accuracies, color='skyblue')
    axes[0, 0].set_title('Model Accuracy Comparison')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_ylim([0, 1])
    for i, v in enumerate(accuracies):
        axes[0, 0].text(i, v + 0.02, f'{v:.3f}', ha='center')
    
    # F1-Score
    axes[0, 1].bar(model_names, f1_scores, color='lightcoral')
    axes[0, 1].set_title('F1-Score Comparison')
    axes[0, 1].set_ylabel('F1-Score')
    axes[0, 1].set_ylim([0, 1])
    for i, v in enumerate(f1_scores):
        axes[0, 1].text(i, v + 0.02, f'{v:.3f}', ha='center')
    
    # Precision
    axes[1, 0].bar(model_names, precisions, color='lightgreen')
    axes[1, 0].set_title('Precision Comparison')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_ylim([0, 1])
    for i, v in enumerate(precisions):
        axes[1, 0].text(i, v + 0.02, f'{v:.3f}', ha='center')
    
    # Recall
    axes[1, 1].bar(model_names, recalls, color='plum')
    axes[1, 1].set_title('Recall Comparison')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].set_ylim([0, 1])
    for i, v in enumerate(recalls):
        axes[1, 1].text(i, v + 0.02, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    filepath = os.path.join(VIZ_DIR, 'model_comparison.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Model comparison plot saved to: {filepath}")


def train_all_models(dataset_path=None, n_samples=1000, test_size=0.2, perform_tuning=False):
    """
    Main training function
    """
    print("="*80)
    print("LEARNING DISABILITY DETECTION - MODEL TRAINING")
    print("="*80)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Models Directory: {MODELS_DIR}")
    
    # Step 1: Load or generate dataset
    if dataset_path and os.path.exists(dataset_path):
        print(f"\nLoading dataset from: {dataset_path}")
        dataset = pd.read_csv(dataset_path)
    else:
        print(f"\nGenerating synthetic dataset ({n_samples} samples per condition)...")
        generator = SyntheticDataGenerator(seed=42)
        dataset = generator.generate_combined_dataset(n_samples_per_condition=n_samples)
        dataset_path, _ = generator.save_dataset(dataset)
    
    print(f"Total samples: {len(dataset)}")
    print(f"Class distribution:\n{dataset['condition'].value_counts()}")
    
    # Step 2: Feature engineering
    print("\n" + "="*80)
    print("FEATURE ENGINEERING")
    print("="*80)
    
    engineer = FeatureEngineer()
    X, y, selected_features = engineer.prepare_features(dataset, is_training=True)
    engineer.save_preprocessors()
    
    print(f"Features after engineering: {X.shape[1]}")
    print(f"Selected features: {len(selected_features)}")
    
    # Step 3: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"\nTrain samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Label names
    label_names = engineer.label_encoder.classes_
    print(f"Labels: {label_names}")
    
    # Step 4: Train models
    model_types = ['random_forest', 'gradient_boosting', 'svm', 'neural_network']
    results = {}
    
    for model_type in model_types:
        print("\n" + "="*80)
        print(f"TRAINING {model_type.upper().replace('_', ' ')}")
        print("="*80)
        
        try:
            # Initialize predictor
            predictor = DisabilityPredictor(model_type=model_type)
            
            # Hyperparameter tuning (optional)
            if perform_tuning:
                tuning_results = predictor.hyperparameter_tuning(X_train, y_train)
            else:
                # Train with default parameters
                predictor.train(X_train, y_train, X_val, y_val)
            
            # Evaluate
            metrics = predictor.evaluate(X_test, y_test, label_names=label_names)
            
            # Save model
            model_path, metadata_path = predictor.save_model()
            
            # Visualizations
            cm = np.array(metrics['confusion_matrix'])
            plot_confusion_matrix(cm, label_names, model_type)
            
            # Feature importance (tree-based models only)
            if model_type in ['random_forest', 'gradient_boosting']:
                importance_df = predictor.get_feature_importance(selected_features)
                if importance_df is not None:
                    plot_feature_importance(importance_df, model_type)
            
            # Store results
            results[model_type] = {
                'predictor': predictor,
                'metrics': metrics,
                'model_path': model_path
            }
            
        except Exception as e:
            print(f"Error training {model_type}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Step 5: Compare models
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    
    if results:
        compare_models(results)
        
        # Find best model
        best_model = max(results.items(), key=lambda x: x[1]['metrics']['accuracy'])
        print(f"\nBest Model: {best_model[0].upper()}")
        print(f"Accuracy: {best_model[1]['metrics']['accuracy']:.4f}")
        print(f"F1-Score: {best_model[1]['metrics']['f1_score']:.4f}")
        
        # Save best model info
        best_model_path = os.path.join(MODELS_DIR, 'best_model.txt')
        with open(best_model_path, 'w') as f:
            f.write(f"Best Model: {best_model[0]}\n")
            f.write(f"Model Path: {best_model[1]['model_path']}\n")
            f.write(f"Accuracy: {best_model[1]['metrics']['accuracy']:.4f}\n")
            f.write(f"F1-Score: {best_model[1]['metrics']['f1_score']:.4f}\n")
        
        print(f"\nBest model info saved to: {best_model_path}")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    
    return results


if __name__ == "__main__":
    # Run training
    results = train_all_models(
        n_samples=1000,  # Samples per condition
        test_size=0.2,
        perform_tuning=False  # Set to True for hyperparameter tuning (takes longer)
    )
