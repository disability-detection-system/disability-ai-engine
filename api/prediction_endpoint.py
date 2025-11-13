"""
Enhanced Flask API endpoint with recommendation engine integration
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
import sys
from datetime import datetime

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from ml.disability_predictor import DisabilityPredictor
from ml.feature_engineering import FeatureEngineer
from ml.recommendation_engine import RecommendationEngine

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Global variables
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
predictor = None
engineer = None
label_encoder = None
recommendation_engine = None


def load_models():
    """Load trained models and preprocessors"""
    global predictor, engineer, label_encoder, recommendation_engine
    
    try:
        # Load feature engineer
        engineer = FeatureEngineer()
        engineer.load_preprocessors(MODELS_DIR)
        
        # Load label encoder
        label_encoder = joblib.load(os.path.join(MODELS_DIR, 'label_encoder.pkl'))
        
        # Load best model
        best_model_path = os.path.join(MODELS_DIR, 'best_model.txt')
        if os.path.exists(best_model_path):
            with open(best_model_path, 'r') as f:
                lines = f.readlines()
                model_file = lines[1].split(': ')[1].strip()
        else:
            # Find most recent model
            model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pkl') and 'scaler' not in f]
            if model_files:
                model_file = os.path.join(MODELS_DIR, sorted(model_files)[-1])
            else:
                raise FileNotFoundError("No trained models found")
        
        predictor = DisabilityPredictor()
        predictor.load_model(model_file)
        
        # Load recommendation engine
        recommendation_engine = RecommendationEngine()
        
        print(f"✓ Models loaded successfully from {MODELS_DIR}")
        return True
        
    except Exception as e:
        print(f"✗ Error loading models: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': predictor is not None,
        'recommendation_engine': recommendation_engine is not None,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Single prediction endpoint with recommendations
    
    Expected JSON:
    {
        "handwriting_features": {...},
        "speech_features": {...},
        "age": 8,
        "student_id": "STU001" (optional)
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Extract features
        hw_features = data.get('handwriting_features', {})
        speech_features = data.get('speech_features', {})
        age = data.get('age', 8)
        student_id = data.get('student_id', 'UNKNOWN')
        
        # Validate required features
        if not hw_features and not speech_features:
            return jsonify({'error': 'At least handwriting or speech features required'}), 400
        
        # Combine features
        feature_dict = {**hw_features, **speech_features, 'age': age}
        input_df = pd.DataFrame([feature_dict])
        
        # Preprocess
        X = engineer.prepare_features(input_df, is_training=False)
        
        # Predict
        prediction = predictor.predict(X)[0]
        prediction_proba = predictor.predict_proba(X)[0]
        
        # Decode label
        predicted_label = label_encoder.inverse_transform([prediction])[0]
        
        # Create probability dictionary
        prob_dict = {
            label: float(prob)
            for label, prob in zip(label_encoder.classes_, prediction_proba)
        }
        
        # Generate recommendations
        recommendations = recommendation_engine.generate_recommendations(
            prediction=predicted_label,
            prediction_proba=prob_dict,
            age=age
        )
        
        # Prepare response
        response = {
            'student_id': student_id,
            'timestamp': datetime.now().isoformat(),
            'prediction': {
                'condition': predicted_label,
                'confidence': float(max(prediction_proba)),
                'probabilities': prob_dict
            },
            'recommendations': recommendations
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Batch prediction endpoint
    
    Expected JSON:
    {
        "samples": [
            {
                "handwriting_features": {...},
                "speech_features": {...},
                "age": 8,
                "student_id": "STU001"
            },
            ...
        ]
    }
    """
    try:
        data = request.get_json()
        samples = data.get('samples', [])
        
        if not samples:
            return jsonify({'error': 'No samples provided'}), 400
        
        results = []
        
        for sample in samples:
            hw_features = sample.get('handwriting_features', {})
            speech_features = sample.get('speech_features', {})
            age = sample.get('age', 8)
            student_id = sample.get('student_id', 'UNKNOWN')
            
            feature_dict = {**hw_features, **speech_features, 'age': age}
            input_df = pd.DataFrame([feature_dict])
            
            X = engineer.prepare_features(input_df, is_training=False)
            prediction = predictor.predict(X)[0]
            prediction_proba = predictor.predict_proba(X)[0]
            predicted_label = label_encoder.inverse_transform([prediction])[0]
            
            prob_dict = {
                label: float(prob)
                for label, prob in zip(label_encoder.classes_, prediction_proba)
            }
            
            recommendations = recommendation_engine.generate_recommendations(
                prediction=predicted_label,
                prediction_proba=prob_dict,
                age=age
            )
            
            results.append({
                'student_id': student_id,
                'prediction': {
                    'condition': predicted_label,
                    'confidence': float(max(prediction_proba)),
                    'probabilities': prob_dict
                },
                'recommendations': recommendations
            })
        
        return jsonify({
            'timestamp': datetime.now().isoformat(),
            'total_samples': len(results),
            'results': results
        }), 200
        
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/progress', methods=['POST'])
def track_progress():
    """
    Progress tracking endpoint
    
    Expected JSON:
    {
        "student_id": "STU001",
        "assessments": [
            {
                "date": "2025-09-01",
                "confidence": 0.85,
                "condition": "dyslexia"
            },
            ...
        ]
    }
    """
    try:
        data = request.get_json()
        student_id = data.get('student_id')
        assessments = data.get('assessments', [])
        
        if not student_id or not assessments:
            return jsonify({'error': 'student_id and assessments required'}), 400
        
        progress_report = recommendation_engine.generate_progress_report(
            student_id=student_id,
            assessments=assessments
        )
        
        return jsonify(progress_report), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/interventions/<condition>', methods=['GET'])
def get_interventions(condition):
    """
    Get available interventions for a specific condition
    
    Query params:
        - age: Student age (optional)
        - severity: mild/moderate/severe (optional)
    """
    try:
        age = request.args.get('age', type=int, default=8)
        severity = request.args.get('severity', default='moderate')
        
        if condition not in ['dyslexia', 'dysgraphia', 'normal']:
            return jsonify({'error': 'Invalid condition'}), 400
        
        condition_data = recommendation_engine.intervention_database.get(condition, {})
        interventions = condition_data.get('interventions', [])
        
        # Filter by severity and age
        filtered = [
            i for i in interventions
            if severity in i.get('severity', [])
        ]
        age_appropriate = recommendation_engine.get_age_appropriate_interventions(
            filtered, age
        )
        
        return jsonify({
            'condition': condition,
            'age': age,
            'severity': severity,
            'interventions': age_appropriate,
            'accommodations': condition_data.get('classroom_accommodations', []),
            'home_strategies': condition_data.get('home_strategies', [])
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/export/report', methods=['POST'])
def export_report():
    """
    Export recommendation report in text format
    
    Expected JSON:
    {
        "prediction": "dyslexia",
        "prediction_proba": {...},
        "age": 8
    }
    """
    try:
        data = request.get_json()
        
        recommendations = recommendation_engine.generate_recommendations(
            prediction=data['prediction'],
            prediction_proba=data['prediction_proba'],
            age=data.get('age', 8)
        )
        
        text_report = recommendation_engine.export_recommendations(
            recommendations, 
            format='text'
        )
        
        return jsonify({
            'report': text_report,
            'format': 'text'
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("LEARNING DISABILITY DETECTION API")
    print("="*60)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Models Directory: {MODELS_DIR}")
    print("="*60 + "\n")
    
    if load_models():
        print("\n✓ Starting Flask API server...")
        print("Available endpoints:")
        print("  • GET  /health")
        print("  • POST /predict")
        print("  • POST /predict/batch")
        print("  • POST /progress")
        print("  • GET  /interventions/<condition>")
        print("  • POST /export/report")
        print("\nServer running on http://localhost:5000\n")
        
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("\n✗ Failed to load models. Please train models first using:")
        print("  python train_models.py\n")
