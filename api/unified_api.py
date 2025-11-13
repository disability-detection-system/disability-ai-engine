"""
Unified API Server - Fixed version with proper audio conversion
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import tempfile
import uuid
from werkzeug.utils import secure_filename
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
import subprocess

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from cv.handwriting_analyzer import HandwritingAnalyzer
from nlp.speech_analyzer import SpeechAnalyzer
from ml.disability_predictor import DisabilityPredictor
from ml.feature_engineering import FeatureEngineer
from ml.recommendation_engine import RecommendationEngine

# Initialize Flask app
app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])

# Initialize analyzers and models
handwriting_analyzer = HandwritingAnalyzer()
speech_analyzer = SpeechAnalyzer()
predictor = None
engineer = None
label_encoder = None
recommendation_engine = None

MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'm4a', 'webm'}

# FFmpeg configuration (adjust path for your system)
FFMPEG_PATHS = [
    'ffmpeg',  # In PATH
    r'C:\Users\SIDDHESH\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0-full_build\bin\ffmpeg.exe',
    'C:\\ffmpeg\\bin\\ffmpeg.exe',
    'C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe',
]

def find_ffmpeg():
    """Find FFmpeg executable"""
    for path in FFMPEG_PATHS:
        try:
            subprocess.run([path, '-version'],
                         stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL,
                         check=True)
            print(f"✓ Found FFmpeg at: {path}")
            return path
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    print("✗ FFmpeg not found in common locations")
    return None

def convert_webm_to_wav(webm_path, wav_path):
    """Convert WebM to WAV format using FFmpeg"""
    ffmpeg_cmd = find_ffmpeg()
    
    if not ffmpeg_cmd:
        raise Exception("FFmpeg not found. Please install FFmpeg.")
    
    try:
        cmd = [
            ffmpeg_cmd, '-i', webm_path,
            '-acodec', 'pcm_s16le',  # PCM WAV format
            '-ar', '16000',           # 16kHz sample rate
            '-ac', '1',                # Mono channel
            '-y',                      # Overwrite output
            wav_path
        ]
        
        print(f"Converting WebM to WAV: {webm_path} -> {wav_path}")
        result = subprocess.run(cmd, 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE,
                              check=True)
        
        if os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
            print(f"✓ Conversion successful: {os.path.getsize(wav_path)} bytes")
            return True
        else:
            raise Exception("Conversion failed - output file empty")
            
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode() if e.stderr else 'Unknown error'
        raise Exception(f"FFmpeg conversion failed: {error_msg}")
    except Exception as e:
        raise Exception(f"Audio conversion error: {str(e)}")

def load_models():
    """Load trained ML models"""
    global predictor, engineer, label_encoder, recommendation_engine
    try:
        print(f"\n{'='*60}")
        print("Loading ML Models...")
        print(f"{'='*60}")
        
        # Check if models directory exists
        if not os.path.exists(MODELS_DIR):
            print(f"✗ Models directory not found: {MODELS_DIR}")
            print("\nPlease train models first:")
            print("  1. python ml/data_generator.py")
            print("  2. python ml/train_models.py")
            return False
        
        # Load feature engineer
        engineer = FeatureEngineer()
        scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
        if os.path.exists(scaler_path):
            engineer.load_preprocessors(MODELS_DIR)
            print("✓ Feature engineer loaded")
        else:
            print("✗ Scaler not found - models need to be trained")
            return False
        
        # Load label encoder
        label_encoder_path = os.path.join(MODELS_DIR, 'label_encoder.pkl')
        if os.path.exists(label_encoder_path):
            label_encoder = joblib.load(label_encoder_path)
            print(f"✓ Label encoder loaded with classes: {list(label_encoder.classes_)}")
        else:
            print("✗ Label encoder not found")
            return False
        
        # Find best model
        best_model_path = os.path.join(MODELS_DIR, 'best_model.txt')
        if os.path.exists(best_model_path):
            with open(best_model_path, 'r') as f:
                lines = f.readlines()
                if len(lines) >= 2:
                    model_file = lines[1].split(': ')[1].strip()
                else:
                    raise Exception("Invalid best_model.txt format")
        else:
            # Find any available model
            model_files = [f for f in os.listdir(MODELS_DIR) 
                          if f.endswith('.pkl') and 'model' in f.lower() 
                          and 'scaler' not in f and 'encoder' not in f]
            if model_files:
                model_file = model_files[0]
                print(f"⚠ Using fallback model: {model_file}")
            else:
                print("✗ No trained models found")
                return False
        
        # Load predictor
        model_path = os.path.join(MODELS_DIR, model_file)
        if not os.path.exists(model_path):
            model_path = model_file  # Try as absolute path
        
        predictor = DisabilityPredictor()
        predictor.load_model(model_path)
        print(f"✓ Predictor loaded from: {model_file}")
        
        # Initialize recommendation engine
        recommendation_engine = RecommendationEngine()
        print("✓ Recommendation engine initialized")
        
        print(f"{'='*60}")
        print("✓ All models loaded successfully!")
        print(f"{'='*60}\n")
        return True
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"✗ Error loading models: {str(e)}")
        print(f"{'='*60}\n")
        import traceback
        traceback.print_exc()
        return False

# ========== HANDWRITING ENDPOINTS ==========

@app.route('/analyze/handwriting', methods=['POST', 'OPTIONS'])
def analyze_handwriting():
    """Handwriting analysis endpoint"""
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        temp_filename = str(uuid.uuid4()) + '.png'
        temp_path = os.path.join(tempfile.gettempdir(), temp_filename)
        file.save(temp_path)
        
        try:
            features = handwriting_analyzer.extract_features(temp_path)
            overall_score = handwriting_analyzer.calculate_overall_score(features)
            
            result = {
                'analysis_id': str(uuid.uuid4()),
                'features': {
                    'avg_letter_size': round(features.avg_letter_size, 2),
                    'line_straightness': round(features.line_straightness, 2),
                    'letter_spacing': round(features.letter_spacing, 2),
                    'word_spacing': round(features.word_spacing, 2),
                    'writing_pressure': round(features.writing_pressure, 2),
                    'letter_formation_quality': round(features.letter_formation_quality, 2),
                    'slant_angle': round(features.slant_angle, 2),
                    'consistency_score': round(features.consistency_score, 2),
                    'contour_count': features.contour_count,
                    'aspect_ratio': round(features.aspect_ratio, 2)
                },
                'overall_score': round(overall_score, 1),
                'status': 'success'
            }
            
            print(f"✓ Handwriting analysis completed: score={overall_score:.1f}")
            return jsonify(result)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        print(f"✗ Handwriting analysis error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'status': 'error'}), 500

# ========== SPEECH ENDPOINTS ==========

def to_serializable(val):
    """Convert numpy types to native Python types"""
    if isinstance(val, np.generic):
        return val.item()
    return val

@app.route('/analyze/speech', methods=['POST', 'OPTIONS'])
def analyze_speech():
    """Speech analysis endpoint with WebM conversion"""
    if request.method == 'OPTIONS':
        return '', 200
    
    temp_path = None
    converted_path = None
    
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        file = request.files['audio']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        filename = secure_filename(file.filename) if file.filename else 'recording.webm'
        file_ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else 'webm'
        
        if file_ext not in ALLOWED_EXTENSIONS:
            return jsonify({'error': f'Unsupported file format: {file_ext}'}), 400
        
        reference_text = request.form.get('referenceText', '')
        
        # Save uploaded file
        temp_filename = str(uuid.uuid4()) + '.' + file_ext
        temp_path = os.path.join(tempfile.gettempdir(), temp_filename)
        file.save(temp_path)
        
        print(f"Received audio file: {filename} ({os.path.getsize(temp_path)} bytes)")
        
        # Convert WebM to WAV if needed
        analysis_path = temp_path
        if file_ext == 'webm':
            converted_filename = str(uuid.uuid4()) + '.wav'
            converted_path = os.path.join(tempfile.gettempdir(), converted_filename)
            
            try:
                convert_webm_to_wav(temp_path, converted_path)
                analysis_path = converted_path
                print(f"✓ Using converted WAV file for analysis")
            except Exception as conv_error:
                print(f"✗ Conversion failed: {str(conv_error)}")
                return jsonify({
                    'error': f'Audio conversion failed: {str(conv_error)}',
                    'hint': 'Please ensure FFmpeg is installed'
                }), 500
        
        # Analyze speech
        print(f"Analyzing speech from: {analysis_path}")
        features = speech_analyzer.analyze_speech_file(analysis_path, reference_text)
        
        result = {
            'analysis_id': str(uuid.uuid4()),
            'features': {k: to_serializable(v) for k, v in {
                'transcript': features.transcript,
                'reading_speed_wpm': round(features.reading_speed_wpm, 2),
                'pause_frequency': round(features.pause_frequency, 2),
                'average_pause_duration': round(features.average_pause_duration, 2),
                'pronunciation_score': round(features.pronunciation_score, 2),
                'fluency_score': round(features.fluency_score, 2),
                'volume_consistency': round(features.volume_consistency, 2),
                'pitch_variation': round(features.pitch_variation, 2),
                'speech_clarity': round(features.speech_clarity, 2),
                'confidence_score': round(features.confidence_score, 2),
                'total_duration': round(features.total_duration, 2),
                'word_count': features.word_count
            }.items()},
            'overall_score': round(features.confidence_score, 2),
            'status': 'success'
        }
        
        print(f"✓ Speech analysis completed: confidence={features.confidence_score:.1f}")
        return jsonify(result)
        
    except Exception as e:
        print(f"✗ Speech analysis error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'status': 'error'}), 500
        
    finally:
        # Clean up temporary files
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        if converted_path and os.path.exists(converted_path):
            os.remove(converted_path)

# ========== PREDICTION ENDPOINTS ==========

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    """Final prediction endpoint with recommendations"""
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        if predictor is None:
            return jsonify({
                'error': 'ML models not loaded. Please train models first.',
                'hint': 'Run: python ml/train_models.py'
            }), 503
        
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        hw_features = data.get('handwriting_features', {})
        speech_features = data.get('speech_features', {})
        age = data.get('age', 8)
        student_id = data.get('student_id', 'UNKNOWN')
        
        print(f"\nReceived prediction request for student: {student_id}")
        print(f"  Handwriting features: {len(hw_features)} keys")
        print(f"  Speech features: {len(speech_features)} keys")
        print(f"  Age: {age}")
        
        if not hw_features and not speech_features:
            return jsonify({'error': 'At least handwriting or speech features required'}), 400
        
        # Combine features
        feature_dict = {**hw_features, **speech_features, 'age': age}
        input_df = pd.DataFrame([feature_dict])
        
        print(f"Combined features: {list(feature_dict.keys())}")
        
        # Preprocess
        X = engineer.prepare_features(input_df, is_training=False)
        print(f"Preprocessed shape: {X.shape}")
        
        # Predict
        prediction = predictor.predict(X)[0]
        prediction_proba = predictor.predict_proba(X)[0]
        
        # Decode label
        predicted_label = label_encoder.inverse_transform([prediction])[0]
        
        print(f"✓ Prediction: {predicted_label} (confidence: {max(prediction_proba):.2%})")
        
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
        print(f"✗ Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

# ========== HEALTH CHECK ==========

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'services': {
            'handwriting': True,
            'speech': True,
            'prediction': predictor is not None,
            'ffmpeg': find_ffmpeg() is not None
        },
        'models_loaded': predictor is not None,
        'timestamp': datetime.now().isoformat()
    })

# ========== MAIN ==========

if __name__ == '__main__':
    print("\n" + "="*60)
    print("UNIFIED LEARNING DISABILITY DETECTION API")
    print("="*60)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Models Directory: {MODELS_DIR}")
    print("="*60)
    
    # Check FFmpeg
    if find_ffmpeg():
        print("✓ FFmpeg available for audio conversion")
    else:
        print("⚠ FFmpeg not found - WebM conversion may fail")
    
    print("="*60)
    
    # Load models
    models_loaded = load_models()
    
    if not models_loaded:
        print("\n⚠ WARNING: Starting API without ML models")
        print("Handwriting and speech analysis will work, but predictions will fail")
        print("\nTo fix this, run:")
        print("  1. cd backend")
        print("  2. python ml/data_generator.py")
        print("  3. python ml/train_models.py")
        print("  4. Restart this API")
    
    print("\n" + "="*60)
    print("Starting Flask API server...")
    print("\nAvailable endpoints:")
    print("  POST /analyze/handwriting")
    print("  POST /analyze/speech")
    print("  POST /predict")
    print("  GET  /health")
    print("\nRunning on http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
