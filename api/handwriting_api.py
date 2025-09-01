from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cv.handwriting_analyzer import HandwritingAnalyzer
import tempfile
import uuid

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])
analyzer = HandwritingAnalyzer()

@app.route('/analyze/handwriting', methods=['POST'])
def analyze_handwriting():
    """API endpoint for handwriting analysis"""
    try:
        # Check if file was uploaded
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save uploaded file temporarily
        temp_filename = str(uuid.uuid4()) + '.png'
        temp_path = os.path.join(tempfile.gettempdir(), temp_filename)
        file.save(temp_path)
        
        try:
            # Analyze handwriting
            features = analyzer.extract_features(temp_path)
            
            # Convert to dictionary for JSON response
            result = {
                'analysis_id': str(uuid.uuid4()),
                'features': {
                    'avg_letter_size': features.avg_letter_size,
                    'line_straightness': features.line_straightness,
                    'letter_spacing': features.letter_spacing,
                    'word_spacing': features.word_spacing,
                    'writing_pressure': features.writing_pressure,
                    'letter_formation_quality': features.letter_formation_quality,
                    'slant_angle': features.slant_angle,
                    'consistency_score': features.consistency_score,
                    'contour_count': features.contour_count,
                    'aspect_ratio': features.aspect_ratio
                },
                'overall_score': (
                    features.line_straightness + 
                    features.letter_formation_quality + 
                    features.consistency_score
                ) / 3,
                'status': 'success'
            }
            
            return jsonify(result)
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'handwriting_analyzer'})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
