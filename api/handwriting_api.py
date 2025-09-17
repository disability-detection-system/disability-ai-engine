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

@app.route('/analyze/handwriting', methods=['POST', 'OPTIONS'])
def analyze_handwriting():
    """API endpoint for handwriting analysis"""
    # Handle preflight request
    if request.method == 'OPTIONS':
        return '', 200
        
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
            
            # *** STEP 2 IMPLEMENTATION: Use new enhanced scoring ***
            overall_score = analyzer.calculate_overall_score(features)
            
            # Convert to dictionary for JSON response
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
                # *** OLD SCORING (REMOVED) ***
                # 'overall_score': (
                #     features.line_straightness + 
                #     features.letter_formation_quality + 
                #     features.consistency_score
                # ) / 3,
                
                # *** NEW ENHANCED SCORING ***
                'overall_score': round(overall_score, 1),
                
                # Additional scoring details for better feedback
                'score_breakdown': {
                    'line_quality': round(features.line_straightness, 1),
                    'consistency': round(features.consistency_score, 1),
                    'letter_formation': round(features.letter_formation_quality, 1),
                'size_appropriateness': round(_score_letter_size(features.avg_letter_size), 1),
                'spacing_quality': round(_score_spacing(features.letter_spacing), 1),
                'pressure_control': round(_score_pressure(features.writing_pressure), 1),
                'slant_control': round(_score_slant(features.slant_angle), 1)
            },
            
            # Quality assessment
            'quality_level': _get_quality_level(overall_score),
            'recommendations': _get_recommendations(features, overall_score),
                
                'status': 'success'
            }
            
            return jsonify(result)
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    except Exception as e:
        print(f"Error in analysis: {str(e)}")  # Add logging
        return jsonify({'error': str(e), 'status': 'error'}), 500

def _score_letter_size(avg_letter_size):
    """Helper function to score letter size (same logic as in analyzer)"""
    if 500 <= avg_letter_size <= 2000:
        return 100
    elif avg_letter_size < 500 and avg_letter_size > 0:
        return (avg_letter_size / 500) * 100
    elif avg_letter_size > 2000:
        return max(0, 100 - ((avg_letter_size - 2000) / 10))
    else:
        return 0

def _score_spacing(letter_spacing):
    """Helper function to score spacing"""
    if 10 <= letter_spacing <= 50:
        return 100
    else:
        return max(0, 100 - abs(letter_spacing - 30) * 2)

def _score_pressure(writing_pressure):
    """Helper function to score pressure"""
    if 20 <= writing_pressure <= 80:
        return 100
    else:
        return max(0, 100 - abs(writing_pressure - 50))

def _score_slant(slant_angle):
    """Helper function to score slant"""
    return max(0, 100 - abs(slant_angle) * 2)

def _get_quality_level(overall_score):
    """Determine quality level based on score"""
    if overall_score >= 85:
        return "Excellent"
    elif overall_score >= 70:
        return "Good"
    elif overall_score >= 55:
        return "Fair"
    elif overall_score >= 40:
        return "Needs Improvement"
    else:
        return "Poor"

def _get_recommendations(features, overall_score):
    """Generate specific recommendations based on analysis"""
    recommendations = []
    
    # Line straightness recommendations
    if features.line_straightness < 60:
        recommendations.append("Practice writing on lined paper to improve line straightness")
    
    # Consistency recommendations
    if features.consistency_score < 50:
        recommendations.append("Focus on maintaining consistent letter sizes and spacing")
    
    # Formation quality recommendations
    if features.letter_formation_quality < 40:
        recommendations.append("Practice letter formation exercises to improve shape consistency")
    
    # Size recommendations
    if features.avg_letter_size < 500:
        recommendations.append("Try writing slightly larger letters for better readability")
    elif features.avg_letter_size > 2000:
        recommendations.append("Practice writing smaller, more controlled letters")
    
    # Spacing recommendations
    if features.letter_spacing < 10:
        recommendations.append("Increase spacing between letters for better readability")
    elif features.letter_spacing > 50:
        recommendations.append("Reduce spacing between letters to improve flow")
    
    # Pressure recommendations
    if features.writing_pressure < 20:
        recommendations.append("Apply slightly more pressure when writing")
    elif features.writing_pressure > 80:
        recommendations.append("Use lighter pressure to avoid hand fatigue")
    
    # Slant recommendations
    if abs(features.slant_angle) > 20:
        recommendations.append("Practice keeping letters more upright and consistent")
    
    # General recommendations based on overall score
    if overall_score < 40:
        recommendations.append("Consider practicing basic handwriting exercises daily")
    elif overall_score < 70:
        recommendations.append("Good foundation - focus on consistency and formation")
    
    return recommendations[:3]  # Return top 3 recommendations

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'handwriting_analyzer'})

if __name__ == '__main__':
    # Make sure to run on all interfaces and the correct port
    app.run(debug=True, host='0.0.0.0', port=5001)
