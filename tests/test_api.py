"""
API Testing Script
Tests all endpoints of the prediction API
"""

import requests
import json
from datetime import datetime

BASE_URL = "http://localhost:5000"

def test_health_check():
    """Test health check endpoint"""
    print("\n" + "="*60)
    print("TEST 1: Health Check")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    assert response.json()['status'] == 'healthy'
    print("✓ Health check passed")


def test_single_prediction():
    """Test single prediction endpoint"""
    print("\n" + "="*60)
    print("TEST 2: Single Prediction")
    print("="*60)
    
    # Sample data (simulating dyslexia)
    data = {
        "student_id": "STU001",
        "age": 8,
        "handwriting_features": {
            "avg_letter_size": 1200,
            "line_straightness": 55,
            "letter_spacing": 28,
            "word_spacing": 56,
            "writing_pressure": 58,
            "letter_formation_quality": 52,
            "slant_angle": 12,
            "consistency_score": 48,
            "contour_count": 65,
            "aspect_ratio": 1.8
        },
        "speech_features": {
            "reading_speed_wpm": 45,
            "pause_frequency": 2.5,
            "average_pause_duration": 1.2,
            "pronunciation_score": 45,
            "fluency_score": 40,
            "volume_consistency": 55,
            "pitch_variation": 48,
            "speech_clarity": 42,
            "word_count": 35,
            "total_duration": 55
        }
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=data)
    print(f"Status Code: {response.status_code}")
    
    # Print full response to see the error
    result = response.json()
    print(f"\nFull Response:")
    print(json.dumps(result, indent=2))
    
    # Check if there's an error
    if response.status_code != 200:
        print("\n✗ API Error occurred!")
        if 'error' in result:
            print(f"Error: {result['error']}")
        if 'traceback' in result:
            print(f"Traceback:\n{result['traceback']}")
        return None
    
    print(f"\nPrediction: {result['prediction']['condition']}")
    print(f"Confidence: {result['prediction']['confidence']*100:.1f}%")
    print(f"Severity: {result['recommendations']['severity_level']}")
    print(f"\nTop 3 Interventions:")
    for i, intervention in enumerate(result['recommendations']['primary_interventions'][:3], 1):
        print(f"{i}. {intervention['name']}")
    
    assert response.status_code == 200
    print("\n✓ Single prediction test passed")
    
    return result



def test_batch_prediction():
    """Test batch prediction endpoint"""
    print("\n" + "="*60)
    print("TEST 3: Batch Prediction")
    print("="*60)
    
    data = {
        "samples": [
            {
                "student_id": "STU001",
                "age": 8,
                "handwriting_features": {
                    "avg_letter_size": 1200,
                    "line_straightness": 55,
                    "letter_spacing": 28,
                    "word_spacing": 56,
                    "writing_pressure": 58,
                    "letter_formation_quality": 52,
                    "slant_angle": 12,
                    "consistency_score": 48,
                    "contour_count": 65,
                    "aspect_ratio": 1.8
                },
                "speech_features": {
                    "reading_speed_wpm": 45,
                    "pause_frequency": 2.5,
                    "average_pause_duration": 1.2,
                    "pronunciation_score": 45,
                    "fluency_score": 40,
                    "volume_consistency": 55,
                    "pitch_variation": 48,
                    "speech_clarity": 42,
                    "word_count": 35,
                    "total_duration": 55
                }
            },
            {
                "student_id": "STU002",
                "age": 7,
                "handwriting_features": {
                    "avg_letter_size": 1800,
                    "line_straightness": 35,
                    "letter_spacing": 45,
                    "word_spacing": 90,
                    "writing_pressure": 35,
                    "letter_formation_quality": 30,
                    "slant_angle": 25,
                    "consistency_score": 25,
                    "contour_count": 45,
                    "aspect_ratio": 2.5
                },
                "speech_features": {
                    "reading_speed_wpm": 78,
                    "pause_frequency": 0.8,
                    "average_pause_duration": 0.4,
                    "pronunciation_score": 72,
                    "fluency_score": 70,
                    "volume_consistency": 68,
                    "pitch_variation": 72,
                    "speech_clarity": 70,
                    "word_count": 65,
                    "total_duration": 35
                }
            }
        ]
    }
    
    response = requests.post(f"{BASE_URL}/predict/batch", json=data)
    print(f"Status Code: {response.status_code}")
    result = response.json()
    
    print(f"\nTotal Samples: {result['total_samples']}")
    for i, res in enumerate(result['results'], 1):
        print(f"\nStudent {i} ({res['student_id']}):")
        print(f"  Prediction: {res['prediction']['condition']}")
        print(f"  Confidence: {res['prediction']['confidence']*100:.1f}%")
    
    assert response.status_code == 200
    print("\n✓ Batch prediction test passed")


def test_progress_tracking():
    """Test progress tracking endpoint"""
    print("\n" + "="*60)
    print("TEST 4: Progress Tracking")
    print("="*60)
    
    data = {
        "student_id": "STU001",
        "assessments": [
            {"date": "2025-09-01", "confidence": 0.85, "condition": "dyslexia"},
            {"date": "2025-10-01", "confidence": 0.78, "condition": "dyslexia"},
            {"date": "2025-11-01", "confidence": 0.65, "condition": "dyslexia"}
        ]
    }
    
    response = requests.post(f"{BASE_URL}/progress", json=data)
    print(f"Status Code: {response.status_code}")
    result = response.json()
    
    print(f"\nStudent ID: {result['student_id']}")
    print(f"Total Assessments: {result['total_assessments']}")
    print(f"Trend: {result['trend']}")
    print(f"Improvement Rate: {result['improvement_rate']:.3f}")
    
    assert response.status_code == 200
    print("\n✓ Progress tracking test passed")


def test_get_interventions():
    """Test get interventions endpoint"""
    print("\n" + "="*60)
    print("TEST 5: Get Interventions")
    print("="*60)
    
    response = requests.get(
        f"{BASE_URL}/interventions/dyslexia",
        params={"age": 8, "severity": "moderate"}
    )
    print(f"Status Code: {response.status_code}")
    result = response.json()
    
    print(f"\nCondition: {result['condition']}")
    print(f"Age: {result['age']}")
    print(f"Severity: {result['severity']}")
    print(f"Available Interventions: {len(result['interventions'])}")
    
    assert response.status_code == 200
    print("\n✓ Get interventions test passed")


def test_export_report():
    """Test export report endpoint"""
    print("\n" + "="*60)
    print("TEST 6: Export Report")
    print("="*60)
    
    data = {
        "prediction": "dyslexia",
        "prediction_proba": {
            "normal": 0.10,
            "dyslexia": 0.75,
            "dysgraphia": 0.15
        },
        "age": 8
    }
    
    response = requests.post(f"{BASE_URL}/export/report", json=data)
    print(f"Status Code: {response.status_code}")
    result = response.json()
    
    print(f"\nReport Preview:")
    print(result['report'][:500])
    print("...")
    
    assert response.status_code == 200
    print("\n✓ Export report test passed")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("API TESTING SUITE")
    print("="*80)
    print(f"Base URL: {BASE_URL}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    try:
        test_health_check()
        test_single_prediction()
        test_batch_prediction()
        test_progress_tracking()
        test_get_interventions()
        test_export_report()
        
        print("\n" + "="*80)
        print("✓ ALL TESTS PASSED")
        print("="*80 + "\n")
        
    except requests.exceptions.ConnectionError:
        print("\n✗ ERROR: Cannot connect to API server")
        print("Make sure the API is running: python api/prediction_endpoint.py\n")
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}\n")
        import traceback
        traceback.print_exc()

