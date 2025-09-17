from flask import Flask, request, jsonify
import os
from flask_cors import CORS
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlp.speech_analyzer import SpeechAnalyzer
import tempfile
import uuid
from werkzeug.utils import secure_filename
import subprocess
from pydub import AudioSegment
import numpy as np

# Add FFmpeg to PATH for this process
ffmpeg_path = r"C:\Users\SIDDHESH\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0-full_build\bin"
if ffmpeg_path not in os.environ.get('PATH', ''):
    os.environ['PATH'] = ffmpeg_path + os.pathsep + os.environ.get('PATH', '')

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])
analyzer = SpeechAnalyzer()

# Updated to include webm format
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'm4a', 'webm'}

def to_serializable(val):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(val, np.generic):
        return val.item()
    return val

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_webm_to_wav(webm_path: str, wav_path: str) -> bool:
    """Convert WebM audio to WAV format using ffmpeg"""
    try:
        # Try different ways to find ffmpeg
        ffmpeg_cmd = None
        possible_paths = [
            'ffmpeg',  # In PATH
            r'C:\Users\SIDDHESH\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0-full_build\bin\ffmpeg.exe',  # WinGet installation
            'C:\\ffmpeg\\bin\\ffmpeg.exe',  # Common Windows installation
            'C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe',
            'C:\\Program Files (x86)\\ffmpeg\\bin\\ffmpeg.exe'
        ]
        
        for path in possible_paths:
            try:
                subprocess.run([path, '-version'], 
                              stdout=subprocess.DEVNULL, 
                              stderr=subprocess.DEVNULL, 
                              check=True)
                ffmpeg_cmd = path
                print(f"Found FFmpeg at: {path}")
                break
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue
        
        if not ffmpeg_cmd:
            print("FFmpeg not found in any common locations")
            return False
        
        # Convert WebM to WAV
        cmd = [
            ffmpeg_cmd, '-i', webm_path, 
            '-acodec', 'pcm_s16le',
            '-ar', '16000',  # 16kHz sample rate for speech recognition
            '-ac', '1',      # Mono channel
            '-y',            # Overwrite output file
            wav_path
        ]
        
        print(f"Converting WebM to WAV: {webm_path} -> {wav_path}")
        result = subprocess.run(cmd, 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE, 
                              check=True)
        
        # Check if output file was created
        if os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
            print(f"Successfully converted WebM to WAV")
            return True
        else:
            print(f"Conversion failed - output file not created or empty")
            return False
        
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg conversion failed: {e}")
        print(f"FFmpeg stderr: {e.stderr.decode() if e.stderr else 'No error details'}")
        return False
    except FileNotFoundError:
        print("FFmpeg not found in PATH")
        return False
    except Exception as e:
        print(f"Unexpected error during conversion: {e}")
        return False

def convert_webm_to_wav_pydub(webm_path: str, wav_path: str) -> bool:
    """Convert WebM audio to WAV format using pydub as fallback"""
    try:
        print(f"Trying pydub conversion: {webm_path} -> {wav_path}")
        # Load WebM file
        audio = AudioSegment.from_file(webm_path, format="webm")
        
        # Convert to mono and 16kHz
        audio = audio.set_channels(1)  # Mono
        audio = audio.set_frame_rate(16000)  # 16kHz
        
        # Export as WAV
        audio.export(wav_path, format="wav")
        
        # Check if output file was created
        if os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
            print(f"Successfully converted WebM to WAV using pydub")
            return True
        else:
            print(f"Pydub conversion failed - output file not created or empty")
            return False
            
    except Exception as e:
        print(f"Pydub conversion failed: {e}")
        return False

@app.route('/analyze/speech', methods=['POST'])
def analyze_speech():
    """Enhanced API endpoint for speech analysis with WebM support"""
    try:
        # Check if audio file was uploaded
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        file = request.files['audio']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Get file extension
        filename = secure_filename(file.filename) if file.filename else 'recording.webm'
        file_ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else 'webm'
        
        if file_ext not in ALLOWED_EXTENSIONS:
            return jsonify({'error': f'Unsupported file format. Use WAV, MP3, OGG, M4A, or WebM'}), 400
        
        # Get optional reference text
        reference_text = request.form.get('reference_text', '')
        
        # Save uploaded file temporarily
        temp_filename = str(uuid.uuid4()) + '.' + file_ext
        temp_path = os.path.join(tempfile.gettempdir(), temp_filename)
        file.save(temp_path)
        
        # Check if file was saved and is not empty
        if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
            return jsonify({'error': 'Uploaded file is empty or corrupted'}), 400
        
        try:
            # Convert WebM to WAV if needed
            analysis_path = temp_path
            converted_path = None
            
            if file_ext == 'webm':
                converted_filename = str(uuid.uuid4()) + '.wav'
                converted_path = os.path.join(tempfile.gettempdir(), converted_filename)
                
                # Try ffmpeg first
                if convert_webm_to_wav(temp_path, converted_path):
                    analysis_path = converted_path
                    print(f"Successfully converted WebM to WAV using ffmpeg: {converted_path}")
                else:
                    # Try pydub as fallback
                    print("FFmpeg conversion failed, trying pydub...")
                    if convert_webm_to_wav_pydub(temp_path, converted_path):
                        analysis_path = converted_path
                        print(f"Successfully converted WebM to WAV using pydub: {converted_path}")
                    else:
                        # Final fallback: try to analyze WebM directly
                        print("Warning: Could not convert WebM to WAV with either method. Trying direct analysis...")
                        print("Note: This may not work with all WebM files")
            
            # Analyze speech
            print(f"Starting speech analysis for: {analysis_path}")
            features = analyzer.analyze_speech_file(analysis_path, reference_text)
            print(f"Speech analysis completed successfully")
            
            # Convert to dictionary for JSON response
            result = {
                'analysis_id': str(uuid.uuid4()),
                'features': {k:to_serializable(v) for k,v in{
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
            
            return jsonify(result)
            
        finally:
            # Clean up temporary files
            if os.path.exists(temp_path):
                os.remove(temp_path)
            if converted_path and os.path.exists(converted_path):
                os.remove(converted_path)
    
    except Exception as e:
        print(f"Error in speech analysis API: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'speech_analyzer'})


if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0', port=5002)
