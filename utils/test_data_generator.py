import cv2
import numpy as np
import os

def create_sample_handwriting_images():
    """Create simple sample handwriting images for testing"""
    
    # Create samples directory
    os.makedirs('samples/handwriting', exist_ok=True)
    
    # Sample 1: Good handwriting simulation
    img1 = np.ones((300, 800, 3), dtype=np.uint8) * 255
    
    # Add some text-like strokes
    cv2.line(img1, (50, 100), (150, 100), (0, 0, 0), 3)  # 'I'
    cv2.line(img1, (180, 80), (280, 120), (0, 0, 0), 3)   # slanted line
    cv2.circle(img1, (320, 100), 20, (0, 0, 0), 3)        # 'O'
    cv2.line(img1, (370, 80), (420, 120), (0, 0, 0), 3)   # 'V' part 1
    cv2.line(img1, (420, 120), (470, 80), (0, 0, 0), 3)   # 'V' part 2
    
    cv2.imwrite('samples/handwriting/sample_good.png', img1)
    
    # Sample 2: Poor handwriting simulation (irregular)
    img2 = np.ones((300, 800, 3), dtype=np.uint8) * 255
    
    # Add irregular strokes
    cv2.line(img2, (50, 90), (140, 110), (0, 0, 0), 4)   # wavy line
    cv2.line(img2, (180, 70), (270, 130), (0, 0, 0), 5)  # thick irregular
    cv2.ellipse(img2, (320, 100), (25, 15), 45, 0, 360, (0, 0, 0), 4)
    
    cv2.imwrite('samples/handwriting/sample_poor.png', img2)
    
    print("Sample handwriting images created in samples/handwriting/")

if __name__ == "__main__":
    create_sample_handwriting_images()
