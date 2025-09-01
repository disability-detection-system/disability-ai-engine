import cv2
import matplotlib.pyplot as plt
import os
import sys

# Add the parent directory to the path so we can import from cv module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cv.handwriting_analyzer import HandwritingAnalyzer

def verify_preprocessing():
    """Verify that image preprocessing is working correctly"""
    print("=== Day 1 Verification: Image Preprocessing ===\n")
    
    analyzer = HandwritingAnalyzer()
    sample_files = [
        'samples/handwriting/sample_good.png',
        'samples/handwriting/sample_poor.png'
    ]
    
    verification_passed = True
    
    for i, sample_file in enumerate(sample_files, 1):
        print(f"Test {i}: Processing {sample_file}")
        
        # Check if file exists
        if not os.path.exists(sample_file):
            print(f"❌ FAIL: File {sample_file} does not exist!")
            verification_passed = False
            continue
        
        try:
            # Test image loading
            original = cv2.imread(sample_file)
            if original is None:
                print(f"❌ FAIL: Could not load {sample_file}")
                verification_passed = False
                continue
            
            print(f"✅ Original image loaded: {original.shape}")
            
            # Test preprocessing
            processed = analyzer.preprocess_image(sample_file)
            print(f"✅ Preprocessed image: {processed.shape}")
            
            # Verify preprocessing results
            if len(processed.shape) != 2:
                print("❌ FAIL: Processed image should be grayscale (2D)")
                verification_passed = False
                continue
            
            # Check if preprocessing actually changed the image
            if processed.max() == 0:
                print("❌ WARNING: Processed image is completely black")
            elif processed.min() == processed.max():
                print("❌ WARNING: Processed image has no variation")
            else:
                print("✅ Processed image has proper variation")
            
            # Count non-zero pixels (should represent text)
            text_pixels = cv2.countNonZero(processed)
            total_pixels = processed.shape[0] * processed.shape[1]
            text_percentage = (text_pixels / total_pixels) * 100
            
            print(f"✅ Text coverage: {text_percentage:.2f}% of image")
            
            if text_percentage < 1:
                print("❌ WARNING: Very little text detected")
            elif text_percentage > 50:
                print("❌ WARNING: Too much text detected (might be inverted)")
            else:
                print("✅ Text coverage looks reasonable")
            
            print("-" * 50)
            
        except Exception as e:
            print(f"❌ FAIL: Error processing {sample_file}: {str(e)}")
            verification_passed = False
    
    # Overall result
    if verification_passed:
        print("\n🎉 DAY 1 VERIFICATION PASSED!")
        print("✅ All sample images created successfully")
        print("✅ Image loading works correctly")
        print("✅ Preprocessing pipeline functional")
        print("\nYou're ready for Day 2!")
    else:
        print("\n❌ DAY 1 VERIFICATION FAILED!")
        print("Please fix the issues above before proceeding to Day 2")
    
    return verification_passed

def show_preprocessing_visual():
    """Show visual comparison of original vs preprocessed images"""
    print("\n=== Visual Verification ===")
    print("Displaying preprocessing results...")
    
    analyzer = HandwritingAnalyzer()
    sample_files = [
        'samples/handwriting/sample_good.png',
        'samples/handwriting/sample_poor.png'
    ]
    
    for sample_file in sample_files:
        if os.path.exists(sample_file):
            try:
                analyzer.visualize_preprocessing(sample_file)
                print(f"✅ Displayed preprocessing for {sample_file}")
            except Exception as e:
                print(f"❌ Could not display {sample_file}: {str(e)}")

if __name__ == "__main__":
    # Run verification
    verification_passed = verify_preprocessing()
    
    # Show visual results
    if verification_passed:
        show_preprocessing_visual()
    
    print("\n" + "="*60)
    print("Day 1 verification complete!")
