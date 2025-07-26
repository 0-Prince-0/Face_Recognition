#!/usr/bin/env python3
"""
Test script to verify face recognition model functionality
"""
import os
import sys

# Check if required modules are available
try:
    import cv2
    import face_recognition
    import numpy as np
    from sklearn.svm import SVC
    print("✓ All required modules are available")
    
    # Test the model
    from mdl import FaceRecognitionModel
    
    print("Testing Face Recognition Model...")
    model = FaceRecognitionModel()
    
    print(f"Known people: {model.get_known_people()}")
    print(f"Number of known encodings: {len(model.known_encodings)}")
    print(f"Model fitted: {model._is_model_fitted() if model.model else False}")
    
    print("✓ Face Recognition Model test completed successfully!")
    
except ImportError as e:
    print(f"✗ Missing required module: {e}")
    print("\nTo install required packages, run:")
    print("pip3 install opencv-python face-recognition scikit-learn numpy flask")
except Exception as e:
    print(f"✗ Error testing model: {e}")
