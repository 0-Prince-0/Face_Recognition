import cv2
import numpy as np
import os
import pickle
import face_recognition
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Input
from tensorflow.keras.optimizers import Adam
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceRecognitionModel:
    def __init__(self, model_path="face_model.pkl", encodings_path="face_encodings.pkl"):
        self.model_path = model_path
        self.encodings_path = encodings_path
        self.known_encodings = []
        self.known_names = []
        self.model = None
        self.label_encoder = LabelEncoder()
        self.load_model()
    
    def extract_face_encoding(self, image_path):
        """Extract face encoding from an image"""
        try:
            # Load image
            image = face_recognition.load_image_file(image_path)
            
            # Find face locations
            face_locations = face_recognition.face_locations(image)
            
            if not face_locations:
                logger.warning(f"No face found in {image_path}")
                return None
            
            # Get face encodings
            face_encodings = face_recognition.face_encodings(image, face_locations)
            
            if face_encodings:
                return face_encodings[0]
            return None
        except Exception as e:
            logger.error(f"Error extracting face encoding from {image_path}: {e}")
            return None
    
    def add_person(self, person_name, images_folder):
        """Add a new person to the model by training on multiple images"""
        try:
            person_encodings = []
            image_files = [f for f in os.listdir(images_folder) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if not image_files:
                logger.error(f"No image files found in {images_folder}")
                return False
            
            for image_file in image_files:
                image_path = os.path.join(images_folder, image_file)
                encoding = self.extract_face_encoding(image_path)
                
                if encoding is not None:
                    person_encodings.append(encoding)
                    self.known_encodings.append(encoding)
                    self.known_names.append(person_name)
            
            if person_encodings:
                logger.info(f"Added {len(person_encodings)} encodings for {person_name}")
                self.train_model()
                self.save_model()
                return True
            else:
                logger.error(f"No valid face encodings found for {person_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error adding person {person_name}: {e}")
            return False
    
    def train_model(self):
        """Train the SVM model with current encodings"""
        if len(self.known_encodings) == 0:
            logger.warning("No encodings available for training")
            return
        
        try:
            # Prepare data
            X = np.array(self.known_encodings)
            y = np.array(self.known_names)
            
            # Encode labels
            y_encoded = self.label_encoder.fit_transform(y)
            
            # Train SVM model
            self.model = SVC(kernel='linear', probability=True, random_state=42)
            self.model.fit(X, y_encoded)
            
            logger.info("Model trained successfully")
            
            # Save the trained model
            self.save_model()
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
    
    def _is_model_fitted(self):
        """Check if the SVM model is properly fitted"""
        try:
            # Check if the model has the required attributes for a fitted SVM
            return hasattr(self.model, 'support_vectors_') and hasattr(self.model, 'classes_')
        except:
            return False
    
    def predict_face(self, image_path=None, image_array=None):
        """Predict the person in an image using SVM model with face distance fallback"""
        try:
            # Extract encoding
            if image_path:
                encoding = self.extract_face_encoding(image_path)
            elif image_array is not None:
                # Convert array to face encoding
                face_locations = face_recognition.face_locations(image_array)
                if not face_locations:
                    return "No face detected", 0.0
                
                face_encodings = face_recognition.face_encodings(image_array, face_locations)
                if not face_encodings:
                    return "No face detected", 0.0
                
                encoding = face_encodings[0]
            else:
                return "No input provided", 0.0
            
            if encoding is None:
                return "No face detected", 0.0
            
            # If no encodings available, return unknown
            if len(self.known_encodings) == 0:
                return "Unknown", 0.0
            
            # Try SVM prediction first if model is available and fitted
            if self.model is not None and self._is_model_fitted():
                try:
                    encoding_array = np.array([encoding])
                    prediction = self.model.predict(encoding_array)
                    probabilities = self.model.predict_proba(encoding_array)
                    
                    # Get the predicted name and confidence
                    predicted_label = prediction[0]
                    confidence = np.max(probabilities[0]) * 100  # Convert to percentage
                    predicted_name = self.label_encoder.inverse_transform([predicted_label])[0]
                    
                    # Set threshold for unknown faces
                    if confidence > 60:  # 60% threshold
                        return predicted_name, confidence
                    else:
                        # If SVM confidence is low, fall back to face distance method
                        logger.info("SVM confidence low, using face distance method")
                
                except Exception as e:
                    logger.warning(f"SVM prediction failed: {e}, falling back to face distance")
            
            # Face distance method (fallback or primary if SVM not available)
            face_distances = face_recognition.face_distance(self.known_encodings, encoding)
            
            # Find the best match
            best_match_index = np.argmin(face_distances)
            min_distance = face_distances[best_match_index]
            
            # Convert distance to confidence (lower distance = higher confidence)
            confidence = max(0.0, (1.0 - min_distance) * 100)  # Convert to percentage
            
            # Set threshold for recognition
            if min_distance < 0.6:  # Good threshold for face recognition
                predicted_name = self.known_names[best_match_index]
                return predicted_name, confidence
            else:
                return "Unknown", confidence
            
        except Exception as e:
            logger.error(f"Error predicting face: {e}")
            return "Error", 0.0
    
    def recognize_face_live(self, frame):
        """Recognize faces in a live video frame using face distance comparison"""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Find face locations and encodings
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            results = []
            
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                predicted_name = "Unknown"
                confidence = 0.0
                
                if len(self.known_encodings) > 0:
                    # Use face distance comparison for live recognition (faster and more reliable)
                    face_distances = face_recognition.face_distance(self.known_encodings, face_encoding)
                    
                    # Find the best match
                    best_match_index = np.argmin(face_distances)
                    min_distance = face_distances[best_match_index]
                    
                    logger.debug(f"Live recognition - min distance: {min_distance:.3f}")
                    
                    # Convert distance to confidence (lower distance = higher confidence)
                    # Distance typically ranges from 0 to 1, so confidence = 1 - distance
                    confidence = max(0.0, (1.0 - min_distance) * 100)  # Convert to percentage
                    
                    # Set threshold for recognition (0.6 distance = 40% confidence)
                    if min_distance < 0.6:  # This is a good threshold for face recognition
                        predicted_name = self.known_names[best_match_index]
                        logger.debug(f"Live recognition - matched: {predicted_name} with {confidence:.1f}% confidence")
                    else:
                        predicted_name = "Unknown"
                        logger.debug(f"Live recognition - no match, confidence too low: {confidence:.1f}%")
                
                results.append({
                    'name': predicted_name,
                    'confidence': confidence,
                    'location': (top, right, bottom, left)
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in live recognition: {e}")
            return []
    
    def save_model(self):
        """Save the trained model and encodings"""
        try:
            # Save encodings and names
            data = {
                'encodings': self.known_encodings,
                'names': self.known_names
            }
            with open(self.encodings_path, 'wb') as f:
                pickle.dump(data, f)
            
            # Save the trained model
            if self.model is not None:
                model_data = {
                    'model': self.model,
                    'label_encoder': self.label_encoder
                }
                with open(self.model_path, 'wb') as f:
                    pickle.dump(model_data, f)
            
            logger.info("Model saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load_model(self):
        """Load the trained model and encodings"""
        try:
            # Load encodings
            if os.path.exists(self.encodings_path):
                with open(self.encodings_path, 'rb') as f:
                    data = pickle.load(f)
                    self.known_encodings = data['encodings']
                    self.known_names = data['names']
                logger.info(f"Loaded {len(self.known_encodings)} face encodings")
            
            # Load model
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.model = model_data['model']
                    self.label_encoder = model_data['label_encoder']
                
                # Check if the model is properly fitted
                if not self._is_model_fitted():
                    logger.warning("Loaded model is not properly fitted. Retraining...")
                    if len(self.known_encodings) > 0:
                        self.train_model()
                    else:
                        self.model = None
                else:
                    logger.info("Model loaded successfully")
            else:
                # If no model exists but we have encodings, train a new model
                if len(self.known_encodings) > 0:
                    logger.info("No saved model found. Training new model...")
                    self.train_model()
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            # If loading fails, try to retrain if we have encodings
            if len(self.known_encodings) > 0:
                logger.info("Attempting to retrain model...")
                self.train_model()
            else:
                self.model = None
    
    def get_known_people(self):
        """Get list of known people"""
        return list(set(self.known_names))
    
    def delete_person(self, person_name):
        """Remove a person from the model"""
        try:
            # Remove encodings for this person
            indices_to_remove = []
            for i, name in enumerate(self.known_names):
                if name == person_name:
                    indices_to_remove.append(i)
            
            # Remove in reverse order to maintain indices
            for i in reversed(indices_to_remove):
                del self.known_encodings[i]
                del self.known_names[i]
            
            if indices_to_remove:
                logger.info(f"Removed {len(indices_to_remove)} encodings for {person_name}")
                # Retrain model if there are still people left
                if self.known_names:
                    self.train_model()
                else:
                    self.model = None
                self.save_model()
                return True
            else:
                logger.warning(f"Person {person_name} not found")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting person {person_name}: {e}")
            return False

    def predict_face_distance_only(self, image_path=None, image_array=None):
        """Predict using only face distance comparison (no SVM)"""
        try:
            # Extract encoding
            if image_path:
                encoding = self.extract_face_encoding(image_path)
            elif image_array is not None:
                # Convert array to face encoding
                face_locations = face_recognition.face_locations(image_array)
                if not face_locations:
                    return "No face detected", 0.0
                
                face_encodings = face_recognition.face_encodings(image_array, face_locations)
                if not face_encodings:
                    return "No face detected", 0.0
                
                encoding = face_encodings[0]
            else:
                return "No input provided", 0.0
            
            if encoding is None:
                return "No face detected", 0.0
            
            # If no encodings available, return unknown
            if len(self.known_encodings) == 0:
                return "Unknown", 0.0
            
            # Face distance method
            face_distances = face_recognition.face_distance(self.known_encodings, encoding)
            
            # Find the best match
            best_match_index = np.argmin(face_distances)
            min_distance = face_distances[best_match_index]
            
            # Convert distance to confidence percentage
            confidence = max(0.0, (1.0 - min_distance) * 100)  # Convert to percentage
            
            # Set threshold for recognition
            if min_distance < 0.6:  # Good threshold for face recognition
                predicted_name = self.known_names[best_match_index]
                return predicted_name, confidence
            else:
                return "Unknown", confidence
            
        except Exception as e:
            logger.error(f"Error in face distance prediction: {e}")
            return "Error", 0.0


# Utility functions for image processing
def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess image for training"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # Resize image
        image = cv2.resize(image, target_size)
        
        # Normalize pixel values
        image = image.astype(np.float32) / 255.0
        
        return image
    except Exception as e:
        logger.error(f"Error preprocessing image {image_path}: {e}")
        return None

def create_training_folder(person_name, base_path="training_data"):
    """Create folder for training images"""
    person_folder = os.path.join(base_path, person_name)
    os.makedirs(person_folder, exist_ok=True)
    return person_folder

def capture_training_images(person_name, num_images=20):
    """Capture training images from webcam"""
    person_folder = create_training_folder(person_name)
    
    cap = cv2.VideoCapture(0)
    count = 0
    
    print(f"Capturing {num_images} images for {person_name}")
    print("Press SPACE to capture image, ESC to exit")
    
    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Display frame
        cv2.putText(frame, f"Images captured: {count}/{num_images}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Press SPACE to capture, ESC to exit", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow('Capture Training Images', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # Space key
            image_path = os.path.join(person_folder, f"{person_name}_{count:03d}.jpg")
            cv2.imwrite(image_path, frame)
            count += 1
            print(f"Captured image {count}/{num_images}")
        elif key == 27:  # ESC key
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    return person_folder if count > 0 else None

if __name__ == "__main__":
    # Example usage
    model = FaceRecognitionModel()
    
    # Test the model
    print("Face Recognition Model initialized")
    print("Known people:", model.get_known_people())