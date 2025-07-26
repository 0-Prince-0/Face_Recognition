from flask import Flask, render_template, request, jsonify, Response
import cv2
import os
import json
import base64
import numpy as np
from mdl import FaceRecognitionModel, create_training_folder
import threading
import time
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'face_recognition_secret_key'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize the face recognition model
face_model = FaceRecognitionModel()

# Global variables for video streaming
camera = None
camera_active = False
latest_results = []  # Store latest face recognition results

class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        
    def __del__(self):
        self.video.release()
        
    def get_frame(self):
        global latest_results
        success, image = self.video.read()
        if not success:
            return None
        
        # Perform face recognition
        results = face_model.recognize_face_live(image)
        
        # Store latest results for the live results endpoint
        latest_results = results
        
        # Draw rectangles and labels on the image
        for result in results:
            top, right, bottom, left = result['location']
            name = result['name']
            confidence = result['confidence']
            
            # Draw rectangle around face
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Draw label with percentage
            label = f"{name} ({confidence:.1f}%)"
            cv2.rectangle(image, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(image, label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()
        
        return frame

def gen_frames():
    global camera
    camera = VideoCamera()
    while camera_active:
        frame = camera.get_frame()
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/live_recognition')
def live_recognition():
    return render_template('live_recognition.html')

@app.route('/image_recognition')
def image_recognition():
    return render_template('image_recognition.html')

@app.route('/image_recognition_page')
def image_recognition_page():
    return render_template('image_recognition_page.html')

@app.route('/add_face')
def add_face():
    return render_template('add_face.html')

@app.route('/manage_people')
def manage_people():
    known_people = face_model.get_known_people()
    return render_template('manage_people.html', people=known_people)

@app.route('/video_feed')
def video_feed():
    global camera_active
    camera_active = True
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_camera')
def stop_camera():
    global camera_active, camera, latest_results
    camera_active = False
    latest_results = []  # Clear results when camera stops
    if camera:
        del camera
        camera = None
    return jsonify({'status': 'Camera stopped'})

@app.route('/live_results')
def live_results():
    """Get the latest face recognition results from live video"""
    global latest_results
    if camera_active and latest_results:
        # Format results for JSON response (remove location data which isn't needed for display)
        formatted_results = []
        for result in latest_results:
            formatted_results.append({
                'name': result['name'],
                'confidence': round(result['confidence'], 1)
            })
        return jsonify(formatted_results)
    else:
        return jsonify([])

@app.route('/capture_images', methods=['POST'])
def capture_images():
    try:
        data = request.get_json()
        person_name = data.get('person_name', '').strip()
        num_images = int(data.get('num_images', 20))
        
        if not person_name:
            return jsonify({'success': False, 'message': 'Person name is required'})
        
        # Create folder for this person
        person_folder = create_training_folder(person_name)
        
        # Capture images from webcam
        cap = cv2.VideoCapture(0)
        captured_count = 0
        
        for i in range(num_images):
            ret, frame = cap.read()
            if ret:
                image_path = os.path.join(person_folder, f"{person_name}_{i:03d}.jpg")
                cv2.imwrite(image_path, frame)
                captured_count += 1
                time.sleep(0.5)  # Small delay between captures
        
        cap.release()
        
        if captured_count > 0:
            # Train the model with new images
            success = face_model.add_person(person_name, person_folder)
            if success:
                return jsonify({
                    'success': True, 
                    'message': f'Successfully captured {captured_count} images and trained model for {person_name}'
                })
            else:
                return jsonify({
                    'success': False, 
                    'message': f'Captured images but failed to train model for {person_name}'
                })
        else:
            return jsonify({
                'success': False, 
                'message': 'Failed to capture any images'
            })
            
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/upload_images', methods=['POST'])
def upload_images():
    try:
        print("Upload request received")  # Debug
        person_name = request.form.get('person_name', '').strip()
        print(f"Person name: {person_name}")  # Debug
        
        if not person_name:
            return jsonify({'success': False, 'message': 'Person name is required'})
        
        files = request.files.getlist('images')
        print(f"Number of files received: {len(files)}")  # Debug
        
        if not files or len(files) == 0:
            return jsonify({'success': False, 'message': 'No images uploaded'})
        
        # Create folder for this person
        person_folder = create_training_folder(person_name)
        print(f"Created folder: {person_folder}")  # Debug
        
        saved_count = 0
        for i, file in enumerate(files):
            if file and file.filename != '':
                # Save uploaded image
                filename = f"{person_name}_{i:03d}_{file.filename}"
                file_path = os.path.join(person_folder, filename)
                file.save(file_path)
                saved_count += 1
        
        if saved_count > 0:
            # Train the model with new images
            success = face_model.add_person(person_name, person_folder)
            if success:
                return jsonify({
                    'success': True, 
                    'message': f'Successfully uploaded {saved_count} images and trained model for {person_name}'
                })
            else:
                return jsonify({
                    'success': False, 
                    'message': f'Uploaded images but failed to train model for {person_name}'
                })
        else:
            return jsonify({
                'success': False, 
                'message': 'Failed to save any images'
            })
            
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/predict_image', methods=['POST'])
def predict_image():
    try:
        file = request.files['image']
        
        if file and file.filename != '':
            # Save temporary image
            temp_path = 'temp_prediction.jpg'
            file.save(temp_path)
            
            # Predict
            name, confidence = face_model.predict_face(temp_path)
            
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return jsonify({
                'success': True,
                'prediction': name,
                'confidence': float(confidence)
            })
        else:
            return jsonify({'success': False, 'message': 'No image provided'})
            
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/delete_person', methods=['POST'])
def delete_person():
    try:
        data = request.get_json()
        person_name = data.get('person_name', '').strip()
        
        if not person_name:
            return jsonify({'success': False, 'message': 'Person name is required'})
        
        success = face_model.delete_person(person_name)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Successfully deleted {person_name} from the model'
            })
        else:
            return jsonify({
                'success': False,
                'message': f'Failed to delete {person_name} - person not found'
            })
            
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/get_known_people')
def get_known_people():
    try:
        people = face_model.get_known_people()
        return jsonify({'success': True, 'people': people})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('training_data', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    # Get port from environment variable (Render uses PORT=10000 by default)
    port = int(os.environ.get('PORT', 8090))
    
    # Run the app (Render needs host='0.0.0.0')
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)
else:
    # This is for when running with gunicorn
    gunicorn_app = app
