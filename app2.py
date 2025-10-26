import os
import io
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image
import face_recognition # <-- IMPORT THE NEW LIBRARY

# --- Configuration ---
app = Flask(__name__)

# Define the path to your "golden image"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FACE_DATA_DIR = os.path.join(BASE_DIR, "face_data")
GOLDEN_IMAGE_PATH = os.path.join(FACE_DATA_DIR, "preethi_golden.jpg")

# --- Pre-load Model & Golden Image Encoding ---
# This is a major optimization. We load the golden image
# and get its facial "encoding" (a 128-number list) ONCE.
# Then we just compare against this list, which is super fast
# and uses almost no memory per request.

golden_face_encoding = None

try:
    print("="*50)
    print("Initializing Server...")
    
    # 1. Check for 'face_data' folder
    if not os.path.exists(FACE_DATA_DIR):
        print(f"Error: 'face_data' folder not found. Please create it.")
        print(f"Expected at: {FACE_DATA_DIR}")
    
    # 2. Check for 'preethi_golden.jpg'
    if not os.path.exists(GOLDEN_IMAGE_PATH):
        print(f"!!! FATAL ERROR: Golden image not found !!!")
        print(f"I am looking for: {GOLDEN_IMAGE_PATH}")
        print("Please add 'preethi_golden.jpg' to the 'face_data' folder.")
        print("="*50)
    else:
        print(f"Found golden image: {GOLDEN_IMAGE_PATH}")
        
    # 3. Check for 'templates' folder
    if not os.path.exists(os.path.join(BASE_DIR, "templates")):
        print(f"Error: 'templates' folder not found. Please create it.")
        print(f"Expected at: {os.path.join(BASE_DIR, 'templates')}")

    # 4. Load face_recognition model and encode golden image
    print("Loading face_recognition model and encoding golden image...")
    
    # Load the golden image file
    golden_image_file = face_recognition.load_image_file(GOLDEN_IMAGE_PATH)
    
    # Get the face encodings (features) from the image
    # This returns a list of faces. We assume the first face [0] is the correct one.
    all_encodings = face_recognition.face_encodings(golden_image_file)
    
    if not all_encodings:
        print(f"!!! FATAL ERROR: No face found in golden image: {GOLDEN_IMAGE_PATH}")
        print("Please use a clear, frontal photo.")
        print("="*50)
    else:
        golden_face_encoding = all_encodings[0]
        print("Golden image encoded successfully.")
        print("Initialization complete.")
        print("="*50)

except Exception as e:
    print(f"!!! FATAL ERROR DURING INITIALIZATION: {e} !!!")
    print("Please check your file paths, installations, and the golden image.")
    print("="*50)

# --- Route 1: The Main Website ---
@app.route('/')
def home():
    """Serves the main index.html file from the 'templates' folder."""
    print(f"DEBUG: Request received for / (the home page). Sending index.html...")
    return render_template('index.html')

# --- Route 2: The Taunting Page ---
@app.route('/taunting')
def taunting_page():
    """Serves the taunting.html file from the 'templates' folder."""
    print(f"DEBUG: Request received for /taunting. Sending taunting.html...")
    return render_template('taunting.html')

# --- Route 3: The Verification API ---
@app.route('/verify', methods=['POST'])
def verify_face():
    print(f"DEBUG: Request received for /verify (API call)...")
    
    if 'image' not in request.files:
        print("Verification FAILED: 'image' not in request")
        return jsonify({"status": "failure", "error": "No image file found"}), 400

    if golden_face_encoding is None:
        print("Verification FAILED: Server is not ready (golden image not loaded).")
        return jsonify({"status": "failure", "error": "Server not ready"}), 500

    file = request.files['image']

    try:
        # Read the image file from the request
        img_bytes = file.read()
        
        # Load the image into the face_recognition library
        # We can load directly from the bytes
        unknown_image = face_recognition.load_image_file(io.BytesIO(img_bytes))
        print("Image received, running verification...")
        
    except Exception as e:
        print(f"Verification FAILED: Could not read image - {e}")
        return jsonify({"status": "failure", "error": "Could not process image"}), 400

    try:
        # Get face encodings for the unknown image
        # This will be a list of all faces found
        unknown_face_encodings = face_recognition.face_encodings(unknown_image)

        if not unknown_face_encodings:
            # This handles the "No face detected" case
            print("Verification FAILED: No face detected in uploaded image.")
            return jsonify({"status": "failure", "error": "No face detected"})

        # Compare the first face found in the upload against our golden encoding
        # compare_faces returns a list of [True] or [False]
        # We check the first result: results[0]
        results = face_recognition.compare_faces(
            [golden_face_encoding], # A list of known encodings (we just have one)
            unknown_face_encodings[0] # The first face found in the unknown image
        )

        if results[0] == True:
            print("Verification SUCCESS: Faces match!")
            return jsonify({"status": "success"})
        else:
            print("Verification FAILURE: Faces do not match.")
            return jsonify({"status": "failure"})

    except Exception as e:
        print(f"Verification FAILED: An unexpected error. Error: {e}")
        return jsonify({"status": "failure", "error": str(e)}) 

# --- Run the Server ---
if __name__ == '__main__':
    print("="*50)
    print("Starting the Magical Birthday Server (Lightweight Mode)...")
    print(f"Open this *exact* URL in your browser: http://127.0.0.1:5000")
    print("Do NOT add '/index.html' to the end.")
    print("="*50)
    app.run(host='127.0.0.1', port=5000, debug=False)
