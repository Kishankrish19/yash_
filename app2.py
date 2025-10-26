import os
import io
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image
from deepface import DeepFace

# --- Configuration ---
app = Flask(__name__)

# Define the path to your "golden image"
# __file__ gives the path to app.py. os.path.dirname gets the folder it's in.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FACE_DATA_DIR = os.path.join(BASE_DIR, "face_data")
GOLDEN_IMAGE_PATH = os.path.join(FACE_DATA_DIR, "preethi_golden.jpg")

# --- Pre-load Model & Check Files ---
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

    # 4. Load DeepFace model
    print("Loading DeepFace model... This might take a moment.")
    DeepFace.build_model("VGG-Face")
    print("Model loaded successfully.")
    
    # 5. Verify golden image is readable by DeepFace
    # We do a simple find operation to cache the model and test the image.
    DeepFace.find(
        img_path=GOLDEN_IMAGE_PATH,
        db_path=FACE_DATA_DIR,
        model_name="VGG-Face",
        enforce_detection=False,
        silent=True
    )
    print("Golden image is valid and ready.")
    print("Initialization complete.")
    print("="*50)

except Exception as e:
    print(f"!!! FATAL ERROR DURING INITIALIZATION: {e} !!!")
    print("Please check your file paths and installations.")
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

    file = request.files['image']

    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img_np = np.array(img)
        print("Image received, running verification...")
        
    except Exception as e:
        print(f"Verification FAILED: Could not read image - {e}")
        return jsonify({"status": "failure", "error": "Could not process image"}), 400

    try:
        result = DeepFace.verify(
            img1_path=img_np,
            img2_path=GOLDEN_IMAGE_PATH,
            model_name="VGG-Face",
            enforce_detection=True
        )

        if result.get("verified") == True:
            print("Verification SUCCESS: Faces match!")
            return jsonify({"status": "success"})
        else:
            print("Verification FAILURE: Faces do not match.")
            return jsonify({"status": "failure"})

    except ValueError as e:
        # This is the most common error: "Face could not be detected"
        print(f"Verification FAILED: Could not find a face. Error: {e}")
        return jsonify({"status": "failure", "error": "No face detected"})
    except Exception as e:
        print(f"Verification FAILED: An unexpected error. Error: {e}")
        return jsonify({"status": "failure", "error": str(e)})

# --- Run the Server ---
if __name__ == '__main__':
    print("="*50)
    print("Starting the Magical Birthday Server...")
    print(f"Open this *exact* URL in your browser: http://127.0.0.1:5000")
    print("Do NOT add '/index.html' to the end.")
    print("="*50)
    # host='0.0.0.0' makes it accessible on your local network
    # (e.g., from your phone), but '127.0.0.1' is safer.
    app.run(host='127.0.0.1', port=5000, debug=False)

