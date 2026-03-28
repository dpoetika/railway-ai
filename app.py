import os
import numpy as np
import tensorflow as tf
import cv2
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
IMG_SIZE = 128
MODEL_PATH = "lung_disease_classifier.keras"
CLASS_NAMES = ['Lung_Opacity', 'Normal', 'Viral Pneumonia']   # match your training
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(filepath):
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        processed = preprocess_image(filepath)
        probs = model.predict(processed)[0]
        idx = int(np.argmax(probs))
        confidence = float(probs[idx])
        result = {
            'class': CLASS_NAMES[idx],
            'confidence': confidence,
            'all_scores': {cls: float(probs[i]) for i, cls in enumerate(CLASS_NAMES)}
        }
    except Exception as e:
        result = {'error': str(e)}
    finally:
        os.remove(filepath)  # Clean up
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)