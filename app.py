from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from pymongo import MongoClient
from datetime import datetime
import logging

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load the pre-trained model
MODEL_PATH = 'C:/Users/mishi/OneDrive/Desktop/flipkart/Fresh_Rotten_Fruits_MobileNetV2_Transfer_Learning.h5'
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
model = load_model(MODEL_PATH)
logging.info("Model loaded successfully.")

# MongoDB setup
MONGO_URI = "mongodb://localhost:27017/"  # Replace with your MongoDB URI if needed
try:
    client = MongoClient(MONGO_URI)
    db = client["fruit_database"]  # Database name
    collection = db["predictions"]  # Collection name
    logging.info("Connected to MongoDB.")
except Exception as e:
    logging.error(f"Failed to connect to MongoDB: {e}")
    raise

# Define constants
TARGET_SIZE = (224, 224)
CLASS_LABELS = ['Fresh Apple', 'Fresh Banana', 'Fresh Orange', 'Rotten Apple', 'Rotten Bananas', 'Rotten Orange']
IMAGE_DIR = 'C:/Users/mishi/OneDrive/Desktop/flipkart/dataset/test'
SHELF_LIFE = {
    'Fresh Apple': 10,
    'Fresh Banana': 5,
    'Fresh Orange': 15,
    'Rotten Apple': 0,
    'Rotten Bananas': 0,
    'Rotten Orange': 0
}

# Function to calculate freshness score
def calculate_freshness(shelf_life):
    if shelf_life == 0:
        return 1  # Rotten
    elif 1 <= shelf_life <= 5:
        return 2  # Not very fresh
    elif 6 <= shelf_life <= 10:
        return 3  # Moderately fresh
    elif 11 <= shelf_life <= 15:
        return 4  # Fairly fresh
    else:
        return 5  # Very fresh

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    try:
        file_path = os.path.join(IMAGE_DIR, file.filename)
        file.save(file_path)
        logging.info(f"File saved at {file_path}")

        img = load_img(file_path, target_size=TARGET_SIZE)
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        predicted_class = CLASS_LABELS[np.argmax(prediction)]
        shelf_life = SHELF_LIFE[predicted_class]
        freshness = calculate_freshness(shelf_life)

        timestamp = datetime.utcnow().isoformat()
        result = {
            "filename": file.filename,
            "prediction": predicted_class,
            "shelf_life": shelf_life,
            "freshness": freshness,
            "timestamp": timestamp
        }
        collection.insert_one(result)
        logging.info("Prediction saved to MongoDB.")

        return render_template('index.html', prediction=predicted_class, shelf_life=shelf_life, freshness=freshness, image_path=file.filename)
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return "An error occurred during prediction. Please try again."

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(IMAGE_DIR, filename)

@app.route('/api/predictions', methods=['GET'])
def get_predictions():
    predictions = list(collection.find({}, {"_id": 0}))
    return jsonify(predictions)

@app.route('/view-predictions', methods=['GET'])
def view_predictions():
    predictions = list(collection.find({}, {"_id": 0}))
    for prediction in predictions:
        prediction["freshness"] = calculate_freshness(prediction["shelf_life"])
    return render_template('predictions.html', predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)
