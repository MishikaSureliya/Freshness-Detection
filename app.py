from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from pymongo import MongoClient
from datetime import datetime

app = Flask(__name__)

# Load the pre-trained model
MODEL_PATH = 'C:/Users/mishi/OneDrive/Desktop/flipkart/Fresh_Rotten_Fruits_MobileNetV2_Transfer_Learning.h5'
model = load_model(MODEL_PATH)

# MongoDB setup
MONGO_URI = "mongodb://localhost:27017/"  # Replace with your MongoDB URI if needed
client = MongoClient(MONGO_URI)
db = client["fruit_database"]  # Database name
collection = db["predictions"]  # Collection name

# Define the target image size
TARGET_SIZE = (224, 224)

# Class labels
CLASS_LABELS = ['Fresh Apple', 'Fresh Banana', 'Fresh Orange', 'Rotten Apple', 'Rotten Bananas', 'Rotten Orange']

# Define the directory where images are saved
IMAGE_DIR = 'C:/Users/mishi/OneDrive/Desktop/flipkart/dataset/test'

# Define shelf life in days for each class
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

# Home route to display the form for image upload
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file:
        # Save the uploaded file to the test folder (no need to move it to static/uploads)
        file_path = os.path.join(IMAGE_DIR, file.filename)
        file.save(file_path)

        # Preprocess the image
        img = load_img(file_path, target_size=TARGET_SIZE)
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Perform prediction
        prediction = model.predict(img_array)
        predicted_class = CLASS_LABELS[np.argmax(prediction)]

        # Get the shelf life for the predicted class
        shelf_life = SHELF_LIFE[predicted_class]

        # Calculate the freshness score
        freshness = calculate_freshness(shelf_life)

        # Save prediction result to MongoDB
        timestamp = datetime.utcnow().isoformat()
        result = {
            "filename": file.filename,
            "prediction": predicted_class,
            "shelf_life": shelf_life,
            "freshness": freshness,
            "timestamp": timestamp
        }
        collection.insert_one(result)

        return render_template('index.html', prediction=predicted_class, shelf_life=shelf_life, freshness=freshness, image_path=file.filename)

# Route to serve images from the dataset/test directory
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(IMAGE_DIR, filename)

# API to view stored predictions in MongoDB (JSON format)
@app.route('/api/predictions', methods=['GET'])
def get_predictions():
    predictions = list(collection.find({}, {"_id": 0}))  # Fetch all predictions, exclude MongoDB's default _id field
    return jsonify(predictions)

# Route to display stored predictions in a tabular format
@app.route('/view-predictions', methods=['GET'])
def view_predictions():
    # Fetch data from MongoDB
    predictions = list(collection.find({}, {"_id": 0}))  # Exclude '_id' for simplicity
    
    # Add freshness score to each prediction based on shelf life
    for prediction in predictions:
        prediction["freshness"] = calculate_freshness(prediction["shelf_life"])
    
    return render_template('predictions.html', predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)
