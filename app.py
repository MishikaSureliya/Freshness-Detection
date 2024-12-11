import streamlit as st
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from pymongo import MongoClient
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load the pre-trained model
MODEL_PATH = 'Fresh_Rotten_Fruits_MobileNetV2_Transfer_Learning.h5'
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

# Streamlit UI
st.title("Fruit Freshness Prediction")
st.write("Upload a fruit image to predict its freshness and shelf life.")

# File uploader
uploaded_file = st.file_uploader("Choose a fruit image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        # Save the uploaded file to a local directory (optional, for logging)
        file_path = os.path.join(IMAGE_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        # Preprocess image
        img = load_img(file_path, target_size=TARGET_SIZE)
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        prediction = model.predict(img_array)
        predicted_class = CLASS_LABELS[np.argmax(prediction)]
        shelf_life = SHELF_LIFE[predicted_class]
        freshness = calculate_freshness(shelf_life)

        # Display results
        st.write(f"Prediction: {predicted_class}")
        st.write(f"Shelf Life: {shelf_life} days")
        st.write(f"Freshness: {freshness} (1=Rotten, 5=Very Fresh)")

        # Save the result to MongoDB
        timestamp = datetime.utcnow().isoformat()
        result = {
            "filename": uploaded_file.name,
            "prediction": predicted_class,
            "shelf_life": shelf_life,
            "freshness": freshness,
            "timestamp": timestamp
        }
        collection.insert_one(result)
        logging.info("Prediction saved to MongoDB.")

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        st.error("An error occurred during prediction. Please try again.")

# Display predictions stored in MongoDB
if st.button("View All Predictions"):
    predictions = list(collection.find({}, {"_id": 0}))
    if predictions:
        st.write(f"Total Predictions: {len(predictions)}")
        for prediction in predictions:
            freshness = calculate_freshness(prediction["shelf_life"])
            prediction["freshness"] = freshness
            st.write(prediction)
    else:
        st.write("No predictions found.")

