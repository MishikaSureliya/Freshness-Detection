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

# MongoDB setup
MONGO_URI = "mongodb://localhost:27017/"
try:
    client = MongoClient(MONGO_URI)
    db = client["fruit_database"]
    collection = db["predictions"]
    logging.info("Connected to MongoDB.")
except Exception as e:
    logging.error(f"Failed to connect to MongoDB: {e}")
    raise

# Define constants
TARGET_SIZE = (224, 224)
CLASS_LABELS = ['Fresh Apple', 'Fresh Banana', 'Fresh Orange', 'Rotten Apple', 'Rotten Bananas', 'Rotten Orange']
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
        return 1
    elif 1 <= shelf_life <= 5:
        return 2
    elif 6 <= shelf_life <= 10:
        return 3
    elif 11 <= shelf_life <= 15:
        return 4
    else:
        return 5

# Streamlit UI
st.title("Fruit Freshness Prediction")
st.write("Upload a fruit image and the pre-trained model to predict freshness.")

# Model file uploader
model_file = st.file_uploader("Upload Model File", type=["h5"])

if model_file is not None:
    try:
        # Load the model from the uploaded file
        model = load_model(model_file)
        logging.info("Model loaded successfully.")

        # Image file uploader
        uploaded_file = st.file_uploader("Choose a fruit image...", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
            try:
                # Process image
                img = load_img(uploaded_file, target_size=TARGET_SIZE)
                img_array = img_to_array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                # Make prediction
                prediction = model.predict(img_array)
                predicted_class = CLASS_LABELS[np.argmax(prediction)]
                shelf_life = SHELF_LIFE[predicted_class]
                freshness = calculate_freshness(shelf_life)

                # Display results
                st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
                st.write(f"Prediction: {predicted_class}")
                st.write(f"Shelf Life: {shelf_life} days")
                st.write(f"Freshness: {freshness} (1=Rotten, 5=Very Fresh)")

                # Save result to MongoDB
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

        else:
            st.warning("Please upload an image for prediction.")

    except Exception as e:
        logging.error(f"Error loading model: {e}")
        st.error("An error occurred while loading the model. Please upload a valid model file.")

else:
    st.warning("Please upload a model file first.")
