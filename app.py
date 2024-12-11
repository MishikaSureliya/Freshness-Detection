import streamlit as st
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from pymongo import MongoClient
from datetime import datetime
import logging
import traceback

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load the pre-trained model
MODEL_PATH = 'Fresh_Rotten_Fruits_MobileNetV2_Transfer_Learning.h5'
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = load_model(MODEL_PATH)
        logging.info("Model loaded successfully.")
    except Exception as e:
        st.error(f"Failed to load the model. Error: {e}")
        logging.error(f"Failed to load the model: {traceback.format_exc()}")
        raise
else:
    st.error(f"Model file not found at {MODEL_PATH}. Please check the path.")
    logging.error(f"Model file not found at {MODEL_PATH}.")
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}.")

# MongoDB connection
MONGO_URI = "mongodb+srv://mishikasureliya29:Mishika%4029@cluster0.ggvst.mongodb.net/fruit_database?retryWrites=true&w=majority"
collection = None
try:
    client = MongoClient(MONGO_URI)
    db = client["FruitFresh"]
    collection = db["Predictions"]
    logging.info("Connected to MongoDB Atlas.")
except Exception as e:
    st.error("Failed to connect to MongoDB Atlas. Please check your MongoDB setup.")
    logging.error(f"Failed to connect to MongoDB Atlas: {traceback.format_exc()}")
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
st.write("Upload multiple fruit images to predict their freshness and shelf life.")

# File uploader for multiple files
uploaded_files = st.file_uploader("Choose fruit images...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

# Function to process each image
def process_images(uploaded_files):
    for uploaded_file in uploaded_files:
        try:
            # Preprocess image
            img = load_img(uploaded_file, target_size=TARGET_SIZE)
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Make prediction
            prediction = model.predict(img_array)
            predicted_class = CLASS_LABELS[np.argmax(prediction)]
            shelf_life = SHELF_LIFE[predicted_class]
            freshness = calculate_freshness(shelf_life)

            # Display results for each image
            st.image(uploaded_file, caption=f"Uploaded Image: {uploaded_file.name}", use_column_width=True)
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
            logging.info(f"Prediction for {uploaded_file.name} saved to MongoDB.")

        except Exception as e:
            logging.error(f"Error during prediction for {uploaded_file.name}: {traceback.format_exc()}")
            st.error(f"An error occurred during prediction for {uploaded_file.name}. Please check the logs.")

# If files are uploaded, process them
if uploaded_files:
    process_images(uploaded_files)

# Button to display all previous predictions from MongoDB
if st.button("View All Predictions"):
    try:
        predictions = list(collection.find({}, {"_id": 0}))
        if predictions:
            st.write(f"Total Predictions: {len(predictions)}")
            for prediction in predictions:
                freshness = calculate_freshness(prediction["shelf_life"])
                prediction["freshness"] = freshness
                st.write(prediction)
        else:
            st.write("No predictions found.")
    except Exception as e:
        logging.error(f"Error retrieving predictions from MongoDB: {traceback.format_exc()}")
        st.error("Failed to retrieve predictions. Please check the logs.")

# Button to test MongoDB connection
if st.button("Test MongoDB Connection"):
    try:
        # Try querying MongoDB to test connection
        collection.count_documents({})  # Just a simple operation to check if the connection works
        st.success("MongoDB connection is successful!")
        logging.info("MongoDB connection is successful.")
    except Exception as e:
        st.error("MongoDB connection failed.")
        logging.error(f"Error while testing MongoDB connection: {traceback.format_exc()}")
