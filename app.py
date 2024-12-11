import streamlit as st
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from pymongo import MongoClient
from datetime import datetime
import logging
# Model file uploader
model_file = st.file_uploader("Upload Model File", type=["h5"])

if model_file is not None:
    try:
        # Load the model from the uploaded file
        model = load_model(model_file)
        logging.info("Model loaded successfully.")
        st.success("Model loaded successfully! You can now upload an image.")

        # Image file uploader
        uploaded_file = st.file_uploader("Upload a fruit image for prediction", type=["jpg", "png", "jpeg"])
        
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
