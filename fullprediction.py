import os
import glob
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np

# Load the model
model = load_model('Fresh_Rotten_Fruits_MobileNetV2_Transfer_Learning.h5')

# Define the class labels
class_labels = ['Fresh Apple', 'Fresh Banana', 'Fresh Orange', 'Rotten Apple', 'Rotten Banana', 'Rotten Orange']

# Function to preprocess the image
def preprocess_image(img_path):
    # Load and resize the image to the input size expected by MobileNetV2
    img = image.load_img(img_path, target_size=(224, 224))
    # Convert the image to an array
    img_array = image.img_to_array(img)
    # Expand dimensions to match the model's expected input (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    # Apply MobileNetV2 preprocessing (normalization)
    img_array = preprocess_input(img_array)
    return img_array

# Function to make predictions
def predict_fruit(img_path, model):
    # Preprocess the image
    img_array = preprocess_image(img_path)
    # Make predictions
    predictions = model.predict(img_array)
    
    # Output prediction probabilities for debugging
    print(f"Prediction probabilities: {predictions}")
    
    # Get the index of the class with the highest probability
    predicted_class = np.argmax(predictions, axis=1)
    # Return the class label corresponding to the predicted class index
    return class_labels[predicted_class[0]]

# Directory containing the test images
test_dir = "C:/Users/mishi/OneDrive/Desktop/flipkart/dataset/test"

# Get a list of all images in the test directory (and subdirectories)
image_paths = glob.glob(os.path.join(test_dir, '**', '*.png'), recursive=True)  # Adjust the file extension if necessary

# Process all images and make predictions
predictions = []  # To store predictions for each image

for img_path in image_paths:
    prediction = predict_fruit(img_path, model)
    predictions.append((img_path, prediction))  # Store the image path and prediction
    print(f"Image: {img_path}, Predicted Label: {prediction}")

# Optionally, you can save the predictions to a CSV file for easier analysis
import pandas as pd

# Convert the predictions to a DataFrame and save as CSV
predictions_df = pd.DataFrame(predictions, columns=['Image Path', 'Predicted Label'])
predictions_df.to_csv('fruit_predictions.csv', index=False)

print("Predictions saved to fruit_predictions.csv")