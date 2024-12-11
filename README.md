**DATABASE HANDLING**
In our Fruit Freshness Detection system, we utilized MongoDB to store and manage the data efficiently. The database table includes the following columns:
	•	File Name: The name of the image file analyzed.
	•	Prediction: The classification result of the fruit’s freshness (e.g., Fresh or Rotten).
	•	Shelf Life: The estimated remaining shelf life of the fruit based on the prediction.
	•	Freshness: A numerical score representing the freshness level predicted by the ML model.
	•	Timestamp: The date and time when the analysis was performed, allowing for accurate record-keeping and tracking.

This structured data supports seamless integration with our web UI, enabling real-time display and retrieval of freshness results.
![Screenshot 2024-12-06 190543](https://github.com/user-attachments/assets/736d4395-96a1-492e-9d47-709100a88628)

**DEPLOYMENT**
This project implements a Fruit Freshness Detector web application using Flask. The application allows users to upload an image of a fruit, and it predicts the fruit's freshness along with its shelf life.

Deployment Overview
Backend Framework: The backend is powered by Flask, a lightweight WSGI web application framework in Python.
Frontend Interface: The user interface is designed with HTML and CSS, providing a clean and intuitive experience.
Model Integration: The application integrates a machine learning model that:
Identifies the fruit in the uploaded image.
Predicts its freshness status (e.g., Fresh Apple).
Displays the estimated shelf life of the fruit.
Workflow
Upload: Users upload an image of a fruit via the interface.
Prediction: The Flask app processes the image, runs it through the trained model, and returns a prediction.
Output: The results, including the freshness prediction and shelf life, are displayed in an easy-to-read format.
Technical Stack
Backend: Python (Flask)
Frontend: HTML, CSS
Machine Learning Libraries: NumPy, Pandas, and other relevant libraries for preprocessing and prediction.
![WhatsApp Image 2024-12-12 at 12 46 20 AM](https://github.com/user-attachments/assets/46a21d4a-a25b-4d36-aa15-d78c1998d17a)
