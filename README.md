# FACE-AND-EMOTION-ANALYSIS

Facial Recognition System with Emotion Analysis

This project is a real-time facial recognition system that incorporates emotion detection to identify people and analyze their emotional state. The system captures live video from a webcam, detects faces, matches them against known faces, and determines the emotion of the detected person using a deep learning-based model. All the information, including the recognized user and their detected emotion, is logged into a MySQL database along with a timestamp.
The application uses the Flask web framework to serve a web interface that streams live video, making it easy for users to interact with the system through a web browser.

Features of the Project:

Facial Recognition: Identifies and recognizes faces in real-time from a live webcam feed. Uses pre-trained face encodings to match against known faces.

Emotion Detection: Detects the emotional state of the recognized individual (e.g., Happy, Sad, Angry, etc.) using a deep learning-based emotion model trained on facial expressions.

Web-Based Interface: Provides a live video feed on a web browser using Flask and OpenCV. Users can monitor real-time facial recognition and emotion detection through a simple interface.

Database Logging: Logs detected users' names, emotions, and the time of detection in a MySQL database for future reference or analysis.

Face and Emotion Tracking: Includes functionality to track face features (eyes, smiles) using Haar cascades, improving facial detection accuracy.

Technologies and Tools Used:

Python: The core programming language for the backend processing.

Flask: A lightweight web framework used to create the web application and stream live video feeds.

OpenCV: For capturing video from the webcam and handling face detection.

Face Recognition Library: A powerful face recognition library based on deep learning for encoding and matching faces.

TensorFlow/Keras: Used for loading the deep learning model that predicts emotions from facial expressions.

MySQL: A relational database system used to store the recognized users' names, emotions, and timestamps.

Haar Cascade Classifiers: Pre-trained models used to detect features like faces, eyes, and smiles in the video feed.

Emotion Categories:

The emotion detection system can identify seven basic emotions:

1.Angry

2.Disgust

3.Fear

4.Happy

5.Neutral

6.Sad

7.Surprise


System Workflow:

Video Capture: The system captures the live video feed from a webcam.

Face Detection: It detects faces in each frame using face recognition and Haar cascades.

Face Recognition:Known face encodings are loaded from a file (.pkl file) containing the encodings of pre-trained faces.For every detected face, the system compares it to the known faces and attempts to match it.If a match is found, the user's name is displayed; otherwise, the face is labeled as "Unknown."

Emotion Detection:Once a face is detected, the emotion model analyzes the facial expression and predicts one of the seven possible emotions.The predicted emotion is then displayed on the screen.

Database Logging:Every time a face is recognized, the system logs the username, emotion, and time of detection into the MySQL database.If the face is not recognized, it is labeled "Unknown."

Web Interface:The system serves a live video stream via a web browser using Flask.Users can access the live feed at localhost:5000 to monitor the real-time face recognition and emotion detection.

Setup Procedure:

1.Download the zip file and extract it in desired location.

2.Install the required packages using pip install -r requirements.txt.

3.Setup up the database by importing the sql file in the MySQL(phpMyAdmin) through Xampp Control Panel.

4.Run the application using "python FACE AND EMOITION ANALYSIS.py" in the command line.
