from flask import Flask, render_template, Response, redirect, url_for
import cv2
import numpy as np
import face_recognition
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from collections import deque
import pickle
import mysql.connector
from mysql.connector import Error
import datetime

app = Flask(__name__)

def load_known_faces(filename):
    with open(filename, 'rb') as f:
        known_face_encodings, known_face_names = pickle.load(f)
    return known_face_encodings, known_face_names

def load_emotion_model(model_path):
    return load_model(model_path)


# Load the known face encodings and names
encodeListKnown, classNames = load_known_faces('DOCUMENTS/FACE_MODEL.pkl')
print(len(encodeListKnown))
print('Encoding Complete')

# Load the classifiers and emotion model
face_classifier = cv2.CascadeClassifier('DOCUMENTS/haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier('DOCUMENTS/haarcascade_eye.xml')
smile_classifier = cv2.CascadeClassifier('DOCUMENTS/haarcascade_smile.xml')
emotion_model = load_emotion_model('DOCUMENTS/EMOTION_MODEL.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def generate_frames():
    cap = cv2.VideoCapture(0)
    start_time = datetime.datetime.now()

    while True:
        success, img = cap.read()
        if not success:
            break
        else:
            imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

            facesCurFrame = face_recognition.face_locations(imgS)
            encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
            username_stored = False

            for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                matchIndex = np.argmin(faceDis)

                if matches[matchIndex]:
                    name = f"{classNames[matchIndex].upper()}"
                else:
                    name = "UNKNOWN"

                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (36, 255, 12), 2)
                cv2.putText(img, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                if name != "UNKNOWN" and not username_stored:
                    elapsed_time = datetime.datetime.now() - start_time

                    if elapsed_time.total_seconds() >= 10:
                        print("10 seconds elapsed. Stopping recognition.")
                        username_stored = True
                        break

                    roi_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    roi_gray = roi_gray[y1:y2, x1:x2]
                    roi_gray = cv2.resize(roi_gray, (48, 48))
                    roi_gray = roi_gray / 255.0
                    roi_gray = np.reshape(roi_gray, (1, 48, 48, 1))

                    prediction = emotion_model.predict(roi_gray)
                    label = emotion_labels[np.argmax(prediction)]
                    cv2.putText(img, label, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 95, 255), 2)
                    
                    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    try:
                        connection = mysql.connector.connect(
                            host='localhost',
                            database='facerecognition',
                            user='root',
                            password=''
                        )
                        mySql_insert_query = ("""INSERT INTO user(username, emotion, time) 
                                                VALUES(%s, %s, %s)""", 
                                              (name, label, current_time,))

                        cursor = connection.cursor()
                        cursor.execute(*mySql_insert_query)
                        connection.commit()
                        cursor.close()
                        username_stored = True

                    except mysql.connector.Error as error:
                        print("Failed to insert record into user table {}".format(error))

                    finally:
                        if connection.is_connected():
                            connection.close()
                            break

                roi_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
                roi_gray_face = roi_gray[y1:y2, x1:x2]  
                roi_gray_face = cv2.equalizeHist(roi_gray_face)  
                eyes = eye_classifier.detectMultiScale(roi_gray_face, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(img, (x1 + ex, y1 + ey), (x1 + ex + ew, y1 + ey + eh), (36, 255, 12), 2)
                
                smiles = smile_classifier.detectMultiScale(roi_gray_face, scaleFactor=1.8, minNeighbors=20)
                for (sx, sy, sw, sh) in smiles:
                    cv2.rectangle(img, (x1 + sx, y1 + sy), (x1 + sx + sw, y1 + sy + sh), (36, 255, 12), 2)

            # Convert the frame to a byte array
            ret, buffer = cv2.imencode('.jpg', img)
            img_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img_bytes + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
