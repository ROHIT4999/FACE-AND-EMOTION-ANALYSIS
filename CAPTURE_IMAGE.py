import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QLineEdit, QMessageBox
from PyQt5.QtCore import Qt
import cv2
import os
import mysql.connector
import datetime

class CaptureImagesWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("CAPTURE IMAGE")
        self.setGeometry(100, 100, 400, 200)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.user_id_label = QLabel("ENTER USER NAME:")
        self.layout.addWidget(self.user_id_label)

        self.user_id_input = QLineEdit()
        self.layout.addWidget(self.user_id_input)

        self.capture_button = QPushButton("Continue")
        self.capture_button.clicked.connect(self.capture_images)
        self.layout.addWidget(self.capture_button)

        self.quit_button = QPushButton("Quit")
        self.quit_button.clicked.connect(self.close)
        self.layout.addWidget(self.quit_button)

        self.used_names = set()
        self.db_config = {
            "host": "localhost",
            "user": "root",
            "password": "",
            "database": "facerecognition"   
        }

    def capture_images(self):
        user_id = self.user_id_input.text().strip()
        if not user_id:
            QMessageBox.warning(self, "WARNING", "Please enter your name.")
            return

        if user_id.upper() in self.used_names:
            QMessageBox.warning(self, "WARNING", "Name already captured. Please enter a different name.")
            return

        capture_successful = capture_user_images(user_id)
        if capture_successful:
            self.used_names.add(user_id.upper())
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            try:

                connection = mysql.connector.connect(**self.db_config)
                cursor = connection.cursor()
                sql = "INSERT INTO users (username,time) VALUES (%s,%s)"
                cursor.execute(sql, (user_id.upper(),current_time,))
                connection.commit()
                QMessageBox.information(self, "SUCCESS", "User image captured successfully.")

            except mysql.connector.Error as err:
                print("Error connecting to database:", err)
                QMessageBox.warning(self, "WARNING", "Failed to save user data to database.")

            finally:
                if connection:
                    connection.close()
                    cursor.close()

            response = QMessageBox.question(self, "ADD USERS", "Do you want to add more user face?", QMessageBox.Yes | QMessageBox.No)
            if response == QMessageBox.Yes:
                self.user_id_input.clear()
                return  
            else:
                QMessageBox.information(self, "SUCCESS", "User images captured successfully.")
                self.close()  
        else:
            QMessageBox.warning(self, "WARNING", "Name already captured. Please enter a different name.")

def capture_user_images(user_id):
    if user_id.upper() + '.jpg' in os.listdir('DATASET'):
        print("Name already captured. Please enter a different name.")
        return False

    video = cv2.VideoCapture(0)  
    if not video.isOpened():
        print("Error opening video capture.")
        return False

    facedetect = cv2.CascadeClassifier('DOCUMENTS/haarcascade_frontalface_default.xml')
    if facedetect.empty():
        print("Error loading face detection model.")
        return False

    count = 0

    while True:
        ret, frame = video.read()
        if not ret:
            print("Error reading frame from video capture.")
            return False

        faces = facedetect.detectMultiScale(frame, 1.3, 10)

        for x, y, w, h in faces:
            count += 1
            name = os.path.join('DATASET', user_id.upper() + '.jpg')
            print("Creating Image: {}".format(name))
            cv2.imwrite(name, frame[y:y+h, x:x+w])  
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

        cv2.imshow("CAPTURING IMAGE", frame)
        if cv2.waitKey(1) == ord('q') or count >= 1:  
            break

    video.release()
    cv2.destroyAllWindows()
    return True

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CaptureImagesWindow()
    window.show()
    sys.exit(app.exec_())
