import cv2
import numpy as np
from keras.models import model_from_json
from selenium import webdriver

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Load json and create model
json_file = open('emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# Load weights into the model
emotion_model.load_weights("emotion_model.h5")
print("Loaded model from disk")

# Start the webcam feed
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error opening video capture")

# Create a global variable to store the detected emotion
detected_emotion = ""

while True:
    # Read frame from the video capture
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    if not ret:
        break
    face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # Process each detected face
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # Predict the emotions
        emotion_prediction = emotion_model.predict(cropped_img)
        max_index = int(np.argmax(emotion_prediction))
        detected_emotion = emotion_dict[max_index]
        cv2.putText(frame, detected_emotion, (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close the webcam
cap.release()
cv2.destroyAllWindows()

# Use the detected emotion to recommend a video
if detected_emotion in ['Angry', 'Disgusted', 'Fearful']:
    browser = webdriver.Chrome(r'C:\webdriver\chromedriver.exe')
    browser.get('https://www.youtube.com/watch?v=RCAj8_wylKw')
elif detected_emotion in ['Happy', 'Neutral', 'Surprised']:
    browser = webdriver.Chrome(r'C:\webdriver\chromedriver.exe')
    browser.get('https://youtu.be/pRbxlpvXw2s?t=121')
elif detected_emotion == 'Sad':
    browser = webdriver.Chrome(r'C:\webdriver\chromedriver.exe')
    browser.get('https://www.youtube.com/watch?v=SBWYGGDYmhg&list=PLHuHXHyLu7BGi-vR7X6j_xh_Tt9wy7pNA')

# Wait for the user to close the browser
input("Press Enter to close the browser...")

# Quit the browser
browser.quit
