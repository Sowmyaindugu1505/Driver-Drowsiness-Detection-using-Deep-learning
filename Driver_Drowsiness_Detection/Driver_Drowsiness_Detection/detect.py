import cv2
import numpy as np
from tensorflow.keras.models import load_model
from pygame import mixer

# Initialize alarm
mixer.init()
sound = mixer.Sound("alarm.wav")

# Load trained model
model = load_model("models/cnnCat2.h5")

# Load Haar cascades
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

# Start webcam
cap = cv2.VideoCapture(0)

score = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        roi_gray = gray[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            eye = roi_gray[ey:ey + eh, ex:ex + ew]
            eye = cv2.resize(eye, (24, 24))
            eye = eye / 255.0
            eye = eye.reshape(1, 24, 24, 1)

            prediction = model.predict(eye, verbose=0)
            state = np.argmax(prediction)

            if state == 0:  # closed
                score += 1
                cv2.putText(frame, "Closed", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:  # open
                score = max(score - 1, 0)
                cv2.putText(frame, "Open", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            break  # check only one eye

    if score > 15:
        sound.play()
        cv2.putText(frame, "DROWSY!", (100, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
