from pathlib import Path

import cv2
import numpy as np
from pygame import mixer
from tensorflow.keras.models import load_model

BASE_DIR = Path(__file__).resolve().parent
EYE_MODEL_PATH = BASE_DIR / "models" / "cnnCat2.h5"
YAWN_MODEL_PATH = BASE_DIR / "models" / "yawn_cnn.h5"
ALARM_PATH = BASE_DIR / "alarm.wav"

# Eye drowsiness thresholds
EYE_DROWSY_SCORE_THRESHOLD = 15

# Yawn thresholds
YAWN_PROBABILITY_THRESHOLD = 0.7
YAWN_CONSECUTIVE_FRAMES = 8
YAWN_EVENT_LIMIT = 3
YAWN_CLASS_INDEX = 1  # train_yawn.py class names: no_yawn=0, yawn=1

# Initialize alarm
mixer.init()
sound = mixer.Sound(str(ALARM_PATH))

# Load trained models
eye_model = load_model(str(EYE_MODEL_PATH))
yawn_model = load_model(str(YAWN_MODEL_PATH))

# Load Haar cascades
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Start webcam
cap = cv2.VideoCapture(0)

eye_score = 0
yawn_count = 0
current_yawn_frames = 0
yawn_in_progress = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # --- Eye detection ---
        roi_gray = gray[y : y + h, x : x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            eye = roi_gray[ey : ey + eh, ex : ex + ew]
            eye = cv2.resize(eye, (24, 24))
            eye = eye / 255.0
            eye = eye.reshape(1, 24, 24, 1)

            prediction = eye_model.predict(eye, verbose=0)
            eye_state = int(np.argmax(prediction))

            if eye_state == 0:  # closed
                eye_score += 1
                cv2.putText(
                    frame,
                    "Eye: Closed",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )
            else:
                eye_score = max(eye_score - 1, 0)
                cv2.putText(
                    frame,
                    "Eye: Open",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
            break

        # --- Yawn detection ---
        mouth_roi = gray[y + h // 2 : y + h, x : x + w]
        if mouth_roi.size > 0:
            mouth_img = cv2.resize(mouth_roi, (64, 64))
            mouth_img = mouth_img / 255.0
            mouth_img = mouth_img.reshape(1, 64, 64, 1)

            yawn_pred = yawn_model.predict(mouth_img, verbose=0)[0]
            yawn_prob = float(yawn_pred[YAWN_CLASS_INDEX])

            if yawn_prob >= YAWN_PROBABILITY_THRESHOLD:
                current_yawn_frames += 1
            else:
                yawn_in_progress = False
                current_yawn_frames = 0

            if current_yawn_frames >= YAWN_CONSECUTIVE_FRAMES and not yawn_in_progress:
                yawn_count += 1
                yawn_in_progress = True

            cv2.putText(
                frame,
                f"Yawn Prob: {yawn_prob:.2f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
            )

        break

    cv2.putText(
        frame,
        f"Eye Score: {eye_score}",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"Yawn Count: {yawn_count}",
        (10, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    is_drowsy = eye_score > EYE_DROWSY_SCORE_THRESHOLD or yawn_count >= YAWN_EVENT_LIMIT
    if is_drowsy:
        sound.play()
        cv2.putText(
            frame,
            "DROWSY ALERT!",
            (100, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 0, 255),
            3,
        )

    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
