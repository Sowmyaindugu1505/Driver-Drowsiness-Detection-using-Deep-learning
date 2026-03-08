import json
from pathlib import Path

import cv2
import numpy as np
from pygame import mixer
from tensorflow.keras.models import load_model

from app_config import (
    ALARM_PATH,
    EYE_MODEL_PATH,
    YAWN_CLASS_MAP_PATH,
    YAWN_MODEL_PATH,
)

# Thresholds (simple and tunable)
EYE_DROWSY_SCORE_THRESHOLD = 15
YAWN_PROBABILITY_THRESHOLD = 0.7
YAWN_CONSECUTIVE_FRAMES = 8
YAWN_EVENT_LIMIT = 3


def load_yawn_class_index() -> int:
    if not YAWN_CLASS_MAP_PATH.exists():
        return 1

    mapping = json.loads(YAWN_CLASS_MAP_PATH.read_text())
    positive_aliases = {"yawn", "yawning", "mouth_open"}
    negative_aliases = {"no_yawn", "not_yawn", "non_yawn", "normal"}

    # 1) Prefer exact positive labels only.
    for class_name, index in mapping.items():
        normalized = class_name.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized in positive_aliases:
            return int(index)

    # 2) If exact positive label is unavailable, select whichever class is not a known negative.
    for class_name, index in mapping.items():
        normalized = class_name.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized not in negative_aliases:
            return int(index)

    # 3) Final fallback for old model mappings.
    return 1


def main() -> None:
    mixer.init()
    sound = mixer.Sound(str(ALARM_PATH))

    eye_model = load_model(str(EYE_MODEL_PATH))
    yawn_model = load_model(str(YAWN_MODEL_PATH))
    yawn_class_idx = load_yawn_class_index()

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    cap = cv2.VideoCapture(0)

    eye_score = 0
    yawn_count = 0
    yawn_frames = 0
    yawn_in_progress = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y : y + h, x : x + w]

            eyes = eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) > 0:
                ex, ey, ew, eh = eyes[0]
                eye = cv2.resize(roi_gray[ey : ey + eh, ex : ex + ew], (24, 24)).astype("float32") / 255.0
                eye_state = int(np.argmax(eye_model.predict(eye.reshape(1, 24, 24, 1), verbose=0)))
                if eye_state == 0:
                    eye_score += 1
                    cv2.putText(frame, "Eye: Closed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    eye_score = max(eye_score - 1, 0)
                    cv2.putText(frame, "Eye: Open", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            mouth = gray[y + h // 2 : y + h, x : x + w]
            if mouth.size > 0:
                mouth_img = cv2.resize(mouth, (64, 64)).astype("float32") / 255.0
                probs = yawn_model.predict(mouth_img.reshape(1, 64, 64, 1), verbose=0)[0]
                yawn_prob = float(probs[yawn_class_idx])

                if yawn_prob >= YAWN_PROBABILITY_THRESHOLD:
                    yawn_frames += 1
                    if yawn_frames >= YAWN_CONSECUTIVE_FRAMES and not yawn_in_progress:
                        yawn_count += 1
                        yawn_in_progress = True
                else:
                    yawn_frames = 0
                    yawn_in_progress = False

                cv2.putText(frame, f"Yawn Prob: {yawn_prob:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            break

        cv2.putText(frame, f"Eye Score: {eye_score}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Yawn Count: {yawn_count}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        is_drowsy = eye_score > EYE_DROWSY_SCORE_THRESHOLD or yawn_count >= YAWN_EVENT_LIMIT
        if is_drowsy:
            sound.play()
            cv2.putText(frame, "DROWSY ALERT!", (100, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        cv2.imshow("Driver Drowsiness Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
