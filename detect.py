import json
from pathlib import Path

import cv2
import numpy as np
from pygame import mixer
from tensorflow.keras.models import load_model

from config import (
    ALARM_PATH,
    EYE_CASCADE_PATH,
    EYE_DROWSY_SCORE_THRESHOLD,
    EYE_MODEL_PATH,
    FACE_CASCADE_PATH,
    YAWN_CLASS_MAP_PATH,
    YAWN_CONSECUTIVE_FRAMES,
    YAWN_EVENT_LIMIT,
    YAWN_MODEL_PATH,
    YAWN_PROBABILITY_THRESHOLD,
)


def load_yawn_class_index() -> int:
    """Dynamically find the model index for 'yawn' class."""
    if not YAWN_CLASS_MAP_PATH.exists():
        print("[Warning] Yawn class map not found, defaulting to index 1.")
        return 1

    try:
        mapping = json.loads(YAWN_CLASS_MAP_PATH.read_text())
        positive_aliases = {"yawn", "yawning", "mouth_open"}
        negative_aliases = {"no_yawn", "not_yawn", "non_yawn", "normal"}

        # Prefer exact positive labels
        for class_name, index in mapping.items():
            normalized = class_name.strip().lower().replace("-", "_").replace(" ", "_")
            if normalized in positive_aliases:
                return int(index)
        
        # Fallback to anything non-negative
        for class_name, index in mapping.items():
            normalized = class_name.strip().lower().replace("-", "_").replace(" ", "_")
            if normalized not in negative_aliases:
                return int(index)
    except Exception as e:
        print(f"[Error] Failed to parse {YAWN_CLASS_MAP_PATH}: {e}")

    return 1


class DrowsinessDetector:
    def __init__(self):
        # State tracking
        self.eye_score = 0
        self.yawn_count = 0
        self.yawn_frames = 0
        self.yawn_in_progress = False

        self._init_audio()
        self._init_models()
        self._init_cascades()

    def _init_audio(self):
        try:
            mixer.init()
            if not ALARM_PATH.exists():
                raise FileNotFoundError(f"Alarm sound missing at {ALARM_PATH}")
            self.sound = mixer.Sound(str(ALARM_PATH))
        except Exception as e:
            print(f"[Warning] Audio initialization failed: {e}. Alarms will be visual only.")
            self.sound = None

    def _init_models(self):
        print("Loading models (this might take a few seconds)...")
        if not EYE_MODEL_PATH.exists():
            raise FileNotFoundError(f"Eye model missing: {EYE_MODEL_PATH}. Run train.py --target eyes")
        if not YAWN_MODEL_PATH.exists():
            raise FileNotFoundError(f"Yawn model missing: {YAWN_MODEL_PATH}. Run train.py --target yawns")

        self.eye_model = load_model(str(EYE_MODEL_PATH))
        self.yawn_model = load_model(str(YAWN_MODEL_PATH))
        self.yawn_class_idx = load_yawn_class_index()
        print("Models loaded successfully.")

    def _init_cascades(self):
        if not FACE_CASCADE_PATH.exists() or not EYE_CASCADE_PATH.exists():
            raise FileNotFoundError("Haar cascades missing. Please check the haarcascades folder.")
        
        self.face_cascade = cv2.CascadeClassifier(str(FACE_CASCADE_PATH))
        self.eye_cascade = cv2.CascadeClassifier(str(EYE_CASCADE_PATH))

    def play_alarm(self):
        """Trigger the alarm sound if available."""
        if self.sound:
            self.sound.play()

    def process_eye(self, frame, roi_gray, ex, ey, ew, eh):
        """Extract eye, predict state, and track drowsiness score."""
        eye = cv2.resize(roi_gray[ey : ey + eh, ex : ex + ew], (24, 24)).astype("float32") / 255.0
        eye = eye.reshape(1, 24, 24, 1)
        
        eye_state = int(np.argmax(self.eye_model.predict(eye, verbose=0)))
        
        if eye_state == 0:  # Closed
            self.eye_score += 1
            cv2.putText(frame, "Eye: Closed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:  # Open
            self.eye_score = max(self.eye_score - 1, 0)
            cv2.putText(frame, "Eye: Open", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    def process_yawn(self, frame, mouth_gray):
        """Extract mouth, predict probability, and track yawn counts."""
        if mouth_gray.size == 0:
            return

        mouth_img = cv2.resize(mouth_gray, (64, 64)).astype("float32") / 255.0
        mouth_img = mouth_img.reshape(1, 64, 64, 1)

        probs = self.yawn_model.predict(mouth_img, verbose=0)[0]
        yawn_prob = float(probs[self.yawn_class_idx])

        if yawn_prob >= YAWN_PROBABILITY_THRESHOLD:
            self.yawn_frames += 1
            if self.yawn_frames >= YAWN_CONSECUTIVE_FRAMES and not self.yawn_in_progress:
                self.yawn_count += 1
                self.yawn_in_progress = True
        else:
            self.yawn_frames = 0
            self.yawn_in_progress = False

        cv2.putText(frame, f"Yawn Prob: {yawn_prob:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[Error] Could not open webcam.")
            return

        print("Starting Driver Drowsiness Detection. Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi_gray = gray[y : y + h, x : x + w]

                # 1. Detect and Process Eyes
                eyes = self.eye_cascade.detectMultiScale(roi_gray)
                if len(eyes) > 0:
                    ex, ey, ew, eh = eyes[0]
                    self.process_eye(frame, roi_gray, ex, ey, ew, eh)

                # 2. Detect and Process Mouth/Yawn
                # Estimate mouth region as the bottom half of the face bounding box.
                mouth_gray = gray[y + h // 2 : y + h, x : x + w]
                self.process_yawn(frame, mouth_gray)
                
                # Only process the primary face detected
                break

            # 3. Draw Global State Metrics
            cv2.putText(frame, f"Eye Score: {self.eye_score}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Yawn Count: {self.yawn_count}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # 4. Trigger Alarm
            is_drowsy = self.eye_score > EYE_DROWSY_SCORE_THRESHOLD or self.yawn_count >= YAWN_EVENT_LIMIT
            if is_drowsy:
                self.play_alarm()
                cv2.putText(frame, "DROWSY ALERT!", (100, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

            # Render display
            cv2.imshow("Driver Drowsiness Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


def main():
    try:
        detector = DrowsinessDetector()
        detector.run()
    except Exception as e:
        print(f"\n[Fatal Error] {e}")


if __name__ == "__main__":
    main()
