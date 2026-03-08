import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
from pygame import mixer
from tensorflow.keras.models import load_model

from src.config import (
    ALARM_PATH,
    EYE_CLOSED_CLASS_INDEX,
    EYE_DATA_DIR,
    EYE_DROWSY_SECONDS_THRESHOLD,
    EYE_FULLY_CLOSED_PROBABILITY_THRESHOLD,
    EYE_MODEL_PATH,
    FACE_CASCADE_PATH,
    LEFT_EYE_CASCADE_PATH,
    RIGHT_EYE_CASCADE_PATH,
    YAWN_CLASS_MAP_PATH,
    YAWN_END_PROBABILITY_THRESHOLD,
    YAWN_EVENT_LIMIT,
    YAWN_MIN_GAP_SECONDS,
    YAWN_MODEL_PATH,
    YAWN_OPEN_SECONDS_THRESHOLD,
    YAWN_PROBABILITY_THRESHOLD,
    YAWN_RELEASE_SECONDS_THRESHOLD,
)


def load_yawn_class_index() -> int:
    if not YAWN_CLASS_MAP_PATH.exists():
        print("[Warning] Yawn class map not found, defaulting to index 1.")
        return 1

    try:
        mapping = json.loads(YAWN_CLASS_MAP_PATH.read_text())
        positive_aliases = {"yawn", "yawning", "mouth_open"}
        negative_aliases = {"no_yawn", "not_yawn", "non_yawn", "normal"}

        for class_name, index in mapping.items():
            normalized = class_name.strip().lower().replace("-", "_").replace(" ", "_")
            if normalized in positive_aliases:
                return int(index)

        for class_name, index in mapping.items():
            normalized = class_name.strip().lower().replace("-", "_").replace(" ", "_")
            if normalized not in negative_aliases:
                return int(index)
    except Exception as e:
        print(f"[Error] Failed to parse {YAWN_CLASS_MAP_PATH}: {e}")

    return 1


class DrowsinessDetector:
    def __init__(self):
        self.eye_closed_duration = 0.0
        self.yawn_count = 0
        self.yawn_open_duration = 0.0
        self.yawn_release_duration = 0.0
        self.yawn_in_progress = False
        self.yawn_prob_ema = 0.0
        self.last_yawn_ts = 0.0
        self.calibrating = True
        self.calibration_samples = 0
        self.eye_calib_vals = []
        self.yawn_calib_vals = []
        self.eye_closed_threshold = EYE_FULLY_CLOSED_PROBABILITY_THRESHOLD
        self.yawn_start_threshold = YAWN_PROBABILITY_THRESHOLD
        self.yawn_end_threshold = YAWN_END_PROBABILITY_THRESHOLD

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
        self.eye_model = None
        self.yawn_model = None

        if EYE_MODEL_PATH.exists():
            self.eye_model = load_model(str(EYE_MODEL_PATH))
            self.eye_closed_class_idx = self._infer_eye_closed_class_index()
        else:
            print(f"[Warning] Eye model missing: {EYE_MODEL_PATH}. Eye-based drowsiness will be disabled.")
            self.eye_closed_class_idx = EYE_CLOSED_CLASS_INDEX

        if YAWN_MODEL_PATH.exists():
            self.yawn_model = load_model(str(YAWN_MODEL_PATH))
        else:
            print(f"[Warning] Yawn model missing: {YAWN_MODEL_PATH}. Yawn-based drowsiness will be disabled.")

        if self.eye_model is None and self.yawn_model is None:
            raise FileNotFoundError("No trained models found in models/. Train at least one target first.")

        self.yawn_class_idx = load_yawn_class_index()
        print("Models loaded successfully.")

    def _infer_eye_closed_class_index(self) -> int:
        """Infer eye class index from available train/open and train/closed samples."""
        try:
            open_dir = EYE_DATA_DIR / "train" / "open"
            closed_dir = EYE_DATA_DIR / "train" / "closed"
            if not open_dir.exists() or not closed_dir.exists():
                return EYE_CLOSED_CLASS_INDEX

            open_files = [p for p in open_dir.glob("*") if p.is_file()]
            closed_files = [p for p in closed_dir.glob("*") if p.is_file()]
            if len(open_files) < 10 or len(closed_files) < 10:
                return EYE_CLOSED_CLASS_INDEX

            open_pick = random.sample(open_files, min(40, len(open_files)))
            closed_pick = random.sample(closed_files, min(40, len(closed_files)))

            def avg_probs(files):
                vals = []
                for p in files:
                    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    eye = cv2.resize(img, (24, 24)).astype("float32") / 255.0
                    eye = eye.reshape(1, 24, 24, 1)
                    vals.append(self.eye_model.predict(eye, verbose=0)[0])
                if not vals:
                    return np.array([0.5, 0.5], dtype=np.float32)
                return np.mean(np.array(vals), axis=0)

            open_mean = avg_probs(open_pick)
            closed_mean = avg_probs(closed_pick)
            # Pick class with larger closed-vs-open separation.
            idx = int(np.argmax(closed_mean - open_mean))
            print(f"[Info] Eye class mapping inferred. closed_idx={idx}, open_mean={open_mean}, closed_mean={closed_mean}")
            return idx
        except Exception as e:
            print(f"[Warning] Could not infer eye class mapping: {e}. Using default index.")
            return EYE_CLOSED_CLASS_INDEX

    def _init_cascades(self):
        if (
            not FACE_CASCADE_PATH.exists()
            or not LEFT_EYE_CASCADE_PATH.exists()
            or not RIGHT_EYE_CASCADE_PATH.exists()
        ):
            raise FileNotFoundError("Haar cascades missing. Please check the haarcascades folder.")

        self.face_cascade = cv2.CascadeClassifier(str(FACE_CASCADE_PATH))
        self.left_eye_cascade = cv2.CascadeClassifier(str(LEFT_EYE_CASCADE_PATH))
        self.right_eye_cascade = cv2.CascadeClassifier(str(RIGHT_EYE_CASCADE_PATH))

    def play_alarm(self):
        if self.sound:
            self.sound.play()

    def predict_eye_closed_probability(self, roi_gray, ex, ey, ew, eh):
        if self.eye_model is None:
            return 0.0
        eye = cv2.resize(roi_gray[ey : ey + eh, ex : ex + ew], (24, 24)).astype("float32") / 255.0
        eye = eye.reshape(1, 24, 24, 1)
        probs = self.eye_model.predict(eye, verbose=0)[0]
        return float(probs[self.eye_closed_class_idx])

    def process_yawn(self, frame, mouth_gray, dt: float):
        if self.yawn_model is None:
            return
        if mouth_gray.size == 0:
            return

        mouth_img = cv2.resize(mouth_gray, (64, 64)).astype("float32") / 255.0
        mouth_img = mouth_img.reshape(1, 64, 64, 1)

        probs = self.yawn_model.predict(mouth_img, verbose=0)[0]
        yawn_prob = float(probs[self.yawn_class_idx])

        # Responsive smoothing
        self.yawn_prob_ema = 0.6 * self.yawn_prob_ema + 0.4 * yawn_prob

        if self.yawn_prob_ema >= self.yawn_start_threshold:
            self.yawn_open_duration += dt
            self.yawn_release_duration = 0.0
            now = time.time()
            if (
                self.yawn_open_duration >= YAWN_OPEN_SECONDS_THRESHOLD
                and not self.yawn_in_progress
                and (now - self.last_yawn_ts) >= YAWN_MIN_GAP_SECONDS
            ):
                self.yawn_count += 1
                self.yawn_in_progress = True
                self.last_yawn_ts = now
        else:
            if self.yawn_in_progress:
                if self.yawn_prob_ema <= self.yawn_end_threshold:
                    self.yawn_release_duration += dt
                else:
                    self.yawn_release_duration = 0.0
                if self.yawn_release_duration >= YAWN_RELEASE_SECONDS_THRESHOLD:
                    self.yawn_in_progress = False
                    self.yawn_open_duration = 0.0
                    self.yawn_release_duration = 0.0
            else:
                self.yawn_open_duration = 0.0

        yawn_state = "Yawning" if self.yawn_open_duration >= YAWN_OPEN_SECONDS_THRESHOLD else "NotYawning"
        cv2.putText(frame, f"Yawn: {yawn_state}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[Error] Could not open webcam.")
            return

        print("Starting Driver Drowsiness Detection. Press 'q' to quit.")
        prev_ts = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            now_ts = time.time()
            dt = max(0.0, now_ts - prev_ts)
            prev_ts = now_ts

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi_gray = gray[y : y + h, x : x + w]
                upper_face = roi_gray[: int(h * 0.60), :]

                eyes_left = self.left_eye_cascade.detectMultiScale(
                    upper_face,
                    scaleFactor=1.1,
                    minNeighbors=6,
                    minSize=(18, 18),
                )
                eyes_right = self.right_eye_cascade.detectMultiScale(
                    upper_face,
                    scaleFactor=1.1,
                    minNeighbors=6,
                    minSize=(18, 18),
                )
                eyes = list(eyes_left) + list(eyes_right)
                eyes = sorted(eyes, key=lambda b: b[2] * b[3], reverse=True)[:2]

                eye_prob_now = None
                if len(eyes) >= 2 and self.eye_model is not None:
                    closed_probs = []
                    for (ex, ey, ew, eh) in eyes:
                        closed_prob = self.predict_eye_closed_probability(upper_face, ex, ey, ew, eh)
                        is_closed = closed_prob >= self.eye_closed_threshold
                        closed_probs.append(closed_prob)

                    eyes_closed_now = all(p >= self.eye_closed_threshold for p in closed_probs)
                    eye_prob_now = float(np.mean(closed_probs))
                    if eyes_closed_now:
                        self.eye_closed_duration += dt
                    else:
                        self.eye_closed_duration = 0.0

                    eye_label = "Eye: Closed" if eyes_closed_now else "Eye: Open"
                    eye_color = (0, 0, 255) if eyes_closed_now else (0, 255, 0)
                    cv2.putText(frame, eye_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, eye_color, 2)
                elif self.eye_model is not None:
                    self.eye_closed_duration = 0.0
                    cv2.putText(frame, "Eye: Open", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                mouth_top = y + int(h * 0.50)
                mouth_bottom = y + h
                mouth_left = x + int(w * 0.05)
                mouth_right = x + int(w * 0.95)
                mouth_gray = gray[mouth_top:mouth_bottom, mouth_left:mouth_right]
                if mouth_gray.size > 0 and self.yawn_model is not None:
                    mouth_img = cv2.resize(mouth_gray, (64, 64)).astype("float32") / 255.0
                    mouth_img = mouth_img.reshape(1, 64, 64, 1)
                    yawn_now = float(self.yawn_model.predict(mouth_img, verbose=0)[0][self.yawn_class_idx])
                else:
                    yawn_now = None

                # 2-second warm-up calibration to reduce startup false positives.
                if self.calibrating:
                    if eye_prob_now is not None:
                        self.eye_calib_vals.append(eye_prob_now)
                    if yawn_now is not None:
                        self.yawn_calib_vals.append(yawn_now)
                    self.calibration_samples += 1

                    self.eye_closed_duration = 0.0
                    self.yawn_count = 0
                    self.yawn_open_duration = 0.0
                    self.yawn_release_duration = 0.0
                    self.yawn_in_progress = False

                    if self.calibration_samples >= 50:
                        if self.eye_calib_vals:
                            eye_base = float(np.mean(self.eye_calib_vals))
                            self.eye_closed_threshold = min(0.98, max(0.75, eye_base + 0.25))
                        else:
                            self.eye_closed_threshold = EYE_FULLY_CLOSED_PROBABILITY_THRESHOLD
                        if self.yawn_calib_vals:
                            yawn_base = float(np.mean(self.yawn_calib_vals))
                            self.yawn_start_threshold = min(0.75, max(0.40, yawn_base + 0.12))
                        else:
                            self.yawn_start_threshold = YAWN_PROBABILITY_THRESHOLD
                        self.yawn_end_threshold = max(0.25, self.yawn_start_threshold - 0.10)
                        self.calibrating = False
                        print(
                            f"[Info] Calibration done. "
                            f"eye_thr={self.eye_closed_threshold:.2f}, "
                            f"yawn_start={self.yawn_start_threshold:.2f}, yawn_end={self.yawn_end_threshold:.2f}"
                        )
                self.process_yawn(frame, mouth_gray, dt)

                break

            cv2.putText(frame, f"Yawn Count: {self.yawn_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            eye_alert = self.eye_model is not None and self.eye_closed_duration >= EYE_DROWSY_SECONDS_THRESHOLD
            yawn_alert = self.yawn_model is not None and self.yawn_count >= YAWN_EVENT_LIMIT
            is_drowsy = eye_alert or yawn_alert
            if is_drowsy:
                self.play_alarm()
                cv2.putText(frame, "DROWSY ALERT!", (100, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)

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
