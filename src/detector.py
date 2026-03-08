"""
src/detector.py  –  Live webcam drowsiness detector.

Key design decisions
────────────────────
1.  Time-based state machine (dt per frame via time.time()).
2.  Non-blocking audio: pygame.mixer.Sound.play() returns immediately.
3.  Calibration phase (first 50 face samples) suppresses false positives
    on startup by tuning per-session thresholds.
4.  process_yawn() is a clean, self-contained method with hysteresis
    (start / end thresholds) to eliminate flicker.
5.  Visual text is driven ONLY by accumulated durations / counts, never
    by raw per-frame predictions → no flicker.

Business logic (matches requirements exactly)
─────────────────────────────────────────────
Eyes
  • Display "Eyes: Closed" (red)  ←→  eye_closed_duration >= 0.3 s
  • Display "Eyes: Open"  (green) ←→  eye_closed_duration <  0.3 s
  • DROWSINESS ALERT + alarm       ←→  eye_closed_duration >= 3.0 s

Yawn
  • Display "Yawning"              ←→  yawn_open_duration  >= 0.25 s
  • Display "Not Yawning"          ←→  yawn_open_duration  <  0.25 s
  • Increment yawn_count           ←→  valid yawn completed
  • DROWSINESS ALERT + alarm       ←→  yawn_count          >= 2
"""

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
    EYE_CLOSED_DISPLAY_SECONDS,        # 0.30 s  – show "Closed" text
    EYE_DATA_DIR,
    EYE_DROWSY_SECONDS_THRESHOLD,      # 3.00 s  – trigger alarm
    EYE_FULLY_CLOSED_PROBABILITY_THRESHOLD,
    EYE_MODEL_PATH,
    FACE_CASCADE_PATH,
    LEFT_EYE_CASCADE_PATH,
    RIGHT_EYE_CASCADE_PATH,
    YAWN_CLASS_MAP_PATH,
    YAWN_END_PROBABILITY_THRESHOLD,
    YAWN_EVENT_LIMIT,                  # 2 yawns – trigger alarm
    YAWN_MIN_GAP_SECONDS,
    YAWN_MODEL_PATH,
    YAWN_OPEN_DISPLAY_SECONDS,         # 0.25 s  – show "Yawning" text
    YAWN_OPEN_SECONDS_THRESHOLD,       # 0.25 s  – count a yawn event
    YAWN_PROBABILITY_THRESHOLD,
    YAWN_RELEASE_SECONDS_THRESHOLD,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: resolve the "yawn-positive" class index from the saved JSON map
# ─────────────────────────────────────────────────────────────────────────────
def load_yawn_class_index() -> int:
    """Return the index of the 'yawning / mouth open' class from the saved map.

    Falls back to 1 if the file is absent or the class names are unrecognised.
    """
    if not YAWN_CLASS_MAP_PATH.exists():
        print("[Warning] Yawn class map not found – defaulting yawn_class_idx=1.")
        return 1

    try:
        mapping = json.loads(YAWN_CLASS_MAP_PATH.read_text())
        positive_aliases = {"yawn", "yawning", "mouth_open"}
        negative_aliases = {"no_yawn", "not_yawn", "non_yawn", "normal"}

        # First pass: explicit positive alias
        for class_name, index in mapping.items():
            normalised = class_name.strip().lower().replace("-", "_").replace(" ", "_")
            if normalised in positive_aliases:
                return int(index)

        # Second pass: anything that is not a negative alias
        for class_name, index in mapping.items():
            normalised = class_name.strip().lower().replace("-", "_").replace(" ", "_")
            if normalised not in negative_aliases:
                return int(index)

    except Exception as exc:
        print(f"[Error] Failed to parse {YAWN_CLASS_MAP_PATH}: {exc}")

    return 1


# ─────────────────────────────────────────────────────────────────────────────
# Main detector class
# ─────────────────────────────────────────────────────────────────────────────
class DrowsinessDetector:
    """Real-time driver drowsiness detector.

    State variables (all initialised in __init__)
    ─────────────────────────────────────────────
    eye_closed_duration   : float  – seconds eyes have been continuously closed
    yawn_open_duration    : float  – seconds mouth has been continuously open
    yawn_release_duration : float  – seconds mouth has been below end-threshold
    yawn_in_progress      : bool   – True while an active yawn is being tracked
    yawn_prob_ema         : float  – exponential moving average of yawn probability
    last_yawn_ts          : float  – wall-clock time of the last counted yawn
    yawn_count            : int    – total valid yawns counted this session
    calibrating           : bool   – True during the startup calibration phase
    """

    def __init__(self):
        # ── Time-based state trackers ────────────────────────────────────────
        self.eye_closed_duration   = 0.0   # accumulates dt while eyes closed
        self.yawn_open_duration    = 0.0   # accumulates dt while mouth open
        self.yawn_release_duration = 0.0   # accumulates dt after mouth closes
        self.yawn_in_progress      = False # True between yawn-start and yawn-end
        self.yawn_prob_ema         = 0.0   # smoothed yawn probability
        self.last_yawn_ts          = 0.0   # timestamp of last counted yawn
        self.yawn_count            = 0     # total valid yawns this session

        # ── Calibration ──────────────────────────────────────────────────────
        self.calibrating         = True
        self.calibration_samples = 0
        self.eye_calib_vals      = []
        self.yawn_calib_vals     = []

        # Per-session adaptive thresholds (may be tuned during calibration)
        self.eye_closed_threshold  = EYE_FULLY_CLOSED_PROBABILITY_THRESHOLD
        self.yawn_start_threshold  = YAWN_PROBABILITY_THRESHOLD
        self.yawn_end_threshold    = YAWN_END_PROBABILITY_THRESHOLD

        # ── Sub-system initialisation ─────────────────────────────────────────
        self._init_audio()
        self._init_models()
        self._init_cascades()

    # ──────────────────────────────────────────────────────────────────────────
    # Initialisation helpers
    # ──────────────────────────────────────────────────────────────────────────
    def _init_audio(self):
        """Initialise pygame mixer.  Failure → visual-only alarms."""
        try:
            mixer.init()
            if not ALARM_PATH.exists():
                raise FileNotFoundError(f"Alarm sound missing at {ALARM_PATH}")
            self.sound = mixer.Sound(str(ALARM_PATH))
        except Exception as exc:
            print(f"[Warning] Audio init failed: {exc}. Alarms will be visual only.")
            self.sound = None

    def _init_models(self):
        """Load the trained CNN models from disk."""
        print("Loading models (this might take a few seconds)…")
        self.eye_model  = None
        self.yawn_model = None

        if EYE_MODEL_PATH.exists():
            self.eye_model = load_model(str(EYE_MODEL_PATH))
            self.eye_closed_class_idx = self._infer_eye_closed_class_index()
        else:
            print(f"[Warning] Eye model missing: {EYE_MODEL_PATH}. Eye drowsiness disabled.")
            self.eye_closed_class_idx = EYE_CLOSED_CLASS_INDEX

        if YAWN_MODEL_PATH.exists():
            self.yawn_model = load_model(str(YAWN_MODEL_PATH))
        else:
            print(f"[Warning] Yawn model missing: {YAWN_MODEL_PATH}. Yawn drowsiness disabled.")

        if self.eye_model is None and self.yawn_model is None:
            raise FileNotFoundError(
                "No trained models found in models/. Train at least one target first."
            )

        self.yawn_class_idx = load_yawn_class_index()
        print("Models loaded successfully.")

    def _infer_eye_closed_class_index(self) -> int:
        """Probe the eye CNN with sample images to determine which output index
        corresponds to 'closed'. Falls back to the config default."""
        try:
            open_dir   = EYE_DATA_DIR / "train" / "open"
            closed_dir = EYE_DATA_DIR / "train" / "closed"
            if not open_dir.exists() or not closed_dir.exists():
                return EYE_CLOSED_CLASS_INDEX

            open_files   = [p for p in open_dir.glob("*")   if p.is_file()]
            closed_files = [p for p in closed_dir.glob("*") if p.is_file()]
            if len(open_files) < 10 or len(closed_files) < 10:
                return EYE_CLOSED_CLASS_INDEX

            open_pick   = random.sample(open_files,   min(40, len(open_files)))
            closed_pick = random.sample(closed_files, min(40, len(closed_files)))

            def _avg_probs(files):
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

            open_mean   = _avg_probs(open_pick)
            closed_mean = _avg_probs(closed_pick)
            idx = int(np.argmax(closed_mean - open_mean))
            print(
                f"[Info] Eye class mapping inferred: closed_idx={idx}, "
                f"open_mean={open_mean}, closed_mean={closed_mean}"
            )
            return idx
        except Exception as exc:
            print(f"[Warning] Could not infer eye class mapping: {exc}. Using default.")
            return EYE_CLOSED_CLASS_INDEX

    def _init_cascades(self):
        """Load Haar cascade classifiers.  Raises if any XML is missing."""
        for path in (FACE_CASCADE_PATH, LEFT_EYE_CASCADE_PATH, RIGHT_EYE_CASCADE_PATH):
            if not path.exists():
                raise FileNotFoundError(f"Haar cascade missing: {path}")

        self.face_cascade      = cv2.CascadeClassifier(str(FACE_CASCADE_PATH))
        self.left_eye_cascade  = cv2.CascadeClassifier(str(LEFT_EYE_CASCADE_PATH))
        self.right_eye_cascade = cv2.CascadeClassifier(str(RIGHT_EYE_CASCADE_PATH))

    # ──────────────────────────────────────────────────────────────────────────
    # Audio (non-blocking because pygame plays on its own thread)
    # ──────────────────────────────────────────────────────────────────────────
    def play_alarm(self):
        """Play the alarm sound if audio is available.

        pygame.mixer.Sound.play() is non-blocking by design – it returns
        immediately and the sound plays on a background channel.
        """
        if self.sound is not None and not mixer.get_busy():
            # 'not mixer.get_busy()' prevents the alarm from re-triggering
            # every frame while the sound is already playing.
            self.sound.play()

    # ──────────────────────────────────────────────────────────────────────────
    # Per-frame eye probability helper
    # ──────────────────────────────────────────────────────────────────────────
    def _predict_eye_closed_prob(self, roi_gray, ex: int, ey: int, ew: int, eh: int) -> float:
        """Return the CNN's probability that the eye ROI is closed.

        Returns 0.0 if the eye model is not available.
        """
        if self.eye_model is None:
            return 0.0
        eye_crop = roi_gray[ey: ey + eh, ex: ex + ew]
        eye_img  = cv2.resize(eye_crop, (24, 24)).astype("float32") / 255.0
        eye_img  = eye_img.reshape(1, 24, 24, 1)
        probs    = self.eye_model.predict(eye_img, verbose=0)[0]
        return float(probs[self.eye_closed_class_idx])

    # ──────────────────────────────────────────────────────────────────────────
    # Yawn state machine
    # ──────────────────────────────────────────────────────────────────────────
    def process_yawn(self, frame, mouth_gray, dt: float):
        """Update yawn state machine and draw yawn label on frame.

        Parameters
        ----------
        frame      : BGR video frame (drawn on in-place)
        mouth_gray : Grayscale crop of the mouth region
        dt         : Elapsed seconds since the previous frame

        State transitions
        -----------------
        yawn_prob_ema >= yawn_start_threshold  →  accumulate yawn_open_duration
            yawn_open_duration >= YAWN_OPEN_SECONDS_THRESHOLD  →  count yawn event
        yawn_prob_ema <  yawn_start_threshold (while in_progress)
            yawn_prob_ema <= yawn_end_threshold  →  accumulate yawn_release_duration
            yawn_release_duration >= YAWN_RELEASE_SECONDS_THRESHOLD → release yawn

        Display rule
        ────────────
        "Yawning"     if yawn_open_duration >= YAWN_OPEN_DISPLAY_SECONDS (0.25 s)
        "Not Yawning" otherwise
        """
        if self.yawn_model is None or mouth_gray.size == 0:
            # Always draw a label even when the model is inactive
            cv2.putText(
                frame, "Yawn: N/A", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2,
            )
            return

        # ── 1. Run CNN on mouth crop ─────────────────────────────────────────
        mouth_img = cv2.resize(mouth_gray, (64, 64)).astype("float32") / 255.0
        mouth_img = mouth_img.reshape(1, 64, 64, 1)
        probs     = self.yawn_model.predict(mouth_img, verbose=0)[0]
        yawn_prob = float(probs[self.yawn_class_idx])

        # ── 2. Smooth with EMA (reduces single-frame noise) ──────────────────
        # α=0.4 gives a responsive but stable estimate
        self.yawn_prob_ema = 0.6 * self.yawn_prob_ema + 0.4 * yawn_prob

        # ── 3. State machine ─────────────────────────────────────────────────
        if self.yawn_prob_ema >= self.yawn_start_threshold:
            # Mouth is open / widening
            self.yawn_open_duration    += dt
            self.yawn_release_duration  = 0.0   # reset close-timer

            # Count a new yawn event once mouth has been open long enough and
            # no yawn is already being tracked and the min-gap has elapsed.
            now = time.time()
            if (
                self.yawn_open_duration >= YAWN_OPEN_SECONDS_THRESHOLD
                and not self.yawn_in_progress
                and (now - self.last_yawn_ts) >= YAWN_MIN_GAP_SECONDS
            ):
                self.yawn_count       += 1
                self.yawn_in_progress  = True
                self.last_yawn_ts      = now
                print(f"[Info] Yawn #{self.yawn_count} registered.")

        else:
            # Mouth is partially or fully closed
            if self.yawn_in_progress:
                # Hysteresis: require the EMA to drop below the lower threshold
                if self.yawn_prob_ema <= self.yawn_end_threshold:
                    self.yawn_release_duration += dt
                else:
                    # Still between end and start threshold → hold, don't reset
                    self.yawn_release_duration = 0.0

                # Yawn is officially over once mouth stays closed long enough
                if self.yawn_release_duration >= YAWN_RELEASE_SECONDS_THRESHOLD:
                    self.yawn_in_progress      = False
                    self.yawn_open_duration    = 0.0
                    self.yawn_release_duration = 0.0
            else:
                # No yawn in progress → simply drain the open-duration counter
                self.yawn_open_duration = 0.0

        # ── 4. Display label (time-gated → no flicker) ───────────────────────
        # "Yawning" is shown ONLY after mouth has been open >= 0.25 s
        if self.yawn_open_duration >= YAWN_OPEN_DISPLAY_SECONDS:
            yawn_label = "Yawn: Yawning"
            yawn_color = (0, 165, 255)          # orange
        else:
            yawn_label = "Yawn: Not Yawning"
            yawn_color = (0, 255, 255)          # cyan

        cv2.putText(
            frame, yawn_label, (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, yawn_color, 2,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Calibration phase finaliser
    # ──────────────────────────────────────────────────────────────────────────
    def _finalise_calibration(self):
        """Compute per-session adaptive thresholds from calibration samples."""
        if self.eye_calib_vals:
            eye_base = float(np.mean(self.eye_calib_vals))
            # Clamp between 0.75 and 0.98 to stay sensible
            self.eye_closed_threshold = float(np.clip(eye_base + 0.25, 0.75, 0.98))
        else:
            self.eye_closed_threshold = EYE_FULLY_CLOSED_PROBABILITY_THRESHOLD

        if self.yawn_calib_vals:
            yawn_base = float(np.mean(self.yawn_calib_vals))
            self.yawn_start_threshold = float(np.clip(yawn_base + 0.12, 0.40, 0.75))
        else:
            self.yawn_start_threshold = YAWN_PROBABILITY_THRESHOLD

        # End threshold is always 0.10 below start, floored at 0.25
        self.yawn_end_threshold = max(0.25, self.yawn_start_threshold - 0.10)
        self.calibrating = False

        print(
            f"[Info] Calibration done → "
            f"eye_thr={self.eye_closed_threshold:.2f}, "
            f"yawn_start={self.yawn_start_threshold:.2f}, "
            f"yawn_end={self.yawn_end_threshold:.2f}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Main detection loop
    # ──────────────────────────────────────────────────────────────────────────
    def run(self):
        """Open the webcam and run the detection loop until 'q' is pressed."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[Error] Could not open webcam.")
            return

        print("Driver Drowsiness Detection started.  Press 'q' to quit.")
        prev_ts = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Warning] Failed to read frame – stopping.")
                break

            # ── Δt (seconds since last frame) ────────────────────────────────
            # Used to accumulate time-based durations independent of frame rate.
            now_ts  = time.time()
            dt      = max(0.0, now_ts - prev_ts)   # clamp to ≥ 0 (clock skew guard)
            prev_ts = now_ts

            # ── Convert to grayscale for cascade detection ────────────────────
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.3, minNeighbors=5
            )

            face_detected = len(faces) > 0

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                roi_gray   = gray[y: y + h, x: x + w]
                upper_face = roi_gray[: int(h * 0.60), :]   # top 60 % = eye region

                # ── Detect eyes inside the upper face ROI ─────────────────────
                eyes_left  = self.left_eye_cascade.detectMultiScale(
                    upper_face, scaleFactor=1.1, minNeighbors=6, minSize=(18, 18)
                )
                eyes_right = self.right_eye_cascade.detectMultiScale(
                    upper_face, scaleFactor=1.1, minNeighbors=6, minSize=(18, 18)
                )
                eyes = list(eyes_left) + list(eyes_right)
                # Keep the two largest detections (most likely the real eyes)
                eyes = sorted(eyes, key=lambda b: b[2] * b[3], reverse=True)[:2]

                # ── Eye state update ─────────────────────────────────────────
                eye_prob_now = None
                if len(eyes) >= 2 and self.eye_model is not None:
                    closed_probs = [
                        self._predict_eye_closed_prob(upper_face, ex, ey, ew, eh)
                        for (ex, ey, ew, eh) in eyes
                    ]
                    eye_prob_now   = float(np.mean(closed_probs))
                    eyes_closed_now = all(p >= self.eye_closed_threshold for p in closed_probs)

                    if eyes_closed_now:
                        # Accumulate closed-duration (gated by calibration below)
                        self.eye_closed_duration += dt
                    else:
                        # Reset as soon as any eye opens
                        self.eye_closed_duration = 0.0

                    # ── Eye display label (0.3 s gate, suppressed during calib) ──
                    # Show "Closed" ONLY after eyes have been shut for ≥ 0.3 s.
                    # During calibration eye_closed_duration is always 0.0 so
                    # this label would always say "Open" — show nothing instead.
                    if not self.calibrating:
                        if self.eye_closed_duration >= EYE_CLOSED_DISPLAY_SECONDS:
                            eye_label = "Eyes: Closed"
                            eye_color = (0, 0, 255)     # red
                        else:
                            eye_label = "Eyes: Open"
                            eye_color = (0, 255, 0)     # green
                        cv2.putText(
                            frame, eye_label, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, eye_color, 2,
                        )

                elif self.eye_model is not None:
                    # Eyes were not found in this frame → treat as open
                    self.eye_closed_duration = 0.0
                    if not self.calibrating:
                        cv2.putText(
                            frame, "Eyes: Open", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                        )

                # ── Mouth ROI extraction ─────────────────────────────────────
                mouth_top    = y + int(h * 0.50)
                mouth_bottom = y + h
                mouth_left   = x + int(w * 0.05)
                mouth_right  = x + int(w * 0.95)
                mouth_gray   = gray[mouth_top:mouth_bottom, mouth_left:mouth_right]

                # Gather raw yawn probability for calibration (before process_yawn)
                yawn_now = None
                if mouth_gray.size > 0 and self.yawn_model is not None:
                    m_img    = cv2.resize(mouth_gray, (64, 64)).astype("float32") / 255.0
                    m_img    = m_img.reshape(1, 64, 64, 1)
                    yawn_now = float(
                        self.yawn_model.predict(m_img, verbose=0)[0][self.yawn_class_idx]
                    )

                # ── Calibration phase (first 50 face samples) ────────────────
                if self.calibrating:
                    if eye_prob_now is not None:
                        self.eye_calib_vals.append(eye_prob_now)
                    if yawn_now is not None:
                        self.yawn_calib_vals.append(yawn_now)
                    self.calibration_samples += 1

                    # Keep all state counters frozen during calibration
                    self.eye_closed_duration   = 0.0
                    self.yawn_open_duration    = 0.0
                    self.yawn_release_duration = 0.0
                    self.yawn_in_progress      = False
                    self.yawn_count            = 0

                    cv2.putText(
                        frame,
                        f"Calibrating… {self.calibration_samples}/50",
                        (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2,
                    )

                    if self.calibration_samples >= 50:
                        self._finalise_calibration()

                # ── Yawn state machine (runs ONLY after calibration is done) ──
                # During calibration all state counters are frozen (see above),
                # so we must NOT call process_yawn yet or it would re-accumulate
                # yawn_open_duration in the same frame we just reset it to 0.
                if not self.calibrating:
                    self.process_yawn(frame, mouth_gray, dt)
                else:
                    # Still calibrating – draw a placeholder yawn label
                    cv2.putText(
                        frame, "Yawn: Calibrating…", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 215, 255), 2,
                    )

                # Process only the first / largest face
                break

            # ── No face visible → gently drain eye-closed timer ──────────────
            if not face_detected:
                # Drain slowly (half per second) rather than hard-reset, so a
                # brief occlusion doesn't destroy a near-alarm state.
                self.eye_closed_duration = max(
                    0.0, self.eye_closed_duration - dt * 0.5
                )

            # ── HUD: Yawn counter ─────────────────────────────────────────────
            cv2.putText(
                frame, f"Yawn Count: {self.yawn_count}",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
            )

            # ── Alert logic ───────────────────────────────────────────────────
            # Eye alarm: eyes continuously closed for >= 3.0 s
            eye_alert  = (
                self.eye_model is not None
                and self.eye_closed_duration >= EYE_DROWSY_SECONDS_THRESHOLD
            )
            # Yawn alarm: >= 2 valid yawns counted this session
            yawn_alert = (
                self.yawn_model is not None
                and self.yawn_count >= YAWN_EVENT_LIMIT
            )
            is_drowsy = eye_alert or yawn_alert

            if is_drowsy:
                self.play_alarm()  # non-blocking (pygame background channel)
                cv2.putText(
                    frame, "DROWSINESS ALERT!",
                    (80, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3,
                )

            # ── Display and exit ──────────────────────────────────────────────
            cv2.imshow("Driver Drowsiness Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point (also called by run.py via detect_main())
# ─────────────────────────────────────────────────────────────────────────────
def main():
    try:
        detector = DrowsinessDetector()
        detector.run()
    except Exception as exc:
        print(f"\n[Fatal Error] {exc}")


if __name__ == "__main__":
    main()
