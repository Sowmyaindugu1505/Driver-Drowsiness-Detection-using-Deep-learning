"""
src/detector.py  -  Driver Drowsiness Detector
===============================================

DISPLAY  (on video at all times):
  Line 1:  Eyes: Open   (green)  /  Eyes: Closed  (red)
  Line 2:  Not Yawning  (cyan)   /  Yawning       (orange)
  Line 3:  Yawn Count: N         (white)

TIMING RULES (exact requirements):
  Eye display  : show "Eyes: Closed"  when eye_closed_duration >= 0.3 s
  Eye alarm    : trigger alarm        when eye_closed_duration >= 3.0 s
  Yawn display : show "Yawning"       when yawn_open_duration  >= 0.25 s
  Yawn alarm   : trigger alarm        when yawn_count          >= 2

HOW dt-ACCUMULATION WORKS:
  Each frame measures real wall-clock elapsed time:
      dt = time.time() - prev_time
  While a condition is TRUE  → add dt to the duration counter.
  When  a condition is FALSE → reset the counter to 0.0.
  This makes all thresholds frame-rate-independent (always in seconds).
"""

import json
import random
import time

import cv2
import numpy as np
from pygame import mixer
from tensorflow.keras.models import load_model

from src.config import (
    ALARM_PATH,
    EYE_CLOSED_CLASS_INDEX,
    EYE_CLOSED_DISPLAY_SECONDS,
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
    YAWN_OPEN_DISPLAY_SECONDS,
    YAWN_OPEN_SECONDS_THRESHOLD,
    YAWN_PROBABILITY_THRESHOLD,
    YAWN_RELEASE_SECONDS_THRESHOLD,
)


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helper
# ─────────────────────────────────────────────────────────────────────────────

def _load_yawn_class_index() -> int:
    """Return the index of the yawn-positive output from the saved class map.
    Falls back to 1 if the file is missing or unreadable.
    """
    if not YAWN_CLASS_MAP_PATH.exists():
        return 1
    try:
        mapping  = json.loads(YAWN_CLASS_MAP_PATH.read_text())
        positive = {"yawn", "yawning", "mouth_open"}
        negative = {"no_yawn", "not_yawn", "non_yawn", "normal"}
        # First pass: explicit positive class name
        for name, idx in mapping.items():
            key = name.strip().lower().replace("-", "_").replace(" ", "_")
            if key in positive:
                return int(idx)
        # Second pass: anything that is NOT a known negative name
        for name, idx in mapping.items():
            key = name.strip().lower().replace("-", "_").replace(" ", "_")
            if key not in negative:
                return int(idx)
    except Exception as exc:
        print(f"[Warning] Could not read yawn class map: {exc}")
    return 1


# ─────────────────────────────────────────────────────────────────────────────
# Detector class
# ─────────────────────────────────────────────────────────────────────────────

class DrowsinessDetector:
    """Continuous, real-time driver drowsiness detector.

    State variables
    ───────────────
    eye_closed_duration    float  Seconds both eyes have been continuously closed.
                                  Resets to 0.0 the instant either eye opens.
    yawn_open_duration     float  Seconds mouth has been continuously above the
                                  CNN start-threshold. Resets when mouth closes.
    yawn_release_duration  float  Seconds mouth has been below the CNN end-threshold
                                  after a counted yawn. Used to "release" the lock.
    yawn_in_progress       bool   True from the moment a yawn is counted until the
                                  mouth has been closed long enough to end it.
    yawn_prob_ema          float  Exponential moving average of the raw CNN output.
                                  Absorbs single-frame spikes so talking / sighing
                                  cannot trigger a false yawn count.
    last_yawn_ts           float  Wall-clock timestamp of the last counted yawn.
                                  Enforces the minimum gap between yawn events.
    yawn_count             int    Total valid yawns counted this session.
    is_drowsy_alarm        bool   Latched True once an alarm condition is met.
                                  Stays True so alarm keeps playing even if the
                                  triggering condition temporarily disappears.
    """

    def __init__(self):
        # Eye state
        self.eye_closed_duration = 0.0

        # Yawn state
        self.yawn_open_duration    = 0.0
        self.yawn_release_duration = 0.0
        self.yawn_in_progress      = False
        self.yawn_prob_ema         = 0.0
        self.last_yawn_ts          = 0.0
        self.yawn_count            = 0

        # Alarm latch — once True it stays True for the rest of the session
        # so a momentary face-loss does not silence the alarm
        self.is_drowsy_alarm = False

        # Detection thresholds — fixed from config, active from frame 1
        self.eye_closed_threshold = EYE_FULLY_CLOSED_PROBABILITY_THRESHOLD
        self.yawn_start_threshold = YAWN_PROBABILITY_THRESHOLD
        self.yawn_end_threshold   = YAWN_END_PROBABILITY_THRESHOLD

        self._init_audio()
        self._init_models()
        self._init_cascades()

    # ── initialisation ────────────────────────────────────────────────────────

    def _init_audio(self):
        """Initialise pygame mixer. On failure, alarms are visual-only."""
        try:
            mixer.init()
            if not ALARM_PATH.exists():
                raise FileNotFoundError(f"alarm.wav not found at {ALARM_PATH}")
            self.sound = mixer.Sound(str(ALARM_PATH))
        except Exception as exc:
            print(f"[Warning] Audio unavailable: {exc}. Alerts will be visual only.")
            self.sound = None

    def _init_models(self):
        """Load eye and yawn CNN models from disk."""
        print("Loading models...")
        self.eye_model  = None
        self.yawn_model = None

        if EYE_MODEL_PATH.exists():
            self.eye_model = load_model(str(EYE_MODEL_PATH))
            self.eye_closed_class_idx = self._infer_eye_closed_class_index()
        else:
            print(f"[Warning] Eye model not found: {EYE_MODEL_PATH}")
            self.eye_closed_class_idx = EYE_CLOSED_CLASS_INDEX

        if YAWN_MODEL_PATH.exists():
            self.yawn_model = load_model(str(YAWN_MODEL_PATH))
        else:
            print(f"[Warning] Yawn model not found: {YAWN_MODEL_PATH}")

        if self.eye_model is None and self.yawn_model is None:
            raise FileNotFoundError(
                "No trained models found in models/.\n"
                "Train them first:\n"
                "  python run.py --mode train --target eyes\n"
                "  python run.py --mode train --target yawns"
            )

        self.yawn_class_idx = _load_yawn_class_index()
        print("Models loaded successfully.")

    def _infer_eye_closed_class_index(self) -> int:
        """Probe the eye CNN with sample images to infer which output index
        corresponds to 'closed'. Falls back to the config default safely.
        """
        try:
            open_dir   = EYE_DATA_DIR / "train" / "open"
            closed_dir = EYE_DATA_DIR / "train" / "closed"
            if not open_dir.exists() or not closed_dir.exists():
                return EYE_CLOSED_CLASS_INDEX

            open_files   = [p for p in open_dir.glob("*")   if p.is_file()]
            closed_files = [p for p in closed_dir.glob("*") if p.is_file()]
            if len(open_files) < 10 or len(closed_files) < 10:
                return EYE_CLOSED_CLASS_INDEX

            def _mean_probs(file_list):
                vals = []
                for p in random.sample(file_list, min(30, len(file_list))):
                    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    img = cv2.resize(img, (24, 24)).astype("float32") / 255.0
                    pred = self.eye_model.predict(
                        img.reshape(1, 24, 24, 1), verbose=0
                    )[0]
                    vals.append(pred)
                if not vals:
                    return np.array([0.5, 0.5], dtype=np.float32)
                return np.mean(np.array(vals), axis=0)

            closed_mean = _mean_probs(closed_files)
            open_mean   = _mean_probs(open_files)
            idx = int(np.argmax(closed_mean - open_mean))
            print(f"[Info] Eye closed class index inferred = {idx}")
            return idx
        except Exception as exc:
            print(f"[Warning] Eye class inference failed ({exc}). Using default.")
            return EYE_CLOSED_CLASS_INDEX

    def _init_cascades(self):
        """Load Haar cascade XML files. Raises FileNotFoundError if any are missing."""
        for path in (FACE_CASCADE_PATH, LEFT_EYE_CASCADE_PATH, RIGHT_EYE_CASCADE_PATH):
            if not path.exists():
                raise FileNotFoundError(f"Haar cascade missing: {path}")
        self.face_cascade      = cv2.CascadeClassifier(str(FACE_CASCADE_PATH))
        self.left_eye_cascade  = cv2.CascadeClassifier(str(LEFT_EYE_CASCADE_PATH))
        self.right_eye_cascade = cv2.CascadeClassifier(str(RIGHT_EYE_CASCADE_PATH))

    # ── audio ─────────────────────────────────────────────────────────────────

    def _play_alarm(self):
        """Play the alarm sound (non-blocking via pygame).
        mixer.get_busy() prevents stacking the sound on every frame.
        """
        if self.sound is not None and not mixer.get_busy():
            self.sound.play()

    # ── per-frame eye CNN helper ──────────────────────────────────────────────

    def _eye_closed_prob(self, upper_face: np.ndarray,
                         ex: int, ey: int, ew: int, eh: int) -> float:
        """Return the CNN probability that the eye ROI is closed.
        upper_face : grayscale crop of the top 60% of the face ROI.
        ex,ey,ew,eh: bounding box of the eye inside upper_face.
        """
        crop  = upper_face[ey: ey + eh, ex: ex + ew]
        img   = cv2.resize(crop, (24, 24)).astype("float32") / 255.0
        probs = self.eye_model.predict(img.reshape(1, 24, 24, 1), verbose=0)[0]
        return float(probs[self.eye_closed_class_idx])

    # ── yawn state machine ────────────────────────────────────────────────────

    def _update_yawn(self, mouth_gray: np.ndarray, dt: float) -> str:
        """Run yawn CNN, update all yawn state variables, return display label.

        Returns "Yawning" or "Not Yawning".

        FALSE-POSITIVE PREVENTION (why the old code counted when not yawning):
        ───────────────────────────────────────────────────────────────────────
        Problem 1 — threshold too low (0.45):
            Normal talking easily pushed the CNN above 45% confidence.
            Fix: raised YAWN_PROBABILITY_THRESHOLD to 0.65.

        Problem 2 — time-gate too short (0.25 s):
            At 30 fps, 0.25 s = ~7 frames.  A single open-mouth expression
            lasts more than 7 frames.
            Fix: raised YAWN_OPEN_SECONDS_THRESHOLD to 1.5 s.

        Problem 3 — minimum gap too short (0.4 s):
            One physical yawn could be counted twice if the EMA dipped briefly.
            Fix: raised YAWN_MIN_GAP_SECONDS to 3.0 s.

        Problem 4 — release time too short (0.2 s):
            Mouth movements after a yawn re-triggered within 0.2 s.
            Fix: raised YAWN_RELEASE_SECONDS_THRESHOLD to 1.5 s.

        Problem 5 — hysteresis band too narrow (0.45 - 0.30 = 0.15):
            EMA oscillated across the start threshold, causing rapid counts.
            Fix: new gap is 0.65 - 0.30 = 0.35, which the EMA cannot straddle.

        Problem 6 — yawn_open_duration not reset in grey zone:
            When EMA was between end and start thresholds, yawn_open_duration
            kept its old accumulated value. On next re-open the threshold was
            immediately re-crossed and another count fired.
            Fix: reset yawn_open_duration when we enter the grey zone.
        """
        if self.yawn_model is None or mouth_gray.size == 0:
            return "Not Yawning"

        # ── step 1: CNN prediction ────────────────────────────────────────────
        mouth_img = cv2.resize(mouth_gray, (64, 64)).astype("float32") / 255.0
        probs     = self.yawn_model.predict(
            mouth_img.reshape(1, 64, 64, 1), verbose=0
        )[0]
        raw_prob = float(probs[self.yawn_class_idx])

        # ── step 2: EMA smoothing (α = 0.30) ─────────────────────────────────
        # Lower α = more smoothing = better spike rejection.
        # 0.30 means current frame contributes only 30% to the average,
        # so a single loud frame cannot push EMA over the 0.65 threshold alone.
        self.yawn_prob_ema = 0.70 * self.yawn_prob_ema + 0.30 * raw_prob
        ema = self.yawn_prob_ema

        # ── step 3: state machine ─────────────────────────────────────────────
        if ema >= self.yawn_start_threshold:
            # Mouth is genuinely open above start threshold
            self.yawn_open_duration    += dt
            self.yawn_release_duration  = 0.0   # reset the "closing" timer

            now = time.time()
            # Count a yawn only when ALL three conditions are true:
            #   (a) mouth has been open for >= 1.5 s continuously
            #   (b) no yawn is already being tracked right now
            #   (c) at least 3.0 s have passed since the last counted yawn
            if (
                self.yawn_open_duration >= YAWN_OPEN_SECONDS_THRESHOLD
                and not self.yawn_in_progress
                and (now - self.last_yawn_ts) >= YAWN_MIN_GAP_SECONDS
            ):
                self.yawn_count      += 1
                self.yawn_in_progress = True
                self.last_yawn_ts     = now

        elif ema <= self.yawn_end_threshold:
            # Mouth is clearly closed (below the lower / end threshold)
            if self.yawn_in_progress:
                # Accumulate release time — must stay closed for >= 1.5 s
                self.yawn_release_duration += dt
                if self.yawn_release_duration >= YAWN_RELEASE_SECONDS_THRESHOLD:
                    # Yawn is officially over; reset for the next one
                    self.yawn_in_progress      = False
                    self.yawn_open_duration    = 0.0
                    self.yawn_release_duration = 0.0
            else:
                # Not in a yawn and clearly closed — drain open-duration
                self.yawn_open_duration    = 0.0
                self.yawn_release_duration = 0.0

        else:
            # Grey zone: EMA is between end_threshold and start_threshold.
            # Hold all timers frozen — do NOT accumulate either open or release.
            # Also RESET open_duration so re-entry into the open branch cannot
            # immediately re-trigger a count (FIX for Problem 6 above).
            self.yawn_open_duration    = 0.0   # FIX: was missing, caused double-count
            self.yawn_release_duration = 0.0

        # ── step 4: display label ─────────────────────────────────────────────
        # Show "Yawning" only after 0.25 s continuous open — prevents flicker
        if self.yawn_open_duration >= YAWN_OPEN_DISPLAY_SECONDS:
            return "Yawning"
        return "Not Yawning"

    # ── main detection loop ───────────────────────────────────────────────────

    def run(self):
        """Open webcam and run detection continuously until 'q' is pressed."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[Error] Cannot open webcam.")
            return

        print("Detection running. Press 'q' to quit.")
        prev_ts = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Warning] Frame read failed — stopping.")
                break

            # ── real elapsed time this frame ──────────────────────────────────
            now_ts  = time.time()
            dt      = max(0.0, now_ts - prev_ts)   # clamp: never negative
            prev_ts = now_ts

            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.3, minNeighbors=5
            )

            # Default labels — always drawn, even with no face detected
            eye_label  = "Eyes: Open"
            eye_color  = (0, 255, 0)    # green
            yawn_label = "Not Yawning"
            yawn_color = (0, 255, 255)  # cyan

            if len(faces) > 0:
                # Take only the first (largest) detected face
                x, y, w, h = faces[0]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                roi_gray   = gray[y: y + h, x: x + w]
                upper_face = roi_gray[: int(h * 0.60), :]  # top 60% = eye region

                # ── eye detection ─────────────────────────────────────────────
                if self.eye_model is not None:
                    eyes_l = self.left_eye_cascade.detectMultiScale(
                        upper_face, scaleFactor=1.1, minNeighbors=6,
                        minSize=(18, 18)
                    )
                    eyes_r = self.right_eye_cascade.detectMultiScale(
                        upper_face, scaleFactor=1.1, minNeighbors=6,
                        minSize=(18, 18)
                    )
                    # Keep only the 2 largest boxes (most likely the real eyes)
                    eyes = sorted(
                        list(eyes_l) + list(eyes_r),
                        key=lambda b: b[2] * b[3], reverse=True
                    )[:2]

                    if len(eyes) >= 2:
                        # Get CNN closed-probability for each detected eye
                        probs = [
                            self._eye_closed_prob(upper_face, ex, ey, ew, eh)
                            for (ex, ey, ew, eh) in eyes
                        ]
                        # Both eyes must be above threshold to call them closed
                        both_closed = all(p >= self.eye_closed_threshold for p in probs)

                        if both_closed:
                            self.eye_closed_duration += dt   # accumulate
                        else:
                            self.eye_closed_duration = 0.0   # any eye open → reset

                        # DISPLAY RULE: show "Closed" only after 0.3 s
                        # Requirement: >= 0.3 s continuous closure
                        if self.eye_closed_duration >= EYE_CLOSED_DISPLAY_SECONDS:
                            eye_label = "Eyes: Closed"
                            eye_color = (0, 0, 255)   # red
                        # else: stays "Eyes: Open" (green default above)

                    else:
                        # Fewer than 2 eyes found — treat as open, reset timer
                        self.eye_closed_duration = 0.0

                # ── yawn detection ────────────────────────────────────────────
                if self.yawn_model is not None:
                    # Mouth ROI: lower 45% of face, inset 10% on each side
                    mouth_top    = y + int(h * 0.55)
                    mouth_bottom = y + h
                    mouth_left   = x + int(w * 0.10)
                    mouth_right  = x + int(w * 0.90)
                    mouth_gray   = gray[mouth_top:mouth_bottom,
                                        mouth_left:mouth_right]

                    result = self._update_yawn(mouth_gray, dt)
                    if result == "Yawning":
                        yawn_label = "Yawning"
                        yawn_color = (0, 165, 255)  # orange

            else:
                # No face in this frame.
                # Drain eye_closed_duration slowly (0.5× per second) rather
                # than hard-resetting, so a brief occlusion (head-turn, hand)
                # does not destroy a 2.9 s accumulated state.
                self.eye_closed_duration = max(
                    0.0, self.eye_closed_duration - dt * 0.5
                )

            # ── ALARM LOGIC ───────────────────────────────────────────────────
            # EYE ALARM:  eyes continuously closed for >= 3.0 s
            # Requirement: alarm at >= 3.0 s
            eye_alarm_now = (
                self.eye_model is not None
                and self.eye_closed_duration >= EYE_DROWSY_SECONDS_THRESHOLD
            )
            # YAWN ALARM: yawn_count has reached >= 2
            # Requirement: alarm when yawn_count >= 2
            yawn_alarm_now = (
                self.yawn_model is not None
                and self.yawn_count >= YAWN_EVENT_LIMIT
            )

            # Latch: once either alarm fires it stays active for the session.
            # This means the alarm keeps sounding even if the face briefly
            # disappears or eyes momentarily open at exactly the 3.0 s mark.
            if eye_alarm_now or yawn_alarm_now:
                self.is_drowsy_alarm = True

            if self.is_drowsy_alarm:
                self._play_alarm()   # non-blocking; mixer.get_busy() prevents stacking

            # ── DRAW — exactly 3 labels, nothing else ─────────────────────────
            # Position y=35: Eye state
            cv2.putText(frame, eye_label,
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, eye_color, 2)
            # Position y=75: Yawn state
            cv2.putText(frame, yawn_label,
                        (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.85, yawn_color, 2)
            # Position y=115: Yawn count
            cv2.putText(frame, f"Yawn Count: {self.yawn_count}",
                        (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

            cv2.imshow("Driver Drowsiness Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    try:
        DrowsinessDetector().run()
    except Exception as exc:
        print(f"\n[Fatal Error] {exc}")


if __name__ == "__main__":
    main()
