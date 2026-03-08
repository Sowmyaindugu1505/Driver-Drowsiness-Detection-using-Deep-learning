"""
src/detector.py  —  Driver Drowsiness Detector
================================================

ON-SCREEN OUTPUT (exactly three lines, nothing else):
  Line 1:  "Eyes: Open"    (green)   OR  "Eyes: Closed"   (red)
  Line 2:  "Not Yawning"   (cyan)    OR  "Yawning"        (orange)
  Line 3:  "Yawn Count: N" (white)

DETECTION RULES:
  Show "Eyes: Closed"   when eye_closed_duration >= 0.3 s, else "Eyes: Open"
  Trigger eye alarm     when eye_closed_duration >= 3.0 s
  Show "Yawning"        when yawn_open_duration  >= 0.25 s, else "Not Yawning"
  Count a yawn          when yawn_open_duration  >= 0.3 s
  Trigger yawn alarm    when yawn_count >= 2

EYE DETECTION  (no Haar eye cascades):
  Eye regions are extracted at fixed proportions from the face bounding box:
    Left  eye: x 55%–90%,  y 10%–42% of face
    Right eye: x 10%–45%,  y 10%–42% of face
  These fixed crops are consistent every frame regardless of eye state.
  The old Haar eye cascades were trained on open eyes and missed ~30% of
  frames when eyes were closed, causing constant timer resets.

YAWN DETECTION:
  Mouth ROI: x 15%–85%,  y 62%–98% of face  (fixed proportions)
  CNN output smoothed with EMA (α=0.60) for noise resistance.
  State machine counts a yawn only after sustained open-mouth >= 0.3 s.
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
    EYE_CONSEC_FRAMES,
    EYE_DATA_DIR,
    EYE_DROWSY_SECONDS_THRESHOLD,
    EYE_FULLY_CLOSED_PROBABILITY_THRESHOLD,
    EYE_MODEL_PATH,
    FACE_CASCADE_PATH,
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


# =============================================================================
# Module-level helper
# =============================================================================

def _load_yawn_class_index() -> int:
    """Return the yawn-positive class index from the JSON file saved by model.py.

    The file maps class-folder names → integer indices, e.g.
        {"no_yawn": 0, "yawn": 1}
    We look for a name that clearly means "yawning" and return its index.
    Falls back to 1 on any problem (Keras alphabetical ordering puts "yawn"
    after "no_yawn", so index 1 is the common default).
    """
    if not YAWN_CLASS_MAP_PATH.exists():
        return 1
    try:
        mapping  = json.loads(YAWN_CLASS_MAP_PATH.read_text())
        positive = {"yawn", "yawning", "mouth_open"}
        negative = {"no_yawn", "not_yawn", "non_yawn", "normal", "closed"}
        # First pass: exact positive match
        for name, idx in mapping.items():
            key = name.strip().lower().replace("-", "_").replace(" ", "_")
            if key in positive:
                return int(idx)
        # Second pass: anything that is NOT a known negative
        for name, idx in mapping.items():
            key = name.strip().lower().replace("-", "_").replace(" ", "_")
            if key not in negative:
                return int(idx)
    except Exception as exc:
        print(f"[Warning] Could not read yawn class map: {exc}")
    return 1


# =============================================================================
# DrowsinessDetector
# =============================================================================

class DrowsinessDetector:
    """
    State machine for real-time drowsiness detection from a webcam.

    Eye state variables
    ───────────────────
    eye_closed_duration  float  Seconds eyes have been continuously closed.
                                Incremented by dt each frame CNN says "closed",
                                reset to 0.0 the frame CNN says "open".
    eye_consec_closed    int    Consecutive frame count of CNN "closed" votes.
                                Duration timer only starts after this reaches
                                EYE_CONSEC_FRAMES (=2), absorbing single-frame noise.

    Yawn state variables
    ────────────────────
    yawn_open_duration    float  Seconds mouth has been continuously above threshold.
    yawn_prob_ema         float  Exponential moving average of raw CNN yawn probability.
                                 α=0.60 so EMA crosses threshold in ~2 frames from cold start.
    yawn_in_progress      bool   True once a yawn is counted; blocks re-counting same yawn.
    last_yawn_ts          float  Wall-clock time of last counted yawn (debounce gap).
    yawn_release_duration float  Seconds mouth has been below end_threshold post-yawn.
    yawn_count            int    Total valid yawns this session.

    Alarm
    ─────
    is_drowsy_alarm  bool  Latched True on first alarm trigger; never reset.
                           Alarm keeps playing until program exits.
    """

    def __init__(self):
        # Eye state
        self.eye_closed_duration = 0.0
        self.eye_consec_closed   = 0
        self.eye_consec_open     = 0
        self.eye_prob_ema        = 0.0
        self.eye_pair_prob_ema   = 0.0

        # Yawn state
        self.yawn_open_duration    = 0.0
        self.yawn_prob_ema         = 0.0
        self.yawn_in_progress      = False
        self.last_yawn_ts          = 0.0
        self.yawn_release_duration = 0.0
        self.yawn_count            = 0

        # Alarm latch — set True once, never reset
        self.is_drowsy_alarm = False
        self.alarm_reason = None  # "eyes" or "yawns"

        # Cache thresholds locally to avoid repeated attribute lookups
        self.eye_thr        = EYE_FULLY_CLOSED_PROBABILITY_THRESHOLD
        self.yawn_start_thr = YAWN_PROBABILITY_THRESHOLD
        self.yawn_end_thr   = YAWN_END_PROBABILITY_THRESHOLD

        self._init_audio()
        self._init_models()
        self._init_cascade()

    # ─────────────────────────────────────────────────────────────────────────
    # Initialisation helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _init_audio(self):
        """Initialise pygame mixer and load alarm sound.
        Sets self.sound to None if audio is unavailable (visual-only fallback).
        """
        try:
            mixer.init()
            if not ALARM_PATH.exists():
                raise FileNotFoundError(f"alarm.wav not found: {ALARM_PATH}")
            self.sound = mixer.Sound(str(ALARM_PATH))
        except Exception as exc:
            print(f"[Warning] Audio unavailable ({exc}). Visual alerts only.")
            self.sound = None

    def _init_models(self):
        """Load eye and yawn CNN models. Raises if neither model file exists."""
        print("Loading models …")
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
                "No trained models found.\n"
                "Run:  python run.py --mode train --target eyes\n"
                "      python run.py --mode train --target yawns"
            )

        # Load yawn class index and clamp to valid range
        raw_yawn_idx = _load_yawn_class_index()
        if self.yawn_model is not None:
            n_yawn_classes = self.yawn_model.output_shape[-1]
            self.yawn_class_idx = min(raw_yawn_idx, n_yawn_classes - 1)
        else:
            self.yawn_class_idx = raw_yawn_idx

        print(
            f"Models loaded.  "
            f"eye_closed_idx={self.eye_closed_class_idx}  "
            f"yawn_idx={self.yawn_class_idx}"
        )

    def _infer_eye_closed_class_index(self) -> int:
        """Probe the eye CNN with training images to find which index = 'closed'.

        Keras flow_from_directory assigns indices alphabetically, so for
        folders named 'closed' and 'open': closed=0, open=1.
        We confirm this empirically to be safe.
        Returns EYE_CLOSED_CLASS_INDEX (0) as fallback on any error.
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

            n_classes = self.eye_model.output_shape[-1]

            def _mean_probs(files):
                vals = []
                for p in random.sample(files, min(30, len(files))):
                    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    img = cv2.resize(img, (24, 24)).astype("float32") / 255.0
                    pred = self.eye_model.predict(
                        img.reshape(1, 24, 24, 1), verbose=0
                    )[0]
                    vals.append(pred)
                if not vals:
                    return np.full(n_classes, 0.5, dtype=np.float32)
                return np.mean(np.array(vals, dtype=np.float32), axis=0)

            diff = _mean_probs(closed_files) - _mean_probs(open_files)
            idx  = int(np.argmax(diff))
            # Clamp to valid range in case model has 1 output (binary sigmoid)
            idx  = min(idx, n_classes - 1)
            print(f"[Info] Eye closed class index = {idx}")
            return idx

        except Exception as exc:
            print(f"[Warning] Eye class inference failed ({exc}). Using default.")
            return EYE_CLOSED_CLASS_INDEX

    def _init_cascade(self):
        """Load the face Haar cascade.  Raises FileNotFoundError if missing."""
        if not FACE_CASCADE_PATH.exists():
            raise FileNotFoundError(f"Face cascade missing: {FACE_CASCADE_PATH}")
        self.face_cascade = cv2.CascadeClassifier(str(FACE_CASCADE_PATH))

    # ─────────────────────────────────────────────────────────────────────────
    # Audio
    # ─────────────────────────────────────────────────────────────────────────

    def _play_alarm(self):
        """Play alarm sound once; mixer.get_busy() prevents per-frame stacking."""
        if self.sound is not None and not mixer.get_busy():
            self.sound.play()

    # ─────────────────────────────────────────────────────────────────────────
    # ROI extraction  (fixed proportions — stable every frame)
    # ─────────────────────────────────────────────────────────────────────────

    def _extract_eye_rois(self, face_gray: np.ndarray):
        """Return (left_eye_crop, right_eye_crop) from face ROI.

        Uses fixed face proportions so crops are identical whether eyes are
        open or closed.  (Haar eye cascades fail on closed eyes.)

        Proportions measured from frontal-face geometry:
          y: 10%–42% of face height  (eye-band, excludes forehead and nose)
          Left  eye (camera-left = person's right): x 55%–90%
          Right eye (camera-right = person's left): x 10%–45%
        """
        h, w  = face_gray.shape
        y0    = int(h * 0.10)
        y1    = int(h * 0.42)
        left  = face_gray[y0:y1, int(w * 0.55):int(w * 0.90)]
        right = face_gray[y0:y1, int(w * 0.10):int(w * 0.45)]
        return left, right

    def _extract_mouth_roi(self, face_gray: np.ndarray) -> np.ndarray:
        """Return mouth crop from face ROI using fixed proportions.

        y: 62%–98% of face height  (covers lips, excludes nose and chin stub)
        x: 15%–85% of face width   (avoids cheek edges)
        """
        h, w = face_gray.shape
        return face_gray[int(h * 0.62):int(h * 0.98),
                         int(w * 0.15):int(w * 0.85)]

    # ─────────────────────────────────────────────────────────────────────────
    # Eye CNN
    # ─────────────────────────────────────────────────────────────────────────

    def _eye_closed_prob(self, eye_crop: np.ndarray) -> float | None:
        """Return CNN probability that this eye crop is closed, or None if crop is empty.

        Returns None (not 0.0) so callers can detect an unusable crop and
        exclude it from the average instead of diluting a valid reading.
        """
        if eye_crop.size == 0:
            return None
        img   = cv2.resize(eye_crop, (24, 24)).astype("float32") / 255.0
        probs = self.eye_model.predict(img.reshape(1, 24, 24, 1), verbose=0)[0]
        # Clamp index in case model has a single sigmoid output
        idx   = min(self.eye_closed_class_idx, len(probs) - 1)
        return float(probs[idx])

    # ─────────────────────────────────────────────────────────────────────────
    # Eye state update
    # ─────────────────────────────────────────────────────────────────────────

    def _update_eyes(self, face_gray: np.ndarray, dt: float) -> str:
        """Update eye_closed_duration, return 'Eyes: Closed' or 'Eyes: Open'.

        Algorithm
        ─────────
        1. Extract left and right eye crops at fixed proportions.
        2. Get CNN closed-probability for each crop.
           Crops returning None (empty) are excluded from the average.
        3. Average valid probabilities only — avoids a 0.0 from an unusable
           crop dragging the average below threshold and hiding a real closure.
        4. avg_prob >= threshold  →  increment eye_consec_closed.
           Once consec count reaches EYE_CONSEC_FRAMES (=2), start accumulating
           eye_closed_duration by dt each frame.
           avg_prob < threshold   →  reset both consec count and duration to 0.
        5. Return 'Eyes: Closed' once duration >= EYE_CLOSED_DISPLAY_SECONDS (0.3 s).
        """
        if self.eye_model is None:
            return "Eyes: Open"

        left_crop, right_crop = self._extract_eye_rois(face_gray)

        probs = [
            p for p in (
                self._eye_closed_prob(left_crop),
                self._eye_closed_prob(right_crop),
            )
            if p is not None
        ]

        if not probs:
            # Both crops somehow empty — treat as ambiguous, hold timer
            return (
                "Eyes: Closed"
                if self.eye_closed_duration >= EYE_CLOSED_DISPLAY_SECONDS
                else "Eyes: Open"
            )

        avg_prob = sum(probs) / len(probs)   # average of 1 or 2 valid probs
        pair_prob = min(probs) if len(probs) >= 2 else probs[0]
        if self.eye_prob_ema <= 0.0:
            self.eye_prob_ema = avg_prob
        else:
            # Smooth frame-to-frame eye probability to remove flicker.
            self.eye_prob_ema = 0.75 * self.eye_prob_ema + 0.25 * avg_prob
        if self.eye_pair_prob_ema <= 0.0:
            self.eye_pair_prob_ema = pair_prob
        else:
            # Pair EMA uses lower eye score, so "closed" needs both eyes to agree.
            self.eye_pair_prob_ema = 0.75 * self.eye_pair_prob_ema + 0.25 * pair_prob

        # Stricter threshold to avoid false "closed" when eyes are open.
        close_thr = max(0.72, self.eye_thr + 0.12)
        open_thr = close_thr - 0.10

        eye_signal = self.eye_pair_prob_ema if len(probs) >= 2 else self.eye_prob_ema
        if eye_signal >= close_thr:
            self.eye_consec_closed += 1
            self.eye_consec_open = 0
        elif eye_signal <= open_thr:
            self.eye_consec_open += 1
            self.eye_consec_closed = 0
        # In hysteresis band keep previous counters (stable state).

        if self.eye_consec_closed >= EYE_CONSEC_FRAMES:
            self.eye_closed_duration += dt
        elif self.eye_consec_open >= EYE_CONSEC_FRAMES:
            self.eye_closed_duration = 0.0

        return (
            "Eyes: Closed"
            if self.eye_closed_duration >= EYE_CLOSED_DISPLAY_SECONDS
            else "Eyes: Open"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Yawn state machine
    # ─────────────────────────────────────────────────────────────────────────

    def _update_yawn(self, face_gray: np.ndarray, dt: float) -> str:
        """Update yawn state, return 'Yawning' or 'Not Yawning'.

        Algorithm
        ─────────
        1. Extract mouth ROI at fixed proportions.
        2. Run yawn CNN; apply EMA smoothing (α=0.60).
           EMA crosses 0.45 threshold in ~2 frames from a cold start.
        3. EMA >= start_threshold (0.45):
               Accumulate yawn_open_duration.
               Reset yawn_release_duration to 0.
               Count a new yawn when:
                 (a) yawn_open_duration >= 0.3 s
                 (b) yawn_in_progress is False
                 (c) time since last counted yawn >= 3.0 s
               Set yawn_in_progress = True after counting.
           EMA <= end_threshold (0.20):
               Mouth is clearly closed.
               If yawn_in_progress: accumulate yawn_release_duration.
                 Once >= 0.8 s: release lock → yawn_in_progress = False,
                 reset timers.
               If not yawn_in_progress: reset both timers immediately.
           Grey zone (0.20 < EMA < 0.45):
               HOLD all timers unchanged. CNN fluctuation during a real yawn
               must not break the accumulation or trigger a spurious release.
        4. Return 'Yawning' once yawn_open_duration >= 0.25 s.
        """
        if self.yawn_model is None:
            return "Not Yawning"

        mouth_roi = self._extract_mouth_roi(face_gray)
        if mouth_roi.size == 0:
            return "Not Yawning"

        # ── CNN inference ─────────────────────────────────────────────────────
        mouth_img = cv2.resize(mouth_roi, (64, 64)).astype("float32") / 255.0
        probs     = self.yawn_model.predict(
            mouth_img.reshape(1, 64, 64, 1), verbose=0
        )[0]
        # Clamp index — handles both softmax (2+ outputs) and sigmoid (1 output)
        idx      = min(self.yawn_class_idx, len(probs) - 1)
        raw_prob = float(probs[idx])

        # ── EMA  α=0.60 ───────────────────────────────────────────────────────
        self.yawn_prob_ema = 0.40 * self.yawn_prob_ema + 0.60 * raw_prob
        ema = self.yawn_prob_ema

        # ── State machine ─────────────────────────────────────────────────────
        if ema >= self.yawn_start_thr:
            # Mouth is open
            self.yawn_open_duration   += dt
            self.yawn_release_duration = 0.0

            now = time.time()
            if (
                self.yawn_open_duration >= YAWN_OPEN_SECONDS_THRESHOLD
                and not self.yawn_in_progress
                and (now - self.last_yawn_ts) >= YAWN_MIN_GAP_SECONDS
            ):
                self.yawn_count      += 1
                self.yawn_in_progress = True
                self.last_yawn_ts     = now

        elif ema <= self.yawn_end_thr:
            # Mouth is clearly closed
            if self.yawn_in_progress:
                self.yawn_release_duration += dt
                if self.yawn_release_duration >= YAWN_RELEASE_SECONDS_THRESHOLD:
                    # Yawn fully over — unlock for next yawn
                    self.yawn_in_progress      = False
                    self.yawn_open_duration    = 0.0
                    self.yawn_release_duration = 0.0
            else:
                self.yawn_open_duration    = 0.0
                self.yawn_release_duration = 0.0

        # else: grey zone — hold all timers (no reset, no accumulation)

        # ── Display label ─────────────────────────────────────────────────────
        return (
            "Yawning"
            if self.yawn_open_duration >= YAWN_OPEN_DISPLAY_SECONDS
            else "Not Yawning"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Main detection loop
    # ─────────────────────────────────────────────────────────────────────────

    def run(self):
        """Open webcam, run detection continuously until 'q' is pressed."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap.release()                          # FIX: release even on open failure
            print("[Error] Cannot open webcam.")
            return

        print("Detection running. Press 'q' to quit.")
        prev_ts = time.time()

        try:                                       # FIX: try/finally guarantees cleanup
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[Warning] Frame read failed — stopping.")
                    break

                # ── Frame timing (frame-rate independent) ─────────────────────
                now_ts  = time.time()
                dt      = max(0.0, now_ts - prev_ts)
                prev_ts = now_ts

                gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.3, minNeighbors=5
                )

                # Default labels (shown even when no face is detected)
                eye_label  = "Eyes: Open"
                eye_color  = (0, 255, 0)     # green
                yawn_label = "Not Yawning"
                yawn_color = (0, 255, 255)   # cyan

                if len(faces) > 0:
                    # Pick the largest face by bounding-box area
                    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                    face_gray = gray[y: y + h, x: x + w]

                    # ── Eye detection ─────────────────────────────────────────
                    if self._update_eyes(face_gray, dt) == "Eyes: Closed":
                        eye_label = "Eyes: Closed"
                        eye_color = (0, 0, 255)   # red

                    # ── Yawn detection ────────────────────────────────────────
                    if self._update_yawn(face_gray, dt) == "Yawning":
                        yawn_label = "Yawning"
                        yawn_color = (0, 165, 255)  # orange

                else:
                    # No face detected — reset eye state to avoid stale "closed" carryover.
                    self.eye_closed_duration = 0.0
                    self.eye_consec_closed = 0
                    self.eye_consec_open = 0
                    self.eye_prob_ema = 0.0
                    self.eye_pair_prob_ema = 0.0

                    # FIX: also reset yawn EMA and open timer on no-face.
                    # Stale EMA from a previous yawn could show "Yawning"
                    # the instant the face reappears, before any inference runs.
                    self.yawn_prob_ema      = 0.0
                    self.yawn_open_duration = 0.0

                # ── Alarm logic ───────────────────────────────────────────────
                eye_alarm_now = (
                    self.eye_model is not None
                    and self.eye_closed_duration >= EYE_DROWSY_SECONDS_THRESHOLD
                )
                yawn_alarm_now = (
                    self.yawn_model is not None
                    and self.yawn_count >= YAWN_EVENT_LIMIT
                )

                if not self.is_drowsy_alarm:
                    if eye_alarm_now:
                        self.is_drowsy_alarm = True
                        self.alarm_reason = "eyes"
                    elif yawn_alarm_now:
                        self.is_drowsy_alarm = True
                        self.alarm_reason = "yawns"

                if self.is_drowsy_alarm:
                    self._play_alarm()

                # ── Draw exactly 3 labels ─────────────────────────────────────
                cv2.putText(frame, eye_label,
                            (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, eye_color, 2)
                cv2.putText(frame, yawn_label,
                            (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.85, yawn_color, 2)
                cv2.putText(frame, f"Yawn Count: {self.yawn_count}",
                            (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
                if self.is_drowsy_alarm:
                    alert_text = "Drowsy Alert!" if self.alarm_reason == "eyes" else "Too many Yawns!"
                    cv2.putText(
                        frame,
                        alert_text,
                        (10, 155),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.95,
                        (0, 0, 255),
                        2,
                    )

                cv2.imshow("Driver Drowsiness Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            # FIX: guaranteed cleanup even on Ctrl+C / unexpected exception
            cap.release()
            cv2.destroyAllWindows()
            if mixer.get_init():
                mixer.quit()


# =============================================================================
# Entry point
# =============================================================================

def main():
    try:
        DrowsinessDetector().run()
    except Exception as exc:
        print(f"\n[Fatal Error] {exc}")


if __name__ == "__main__":
    main()
