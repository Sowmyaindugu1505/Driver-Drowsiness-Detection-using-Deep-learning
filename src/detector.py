import cv2
import mediapipe as mp
import numpy as np
import time
from pygame import mixer

# ------------------ Thresholds ------------------

EYE_AR_THRESH = 0.23
EYE_CLOSED_TIME = 0.25
DROWSY_TIME = 3

MOUTH_AR_THRESH = 0.65
YAWN_TIME = 0.3
YAWN_LIMIT = 2

# ------------------ Alarm ------------------

mixer.init()
sound = mixer.Sound("assets/alarm.wav")

# ------------------ Mediapipe ------------------

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# ------------------ Landmark Indexes ------------------

LEFT_EYE = [33,160,158,133,153,144]
RIGHT_EYE = [362,385,387,263,373,380]
MOUTH = [13,14,78,308]

# ------------------ Helper Functions ------------------

def eye_aspect_ratio(landmarks, eye):

    p1 = landmarks[eye[0]]
    p2 = landmarks[eye[1]]
    p3 = landmarks[eye[2]]
    p4 = landmarks[eye[3]]
    p5 = landmarks[eye[4]]
    p6 = landmarks[eye[5]]

    vertical1 = np.linalg.norm(p2 - p6)
    vertical2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)

    return (vertical1 + vertical2) / (2.0 * horizontal)


def mouth_aspect_ratio(landmarks):

    top = landmarks[MOUTH[0]]
    bottom = landmarks[MOUTH[1]]
    left = landmarks[MOUTH[2]]
    right = landmarks[MOUTH[3]]

    vertical = np.linalg.norm(top - bottom)
    horizontal = np.linalg.norm(left - right)

    return vertical / horizontal


# ------------------ Variables ------------------

eye_close_start = None
yawn_start = None
yawn_count = 0

drowsy_alarm_playing = False
yawn_alarm_playing = False

# ------------------ Webcam ------------------

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame,1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = face_mesh.process(rgb)

    eye_state = "Eye: Open"
    yawn_state = "Not Yawning"

    if result.multi_face_landmarks:

        mesh = result.multi_face_landmarks[0]

        h,w,_ = frame.shape

        landmarks = []

        for lm in mesh.landmark:
            x,y = int(lm.x*w), int(lm.y*h)
            landmarks.append(np.array([x,y]))

        # ------------------ Eye Detection ------------------

        leftEAR = eye_aspect_ratio(landmarks, LEFT_EYE)
        rightEAR = eye_aspect_ratio(landmarks, RIGHT_EYE)

        ear = (leftEAR + rightEAR) / 2

        if ear < EYE_AR_THRESH:

            if eye_close_start is None:
                eye_close_start = time.time()

            duration = time.time() - eye_close_start

            if duration >= EYE_CLOSED_TIME:
                eye_state = "Eye: Closed"

            if duration >= DROWSY_TIME:

                cv2.putText(frame,"DROWSY ALERT!",
                            (300,90),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.2,
                            (0,0,255),
                            3)

                if not drowsy_alarm_playing:
                    sound.play()
                    drowsy_alarm_playing = True

        else:

            eye_close_start = None
            eye_state = "Eye: Open"
            drowsy_alarm_playing = False


        # ------------------ Yawn Detection ------------------

        mar = mouth_aspect_ratio(landmarks)

        if mar > MOUTH_AR_THRESH:

            if yawn_start is None:
                yawn_start = time.time()

            duration = time.time() - yawn_start

            if duration >= YAWN_TIME:
                yawn_state = "Yawning"

        else:

            if yawn_start is not None:

                duration = time.time() - yawn_start

                if duration >= YAWN_TIME:
                    yawn_count += 1

            yawn_start = None


    # ------------------ Too Many Yawns Alert ------------------

    if yawn_count >= YAWN_LIMIT:

        cv2.putText(frame,"Too Many Yawns!",
                    (280,90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0,0,255),
                    3)

        if not yawn_alarm_playing:
            sound.play()
            yawn_alarm_playing = True


    # ------------------ Display ------------------

    cv2.putText(frame,eye_state,(10,30),
                cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

    cv2.putText(frame,f"Yawn: {yawn_state}",
                (10,60),
                cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,0),2)

    cv2.putText(frame,f"Yawn Count: {yawn_count}",
                (10,90),
                cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

    cv2.imshow("Driver Drowsiness Detection",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
