import cv2
import time
import pygame
from scipy.spatial import distance as dist

# Initialize pygame mixer for sound alerts
pygame.mixer.init()
pygame.mixer.music.load("alert.wav")

# Eye aspect ratio calculation
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Load haarcascade classifiers
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')

# EAR threshold and frame count
EAR_THRESHOLD = 0.22
CONSEC_FRAMES = 20
counter = 0

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) < 2:
            counter += 1
            if counter >= CONSEC_FRAMES:
                cv2.putText(frame, "DROWSY!", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                pygame.mixer.music.play()
        else:
            counter = 0

    cv2.imshow('Drowsiness Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
