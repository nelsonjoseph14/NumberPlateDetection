import cv2
import numpy as np
import imutils
from matplotlib import pyplot as plt

class FaceDetector:
    def __init__(self, cascadepath):
        self.cascade = cv2.CascadeClassifier(cascadepath)

    def detect_face(self, image):
        rects = self.cascade.detectMultiScale(image, scaleFactor=1.5, minNeighbors=5, minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE)
        return rects

#image = cv2.imread("check.jpeg")
camera = cv2.VideoCapture(0)

while True:
    _, frame = camera.read()
    frame = imutils.resize(frame, width=300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fd = FaceDetector("haarcascade_russian_plate_number.xml")

    rectan = fd.detect_face(gray)

    for (x, y, w, h) in rectan:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("face", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
