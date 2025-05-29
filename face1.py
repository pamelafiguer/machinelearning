import cv2
import numpy as np

faceClassif = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
image = cv2.imread('oficina2.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

face = faceClassif.detectMultiScale(gray,
                                   scaleFactor=1.1,
                                   minNeighbors=4,
                                   minSize=(20, 20),
                                   maxSize=(100, 100))
for (x, y, w, h) in face:
 cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()