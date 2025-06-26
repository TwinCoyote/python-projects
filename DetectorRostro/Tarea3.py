"""Script para detección de rostros con OpenCV"""
import cv2

faceClassifier = cv2.CascadeClassifier(  # pylint: disable=no-member
    'assets/haarcascade_frontalface_default.xml')
if faceClassifier.empty():
    print("Error: No se pudo cargar el archivo XML del cascade classifier.")
    exit()
image = cv2.imread('assets/xd.jpg')       # pylint: disable=no-member

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # pylint: disable=no-member


faces = faceClassifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(
    20, 20), maxSize=(700, 700))  # pylint: disable=no-member


for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0),  # pylint: disable=no-member
                  2)


cv2.imshow('image', image)  # pylint: disable=no-member

cv2.waitKey(0)  # pylint: disable=no-member

cv2.destroyAllWindows()  # pylint: disable=no-member
