# pylint: disable=no-member
import cv2
import numpy as np
import imutils
import os


Datos = 'n'
if not os.path.exists(Datos):
    print('Carpeta Creada:', Datos)
    os.makedirs(Datos)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

x1, y1 = 220, 80
x2, y2 = 450, 360

count = 0

while True:
    ret, frame = cap.read()
    if ret == False:
        break
    imAux = frame.copy()
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    objeto = imAux[y1:y2, x1:x2]
    objeto = imutils.resize(objeto, width=38)

    k = cv2.waitKey(1)

    if k == ord('s'):
        cv2.imwrite(Datos+'/objetos_{}.jpg'.format(count),
                    objeto)  # pylint: disable=no-member
        print('Imagen guardada:'+'/objeto_{}.jpg'.format(count))
        count = count + 1

    if k == 27:
        break

    cv2.imshow('frame', frame)
    cv2.imshow('objeto', objeto)
cap.release()
cv2.destroyAllWindows()
