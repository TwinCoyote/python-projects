# pylint: disable=no-member
import cv2
import numpy as np
import imutils

cap = cv2.VideoCapture('assets/VC.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2(
    history=500, varThreshold=100, detectShadows=True)

fgbg.setVarThreshold(500)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
car_dict = {}
car_counter = 0

# Línea de conteo
line_x = 150
line_y = 300
min_contour_area = 1500  # Más alto para evitar ruido

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=640)

    # Define área de interés
    # area_pts = np.array([[80, 158], [frame.shape[1]-530, 158],
   #                      [frame.shape[1]-330, 216], [80, 216]])
    area_pts = np.array([
        [0, 120],
        [640, 120],
        [640, 270],
        [0, 270]])

    imAux = np.zeros(shape=frame.shape[:2], dtype=np.uint8)
    imAux = cv2.drawContours(imAux, [area_pts], -1, 255, -1)
    image_area = cv2.bitwise_and(frame, frame, mask=imAux)

    # Procesamiento de la máscara
    fgmask = fgbg.apply(image_area)
    fgmask[fgmask == 127] = 0
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    fgmask = cv2.dilate(fgmask, None, iterations=5)

    cnts, _ = cv2.findContours(
        fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area > min_contour_area:
            x, y, w, h = cv2.boundingRect(cnt)
            cx = x + w // 2
            cy = y + h // 2
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 1)
            line_y = 300

            # Si cruza la línea de conteo
            if (line_y - 10) < cy < (line_y + 10):
                if id(cnt) not in car_dict:
                    car_counter += 1
                    car_dict[id(cnt)] = True
                    # verde si detecta
                    cv2.line(frame, (0, line_y),
                             (frame.shape[1], line_y), (0, 255, 0), 4)

    # Limpiar auto_dict de objetos lejanos para evitar crecimiento indefinido
    if len(car_dict) > 1000:
        car_dict.clear()

    # Dibujar zona y línea de conteo
   # cv2.drawContours(frame, [area_pts], -1, (0, 127, 255), 2)
    line_y = 300
    # línea de referencia
    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 0, 255), 2)
    # cv2.rectangle(frame, (5, 158), (70, 216), (0, 255, 0), 2)
    cv2.rectangle(frame, (0, 0), (50, 50), (0, 255, 0), 2)
    cv2.putText(frame, str(car_counter), (13, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow('frame', frame)
    cv2.imshow('fgmask', fgmask)
    exit_key = cv2.waitKey(10) & 0xFF
    if exit_key == 27:  # Esc
        break

cap.release()
cv2.destroyAllWindows()
