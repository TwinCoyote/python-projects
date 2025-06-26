# pylint: disable=no-member
# pylint: disable=invalid-name
import cv2
import numpy as np
import imutils

cap = cv2.VideoCapture('assets/VC.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2(
    history=500, varThreshold=100, detectShadows=True)
fgbg.setVarThreshold(290)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

line_y = 280
min_contour_area = 1500

car_counter = 0
car_tracks = {}  # id : {'centroid': (x,y), 'counted': True/False}
next_car_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=640)

    # Define área de interés
    area_pts = np.array([
        [0, 120],
        [470, 120],
        [640, 320],
        [0, 320]])

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

    current_centroids = []

    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area > min_contour_area:
            x, y, w, h = cv2.boundingRect(cnt)
            cx = x + w // 2
            cy = y + h // 2
            current_centroids.append((cx, cy))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    # Asociación simple por distancia
    new_tracks = {}
    for (cx, cy) in current_centroids:
        assigned = False
        for car_id, data in car_tracks.items():
            pcx, pcy = data['centroid']
            dist = abs(cx - pcx) + abs(cy - pcy)
            if dist < 90:
                new_tracks[car_id] = {'centroid': (
                    cx, cy), 'counted': data['counted']}
                assigned = True
                # Solo cuenta si no se ha contado y cruza la línea
                if not data['counted'] and pcy < line_y <= cy:
                    car_counter += 1
                    new_tracks[car_id]['counted'] = True
                break
        if not assigned:
            new_tracks[next_car_id] = {'centroid': (cx, cy), 'counted': False}
            next_car_id += 1

    car_tracks = new_tracks

    # Dibujos
    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 0, 255), 2)
    cv2.putText(frame, str(car_counter), (13, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
   # cv2.polylines(frame, [area_pts], isClosed=True, color = (0, 127, 255), thickness = 2)

    cv2.imshow('frame', frame)
    cv2.imshow('fgmask', fgmask)
    exit_key = cv2.waitKey(1) & 0xFF
    if exit_key == 27:
        break

cap.release()
cv2.destroyAllWindows()
