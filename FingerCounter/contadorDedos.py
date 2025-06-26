# pylint: disable=no-member
import cv2
import mediapipe as mp
import numpy as np
from math import acos, degrees


def palm_centroid(coordinates_list):
    coordinates = np.array(coordinates_list)
    centroid = np.mean(coordinates, axis=0)
    return int(centroid[0]), int(centroid[1])


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

thumb_points = [1, 2, 4]
palm_points = [0, 1, 2, 5, 9, 13, 17]
fingertips_points = [8, 12, 16, 20]
finger_base_points = [6, 10, 14, 18]

PEACH = (180, 229, 255)

with mp_hands.Hands(
        model_complexity=1,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        fingers_counter = '_'
        thickness = [2, 2, 2, 2, 2]

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                coordinates_thumb = []
                coordinates_palm = []
                coordinates_ft = []
                coordinates_fb = []

                # Dedos
                for index in thumb_points:
                    x = int(hand_landmarks.landmark[index].x * width)
                    y = int(hand_landmarks.landmark[index].y * height)
                    coordinates_thumb.append([x, y])

                for index in palm_points:
                    x = int(hand_landmarks.landmark[index].x * width)
                    y = int(hand_landmarks.landmark[index].y * height)
                    coordinates_palm.append([x, y])

                for index in fingertips_points:
                    x = int(hand_landmarks.landmark[index].x * width)
                    y = int(hand_landmarks.landmark[index].y * height)
                    coordinates_ft.append([x, y])

                for index in finger_base_points:
                    x = int(hand_landmarks.landmark[index].x * width)
                    y = int(hand_landmarks.landmark[index].y * height)
                    coordinates_fb.append([x, y])

                # Ángulo del pulgar
                p1 = np.array(coordinates_thumb[0])
                p2 = np.array(coordinates_thumb[1])
                p3 = np.array(coordinates_thumb[2])
                l1 = np.linalg.norm(p2 - p3)
                l2 = np.linalg.norm(p1 - p3)
                l3 = np.linalg.norm(p1 - p2)
                angle = degrees(acos((l1**2 + l3**2 - l2**2) / (2 * l1 * l3)))
                thumb_finger = angle > 150

                # Centro de la palma
                nx, ny = palm_centroid(coordinates_palm)
                cv2.circle(frame, (nx, ny), 3, (0, 255, 0), 2)
                coordinates_centroid = np.array([nx, ny])
                coordinates_ft = np.array(coordinates_ft)
                coordinates_fb = np.array(coordinates_fb)

                d_centroid_ft = np.linalg.norm(
                    coordinates_centroid - coordinates_ft, axis=1)
                d_centroid_fb = np.linalg.norm(
                    coordinates_centroid - coordinates_fb, axis=1)
                fingers = d_centroid_ft > d_centroid_fb
                fingers = np.append(thumb_finger, fingers)
                fingers_counter = str(np.count_nonzero(fingers == True))

                for i, finger in enumerate(fingers):
                    if finger:
                        thickness[i] = -1

                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

        # Interfaz de resultados
        cv2.rectangle(frame, (0, 0), (80, 80), (125, 220, 0), -1)
        cv2.putText(frame, fingers_counter, (15, 65), 1, 5, (255, 255, 255), 2)

        labels = ["Pulgar", "Indice", "Medio", "Anular", "Mensq1ique"]
        for i in range(5):
            x1 = 100 + i * 70
            y1 = 10
            x2 = x1 + 50
            y2 = 60
            cv2.rectangle(frame, (x1, y1), (x2, y2), PEACH, thickness[i])
            cv2.putText(frame, labels[i], (x1, 80), 1, 1, (255, 255, 255), 2)

        cv2.imshow("Finger Counter", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
