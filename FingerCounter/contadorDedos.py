import cv2
import mediapipe as mp
import numpy as np
from math import acos, degrees


def palm_centroid(coordinates_list):

    coordinates = np.array(coordinates_list)
    centroid = np.mean(coordinates, axis=0)
    centroid = int(centroid[0]), int(centroid[1])
    return centroid


mp_drawing = mp.solution.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

thumb_points = [1, 2, 4]

palm_points = [0, 1, 2, 5, 9, 13, 17]
fingertips_points = [8, 12, 16, 20]
finger_base_points = [6, 10, 14, 18]

GREEN = (48, 255, 48)
BLUE = (192, 101, 21)
YELLOW = (0, 204, 255)
PURPLE = (128, 64, 128)
PEACH = (180, 229, 255)

with mp_hands.Hands(
        model_complexity=1,
        max_num_hands=1,
        min_detection_confidance=0.5,
        min_traking_confidance=0.5) as hands:

    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        fingers_counter = '_'
        thickness = [2, 2, 2, 2, 2]

        if results.multi_hand_landmarks:
            coordinates_thumb = []
            coordinates_palm = []
            coordinates_ft = []
            coordinates_fb = []
            for hand_landmarks in results.multi_hand_landmarks:
                for index in thumb_points:
                    x = int(
                        hand_landmarks.landmarks[index].x * width)
                    y = int(
                        hand_landmarks.landmarks[index].y * height)
                    coordinates_thumb.append([x, y])

                for index in palm_points:
                    x = int(hand_landmarks.landmarks[index].x * width)
                    y = int(hand_landmarks.landmarks[index].y * height)
                    coordinates_thumb.append([x, y])

                for index in fingertips_points:
                    x = int(hand_landmarks.landmarks[index].x * width)
                    y = int(hand_landmarks.landmarks[index].y * height)
                    coordinates_thumb.append([x, y])

                for index in finger_base_points:
                    x = int(hand_landmarks.landmarks[index].x * width)
                    y = int(hand_landmarks.landmarks[index].y * height)
                    coordinates_thumb.append([x, y])

                p1 = np.array(coordinates_thumb[0])
                p2 = np.array(coordinates_thumb[1])
                p3 = np.array(coordinates_thumb[2])
