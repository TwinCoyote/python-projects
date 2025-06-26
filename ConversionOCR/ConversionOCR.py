# pylint: disable=no-member
import cv2
import easyocr

reader = easyocr.Reader(['en'], gpu=True)
print("Video o imagen? (v/i)", end="")
option = input().lower()
if option == "i":

    image = cv2.imread('assets/L76.jpg')  # pylint: disable=no-member
    result = reader.readtext(image, paragraph=True)

    for res in result:
        print("res:", res)
        pt0 = res[0][0]
        pt1 = res[0][1]
        pt2 = res[0][2]
        pt3 = res[0][3]

        cv2.rectangle(image, pt0, (pt1[0], pt1[1] - 23),  # pylint: disable=no-member
                      (166, 56, 242), -1)  # pylint: disable=no-member
        cv2.putText(image, res[1], (pt0[0], pt0[1]-3), 2, 0.8,  # pylint: disable=no-member
                    (0, 0, 0), 1)  # pylint: disable=no-member

        cv2.rectangle(image, pt0, pt2, (166, 56, 242),  # pylint: disable=no-member
                      2)  # pylint: disable=no-member

        cv2.circle(image, pt0, 2, (255, 0, 0), 2)  # pylint: disable=no-member
        cv2.circle(image, pt1, 2, (0, 255, 0), 2)  # pylint: disable=no-member
        cv2.circle(image, pt2, 2, (0, 0, 255), 2)  # pylint: disable=no-member
        cv2.circle(image, pt3, 2, (0, 255, 255),
                   2)  # pylint: disable=no-member

        cv2.imshow('image', image)  # pylint: disable=no-member
        cv2.waitKey(0)  # pylint: disable=no-member
    cv2.destroyAllWindows()  # pylint: disable=no-member

if option == "v":

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = reader.readtext(frame, paragraph=False)

        for res in result:
            pt0 = res[0][0]
            pt1 = res[0][1]
            pt2 = res[0][2]
            pt3 = res[0][3]

            cv2.rectangle(image, pt0, (pt1[0], pt1[1] - 23),  # pylint: disable=no-member
                          (166, 56, 242), -1)  # pylint: disable=no-member
            cv2.putText(image, res[1], (pt0[0], pt0[1]-3), 2, 0.8,  # pylint: disable=no-member
                        (0, 0, 0), 1)  # pylint: disable=no-member

            cv2.rectangle(image, pt0, pt2, (166, 56, 242),  # pylint: disable=no-member
                          2)  # pylint: disable=no-member

            cv2.circle(image, pt0, 2, (255, 0, 0),
                       2)  # pylint: disable=no-member
            cv2.circle(image, pt1, 2, (0, 255, 0),
                       2)  # pylint: disable=no-member
            cv2.circle(image, pt2, 2, (0, 0, 255),
                       2)  # pylint: disable=no-member
            cv2.circle(image, pt3, 2, (0, 255, 255),
                       2)  # pylint: disable=no-member
        cv2.imshow("OCR Result", frame)

        if cv2.waitKey(350) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
