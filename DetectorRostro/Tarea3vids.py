import cv2
faceClassifier = cv2.CascadeClassifier(  # pylint: disable=no-member
    'assets/haarcascade_frontalface_default.xml')
if faceClassifier.empty():
    print("Error: No se pudo cargar el archivo XML del cascade classifier.")
    exit()

cap = cv2.VideoCapture(0)  # pylint: disable=no-member

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # pylint: disable=no-member

    faces = faceClassifier.detectMultiScale(  # pylint: disable=no-member
        gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 102, 0),  # pylint: disable=no-member
                      2)

    cv2.imshow('Captura', frame)  # pylint: disable=no-member
    if cv2.waitKey(1) == ord('s'):  # pylint: disable=no-member
        break
cap.release()
cv2.destroyAllWindows()  # pylint: disable=no-member
