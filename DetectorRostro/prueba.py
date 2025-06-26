import cv2
faceClassifier = cv2.CascadeClassifier(  # pylint: disable=no-member
    'assets/haarcascade_frontalface_default.xml')
if faceClassifier.empty():
    print("Error: No se pudo cargar el archivo XML del cascade classifier.")
    exit()

print("Captura de video o imagen? (v/c): ", end="")
option = input().lower()
if option == "v":
    cap = cv2.VideoCapture(0)  # pylint: disable=no-member

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(
            frame, cv2.COLOR_BGR2GRAY)  # pylint: disable=no-member

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
if option == "c":
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
