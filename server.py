import flask
import os
import socket
import cv2

app = flask.Flask(__name__)


@app.route("/")
def hello():
    return "Hello World!"


@app.route("/face-detect")
def faceDetection():
    try:
        imageDetection()
        return "face detection completed"
    except:
        print("error")
        return "face detection completed"


def imageDetection():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture('3.MP4')

    while cap.isOpened():
        # Read the frame
        _, img = cap.read()
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        cv2.imshow('frame', gray)

        # Detect the faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # Draw the rectangle around each face
        for (x, y, w, h) in faces:
            cv2.rectangle(gray, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face = gray[y:y + h, x:x + w]
            cv2.imwrite('facedetected.jpg', face)
        # Display
        #  cv2.imshow('Video', gray)
        # Stop if escape key is pressed
        # k = cv2.waitKey(30) & 0xff
        # if k == 27:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the VideoCapture object
    cap.release()


PORT = int(os.getenv('PORT', 8000))
# Change current directory to avoid exposure of control files
# os.chdir('/static')
host_name = socket.gethostname()
host_ip = socket.gethostbyname(host_name)

# driver function
if __name__ == '__main__':
    app.run(debug=True, host=host_ip, port=PORT)
