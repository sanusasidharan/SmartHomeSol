import base64
import os
import socket
import cv2
import flask
import sys

from PIL import Image
from flask import render_template
from google.cloud import aiplatform
from google.cloud import automl_v1beta1



app = flask.Flask(__name__)


@app.route("/")
def hello():
    return render_template("index.html")


@app.route("/hello")
def hellos():
    return "Hello World!"


@app.route("/face-detect")
def faceDetection():
    try:
        imageDetection()
        return render_template("output.html")

    except:
        print("error")
        return render_template("output.html")


@app.route("/face-autopredict")
def faceautopredict():
    try:
            #poc-smarthome-telezenm
            imagetocheck = Image.open('facedetected.jpg')
            return get_prediction(imagetocheck,"44045539977","ICN7033859556883038208")

    except:
        print("error")
        return render_template("output.html")



@app.route("/face-predict")
def faceprediction():
    try:

        imagetocheck = Image.open('facedetected.jpg')
        #encoded_string = base64.standard_b64encode(cv2.imread('facedetected.jpg'))
        #print(encoded_string.decode('utf-8'))
        # image = "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"
        predictions = predict_image_classification_sample(
             "projects/44045539977/locations/us-central1/endpoints/625613320111521792",
             {"content": imagetocheck, "mimeType": "image/jpeg"},
             {"maxPredictions": 5, "confidenceThreshold": 0.5}
         )
        return predictions

    except:
        print("error")
        encoded_string = base64.standard_b64encode(cv2.imread('facedetected.jpg'))
        print(encoded_string.decode('utf-8'))
        #return (encoded_string.decode('utf-8'))
        return render_template("prediction.html")


def get_prediction(content, project_id, model_id):
  prediction_client = automl_v1beta1.PredictionServiceClient()

  name = 'projects/{}/locations/us-central1/models/{}'.format(project_id, model_id)
  payload = {'image': {'image_bytes': content }}
  params = {}
  request = prediction_client.predict(name, payload, params)
  return request  # waits till request is returned

if __name__ == '__main__':
  file_path = sys.argv[1]
  project_id = sys.argv[2]
  model_id = sys.argv[3]

  with open(file_path, 'rb') as ff:
    content = ff.read()






def predict_image_classification_sample(
        endpoint: str, instance: dict, parameters_dict: dict
):
    client_options = dict(api_endpoint="us-central1-prediction-aiplatform.googleapis.com")
    client = aiplatform.PredictionService(client_options=client_options)
    from google.protobuf import json_format
    from google.protobuf.struct_pb2 import Value

    # See gs://google-cloud-aiplatform/schema/predict/params/image_classification_1.0.0.yaml for the format of the parameters.
    parameters = json_format.ParseDict(parameters_dict, Value())

    # See gs://google-cloud-aiplatform/schema/predict/instance/image_classification_1.0.0.yaml for the format of the instances.
    instances_list = [instance]
    instances = [json_format.ParseDict(s, Value()) for s in instances_list]
    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )

    print("response")
    print(" deployed_model_id:", response.deployed_model_id)
    predictions = response.predictions
    print("predictions")
    for prediction in predictions:
        # See gs://google-cloud-aiplatform/schema/predict/prediction/classification_1.0.0.yaml for the format of the predictions.
        print(" prediction:", dict(prediction))

    return predictions


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
