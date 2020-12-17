import base64
import os
import socket
import cv2
import flask
import sys


from flask import render_template
from google.cloud import aiplatform
from google.cloud import automl


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
            encoded = base64.b64encode(open('facedetected.jpg', "rb").read())
            print(encoded)
            prediction=get_prediction('facedetected.jpg',"poc-smarthome-telezenm","ICN7033859556883038208")
            print(prediction)
            return prediction.preprocessed_input.message

    except:
        print("error")
        return render_template("output.html")



@app.route("/face-predict")
def faceprediction():
    try:

        encoded = base64.b64encode(open("facedetected.jpg", "rb").read())
        predictions = predict_image_classification_sample(
             "projects/44045539977/locations/us-central1/endpoints/625613320111521792",
             {"content": encoded, "mimeType": "image/jpeg"},
             {"maxPredictions": 5, "confidenceThreshold": 0.5}
         )
        print(predictions)
        return predictions

    except:
        print("error")
        encoded_string = base64.standard_b64encode(cv2.imread('facedetected.jpg'))
        print(encoded_string.decode('utf-8'))
        return render_template("prediction.html")


def get_prediction(content, project_id, model_id):
  prediction_client = automl.PredictionServiceClient()

  name = 'projects/{}/locations/us-central1/models/{}'.format(project_id, model_id)
  payload = {'image': {'image_bytes': content}}
  params = {}
  request = prediction_client.predict(name, payload, params)
  return request  # waits till request is returned

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
