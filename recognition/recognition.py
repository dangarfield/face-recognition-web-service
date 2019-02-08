"""
Based on - https://github.com/ageitgey/face_recognition

This is an example of using the k-nearest-neighbors (KNN) algorithm for face recognition.

When should I use this example?
This example is useful when you wish to recognize a large set of known people,
and make a prediction for an unknown person in a feasible computation time.
A web service wrapper has been added to allow this to be a stand alone service for distributed devices

Algorithm Description:
The knn classifier is first trained on a set of labeled (known) faces and can then predict the person
in an unknown image by finding the k most similar faces (images with closet face-features under eucledian distance)
in its training set, and performing a majority vote (possibly weighted) on their label.

* This implementation uses a weighted vote, such that the votes of closer-neighbors are weighted more heavily.

Installation:

1. Setup python
2. I prefer to use pm2 (node.js) for process management, this will need to be installed global also
3. run 'run-native.sh' or 'pm2 start pm2.json'

Docker:

1. Setup docker
2. Run 'docker-compose up' for local development. This will map your local code and data directory to your docker instance
3. Run 'docker-compose build' to create your production docker artefact
4. Run 'docker-compose -f docker-compose.yml -f docker-compose-prod.yml up' to run you production instance with the relevant config


Usage:

1. Clear any existing or all indexed faces by posting to '/clear-index' or '/clear-index-one'

2. Load images into training data by posting the image and id to '/index-one'. This extracts the features
   from the image into the training_data object and also persists to disk.
   Optional validation of image to match a single face in '/validate-image'

3. Once all images are added, generate index by posting '/index-generate'. This trains a K-nearest-neighbours
   classifier and holds the model in memory and persists to disk. Your service is now ready for predictions

4. To predict, post image to '/predict'. This will extract the faces and features and for each, search the KNN model for
   a best fit, returning the id.

5. The model and training data can be shared and store amongst different instances through the get and post to '/model' and '/training-data'

"""

import math
from sklearn import neighbors
import os
import os.path
import pickle
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
from flask import Flask, jsonify, request, redirect, send_file

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

MODEL_DIRECTORY = '../data/models'
TRAINING_DATA_SAVE_PATH = MODEL_DIRECTORY + '/trained_data.clf'
MODEL_SAVE_PATH = MODEL_DIRECTORY + '/trained_model.clf'

training_data = {
    "X": [],
    "y": []
}
model = None
status = 'no-model'
debug = False


@app.route('/clear-index', methods=['POST'])
def clear_index():
    global model
    print('Clear index')
    training_data['X'] = []
    training_data['y'] = []
    print('Index size: ' +
          str(len(training_data['X'])) + '(' + str(len(training_data['y'])) + ')')
    result = {
        "X": len(training_data['X']),
        "y": len(training_data['y'])
    }
    model = None
    remove_model()
    remove_training_data()

    return jsonify(result)


@app.route('/clear-index-one', methods=['POST'])
def clear_index_one():
    print('Clear index one')
    id = request.form['id']
    index = training_data['y'].index(id)
    del training_data['X'][index]
    del training_data['y'][index]

    print('Index size: ' +
          str(len(training_data['X'])) + '(' + str(len(training_data['y'])) + ')')
    result = {
        "X": len(training_data['X']),
        "y": training_data['y']
    }

    return jsonify(result)


@app.route('/validate-image', methods=['POST'])
def validate_image():
    if 'file' not in request.files:
        print('NO FILE')
        result_error = {
            "error": "No file present"
        }
        return jsonify(result_error)
    else:
        file = request.files['file']
        image = face_recognition.load_image_file(file)
        face_bounding_boxes = face_recognition.face_locations(image)
        result = {
            "face_bounding_boxes": face_bounding_boxes
        }
        if len(face_bounding_boxes) != 1:
            errorMsg = "{}".format("No faces found" if len(
                face_bounding_boxes) < 1 else "Found more than one face")
            print(errorMsg)
            result["error"] = errorMsg
        else:
            result["status"] = 'Valid image'
        return jsonify(result)


@app.route('/index-one', methods=['POST'])
def index_one():
    global training_data
    print('Index one')
    print(request)
    if 'file' not in request.files:
        print('NO FILE')

    file = request.files['file']
    id = request.form['id']

    image = face_recognition.load_image_file(file)
    face_bounding_boxes = face_recognition.face_locations(image)
    result = {
        "id": id,
        "face_bounding_boxes": face_bounding_boxes
    }
    if len(face_bounding_boxes) != 1:
        errorMsg = "{}".format("No faces found" if len(
            face_bounding_boxes) < 1 else "Found more than one face")
        print(errorMsg)
        result["error"] = errorMsg
    elif id in training_data['y']:
        print("Image for {} is already in training data".format(id))
    else:
        training_data['X'].append(face_recognition.face_encodings(
            image, known_face_locations=face_bounding_boxes)[0])
        training_data['y'].append(id)
        save_training_data()

    result["X"] = len(training_data['X'])
    result["y"] = len(training_data['y'])

    return jsonify(result)


@app.route('/index-generate', methods=['POST'])
def index_generate(knn_algo='ball_tree'):
    global model
    global status
    print('Generate index')
    X = training_data['X']
    y = training_data['y']
    training_data['in_progress'] = True
    print('Index size: ' +
          str(len(X)) + ' (' + str(len(y)) + ')')

    n_neighbors = int(round(math.sqrt(len(X))))

    if len(training_data['y']) == 0:
        print('No training data set')
    else:
        print('Training classifier')
        knn_clf = neighbors.KNeighborsClassifier(
            n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
        knn_clf.fit(X, y)
        print('Training complete')

        model = knn_clf
        save_model()
        status = 'ready'

    result = {
        "X": len(training_data['X']),
        "y": training_data['y']
    }
    return jsonify(result)


@app.route('/status', methods=['GET'])
def alive():
    global status
    result = {
        "status": status,
        "count": len(training_data['y']),
        "data": training_data['y']
    }
    return jsonify(result)


@app.route('/predict', methods=['POST'])
def predict(distance_threshold=0.6):
    if 'file' not in request.files:
        print('No file part')
        return jsonify({"error": "no file part"})
    file = request.files['file']
    if file.filename == '':
        file.filename = 'blob.jpg'
        print('Set file name')
    if file:
        print('Processing predict file')

        X_img = face_recognition.load_image_file(file)
        X_face_locations = face_recognition.face_locations(X_img)

        # If no faces are found in the image, return an empty result.
        if len(X_face_locations) == 0:
            return jsonify([])

        # Find encodings for faces in the test iamge
        faces_encodings = face_recognition.face_encodings(
            X_img, known_face_locations=X_face_locations)

        # Use the KNN model to find the best matches for the test face
        closest_distances = model.kneighbors(faces_encodings, n_neighbors=1)
        are_matches = [closest_distances[0][i][0] <=
                       distance_threshold for i in range(len(X_face_locations))]

        # Predict classes and remove classifications that aren't within the
        # threshold
        return jsonify([(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in
                        zip(model.predict(faces_encodings), X_face_locations, are_matches)])

    return jsonify({"error": "not sure - catch all"})


@app.route('/model', methods=['GET'])
def get_model():
    print('/model GET')
    MODEL_SAVE_PATH
    return send_file(MODEL_SAVE_PATH, attachment_filename='trained_model.clf', as_attachment=True)


@app.route('/model', methods=['POST'])
def set_model():
    global model
    print('/model SET')
    if 'file' not in request.files:
        print('NO FILE')
        result_error = {
            "error": "No file present"
        }
        return jsonify(result_error)
    else:
        file = request.files['file']
        model = pickle.load(file)
        result = {
            "success": "true"
        }
        return jsonify(result)


@app.route('/training-data', methods=['GET'])
def get_training_data():
    print('/training-data GET')
    return send_file(TRAINING_DATA_SAVE_PATH, attachment_filename='trained_data.clf', as_attachment=True)


@app.route('/training-data', methods=['POST'])
def set_training_data():
    global training-data
    print('/training-data SET')
    if 'file' not in request.files:
        print('NO FILE')
        result_error = {
            "error": "No file present"
        }
        return jsonify(result_error)
    else:
        file = request.files['file']
        training-data = pickle.load(file)
        result = {
            "success": "true"
        }
        return jsonify(result)


def load_model():
    global model
    global status
    if os.path.isfile(MODEL_SAVE_PATH):
        with open(MODEL_SAVE_PATH, 'rb') as f:
            model = pickle.load(f)
            status = 'ready'


def save_model():
    global model
    with open(MODEL_SAVE_PATH, 'wb') as f:
        pickle.dump(model, f)


def remove_model():
    if os.path.isfile(MODEL_SAVE_PATH):
        os.remove(MODEL_SAVE_PATH)


def load_training_data():
    global training-data
    if os.path.isfile(TRAINING_DATA_SAVE_PATH):
        with open(TRAINING_DATA_SAVE_PATH, 'rb') as f:
            training_data = pickle.load(f)


def save_training_data():
    global training_data
    with open(TRAINING_DATA_SAVE_PATH, 'wb') as f:
        pickle.dump(training_data, f)


def remove_training_data():
    if os.path.isfile(TRAINING_DATA_SAVE_PATH):
        os.remove(TRAINING_DATA_SAVE_PATH)


def ensure_file_folders_exist():
    print('Does folder not exist' + MODEL_DIRECTORY)
    if not os.path.exists(MODEL_DIRECTORY):
        print('Creating folder')
        os.makedirs(MODEL_DIRECTORY)


if __name__ == "__main__":
    print('Init')
    ensure_file_folders_exist()
    load_model()
    load_training_data()
    app.run(host='0.0.0.0', port=3002, debug=debug)
