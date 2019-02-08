import face_recognition
from flask import Flask, jsonify, request, redirect

app = Flask(__name__)


@app.route('/validate-image', methods=['POST'])
def validate_image():
    if 'file' not in request.files:
        print ('NO FILE')
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
            print (errorMsg)
            result["error"] = errorMsg
        else:
            result["status"] = 'Valid image'
        return jsonify(result)


if __name__ == "__main__":
    print ('Validator init')
    app.run(host='0.0.0.0', port=3002, debug=False)
