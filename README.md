# Face Recognition Web Service

> Facial Recognition web service for edge devices

_Based on - https://github.com/ageitgey/face_recognition_

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