# Can a Neural Network spot Tyler Durden? Using Dlib and OpenCV for Face Detection and Recognition

This repository is in support of this [blog](https://medium.com/@akarshzingade/can-a-neural-network-spot-tyler-durden-using-dlib-and-opencv-for-face-detection-and-recognition-fd16f744be3e). It contains code for Face Detection and Face Recognition using OpenCV and Dlib libraries. 

## Face Detection
Face Detection can be done using OpenCV's dnn module, OpenCV's Cascade Classifier and Dlib's "get_frontal_face_detector" method. 

The following scripts are for Face Detection:
1) face_detection_dnn.py
2) face_detection_cascade.py
3) face_detection_dlib.py

### face_deteciton_dnn.py 
This script uses OpenCV's dnn module to detect faces. 

```
usage: face_detection_dnn.py [-h] [-i IMAGE] [-d IMAGE_DIR] -s SAVE_DIR -p
                             PROTOTXT -m MODEL [-c CONFIDENCE_THRESHOLD]

optional arguments:
  -h, --help            show this help message and exit
  -i IMAGE, --image IMAGE
                        path to the input image
  -d IMAGE_DIR, --image_dir IMAGE_DIR
                        path to the directory containing input image(s)
  -s SAVE_DIR, --save_dir SAVE_DIR
                        path to the directory save the output image(s)
  -p PROTOTXT, --prototxt PROTOTXT
                        path to the Caffe Deploy prototxt file
  -m MODEL, --model MODEL
                        path to the Caffe pre-trained model
  -c CONFIDENCE_THRESHOLD, --confidence_threshold CONFIDENCE_THRESHOLD
                        threshold for probability to filter not-so-confident
                        detections
```

### face_detection_cascade.py
This script uses OpenCV's Cascade Classifier to detect faces. 

```
usage: face_detection_cascade.py [-h] [-i IMAGE] [-d IMAGE_DIR] -s SAVE_DIR -c
                                 CASCADE

optional arguments:
  -h, --help            show this help message and exit
  -i IMAGE, --image IMAGE
                        path to the input image
  -d IMAGE_DIR, --image_dir IMAGE_DIR
                        path to the directory containing input image(s)
  -s SAVE_DIR, --save_dir SAVE_DIR
                        path to the directory save the output image(s)
  -c CASCADE, --cascade CASCADE
                        path to the Cascade file
```

### face_detection_dlib.py
This script uses Dlib's "get_frontal_face_detector" method to detect faces.

```
usage: face_detection_dlib.py [-h] [-i IMAGE] [-d IMAGE_DIR] -s SAVE_DIR

optional arguments:
  -h, --help            show this help message and exit
  -i IMAGE, --image IMAGE
                        path to the input image
  -d IMAGE_DIR, --image_dir IMAGE_DIR
                        path to the directory containing input image(s)
  -s SAVE_DIR, --save_dir SAVE_DIR
                        path to the directory save the output image(s)
```


## Face Recognition
Face Recognition can be done using OpenCV's Eigen Face Recognizer, OpenCV's Fischer Face Recognizer, OpenCV's LBPH Face Recognizer and Dlib's Face Recognition Embeddings.

The following scripts are for Face Detection:
1) face_recognition_opencv_train.py
2) face_recognition_opencv_predict.py
3) face_recognition_dlib_train.py
4) face_recognition_dlib_predict.py

### face_recognition_opencv_train.py
This is script is used to train the Face Recognition model using OpenCV's Eigen Face Recognizer or OpenCV's Fischer Face Recognizer or OpenCV's LBPH Face Recognizer.

```
usage: face_recognition_opencv_train.py [-h] -d DATASET_DIR -s SAVE_MODEL
                                        [-f FACE_DETECTION_METHOD]
                                        [-r FACE_RECOGNITION_METHOD]
                                        [-p PROTOTXT] [-m DNN_MODEL]
                                        [-c CASCADE] -l LABEL

optional arguments:
  -h, --help            show this help message and exit
  -d DATASET_DIR, --dataset_dir DATASET_DIR
                        path to the directory containing input image(s)
  -s SAVE_MODEL, --save_model SAVE_MODEL
                        path to save the trained face recognizer
  -f FACE_DETECTION_METHOD, --face_detection_method FACE_DETECTION_METHOD
                        face detection model to use: `dnn`, `haar`, 'lbp' or
                        'hog
  -r FACE_RECOGNITION_METHOD, --face_recognition_method FACE_RECOGNITION_METHOD
                        face recognition model to use: `eigen`, `fisher` or
                        'lbph
  -p PROTOTXT, --prototxt PROTOTXT
                        path to the Caffe Deploy prototxt file
  -m DNN_MODEL, --dnn_model DNN_MODEL
                        path to the Caffe pre-trained model
  -c CASCADE, --cascade CASCADE
                        path to the Cascade file
  -l LABEL, --label LABEL
                        path to save the labels

```

### face_recognition_opencv_predict.py
This is script is used for prediction once you have trained the Face Recognition model using the above train script.

```
usage: face_recognition_opencv_predict.py [-h] [-i IMAGE] [-d IMAGE_DIR] -s
                                          SAVE_DIR -m LOAD_MODEL
                                          [-r FACE_RECOGNITION_METHOD]
                                          [-f FACE_DETECTION_METHOD] -l LABEL
                                          [-p PROTOTXT] [-dm DNN_MODEL]
                                          [-c CASCADE]

optional arguments:
  -h, --help            show this help message and exit
  -i IMAGE, --image IMAGE
                        path to the input image
  -d IMAGE_DIR, --image_dir IMAGE_DIR
                        path to the directory containing input image(s)
  -s SAVE_DIR, --save_dir SAVE_DIR
                        path to the directory save the output image(s)
  -m LOAD_MODEL, --load_model LOAD_MODEL
                        path to the face recognition model file
  -r FACE_RECOGNITION_METHOD, --face_recognition_method FACE_RECOGNITION_METHOD
                        face recognition model to use: `eigen`, `fisher` or
                        'lbph
  -f FACE_DETECTION_METHOD, --face_detection_method FACE_DETECTION_METHOD
                        face detection model to use: `dnn`, `haar`, 'lbp' or
                        'hog
  -l LABEL, --label LABEL
                        path to the labels
  -p PROTOTXT, --prototxt PROTOTXT
                        path to the Caffe Deploy prototxt file
  -dm DNN_MODEL, --dnn_model DNN_MODEL
                        path to the Caffe pre-trained model
  -c CASCADE, --cascade CASCADE
                        path to the Cascade file

```

### face_recognition_dlib_train.py
This is script is used to extract the embeddings that describe all the faces in your dataset. 

```
usage: face_recognition_dlib_train.py [-h] -d DATASET_DIR -e SAVE_ENCODINGS
                                      [-f FACE_DETECTION_METHOD] -sp
                                      SHAPE_PREDICTOR -r
                                      FACE_RECOGNITION_MODEL [-p PROTOTXT]
                                      [-m DNN_MODEL] [-c CASCADE] -l LABEL

optional arguments:
  -h, --help            show this help message and exit
  -d DATASET_DIR, --dataset_dir DATASET_DIR
                        path to the directory containing input image(s)
  -e SAVE_ENCODINGS, --save_encodings SAVE_ENCODINGS
                        path to save the face encodings
  -f FACE_DETECTION_METHOD, --face_detection_method FACE_DETECTION_METHOD
                        face detection model to use: `dnn`, `haar`, 'lbp' or
                        'hog
  -sp SHAPE_PREDICTOR, --shape_predictor SHAPE_PREDICTOR
                        path to facial landmark predictor
  -r FACE_RECOGNITION_MODEL, --face_recognition_model FACE_RECOGNITION_MODEL
                        path to Face Recognition Model
  -p PROTOTXT, --prototxt PROTOTXT
                        path to the Caffe Deploy prototxt file
  -m DNN_MODEL, --dnn_model DNN_MODEL
                        path to the Caffe pre-trained model
  -c CASCADE, --cascade CASCADE
                        path to the Cascade file
  -l LABEL, --label LABEL
                        path to save the labels

```

### face_recognition_dlib_predict.py
This is script is used for prediction once you have extracted the embedding from the faces in your dataset.

```
usage: face_recognition_dlib_predict.py [-h] [-i IMAGE] [-d IMAGE_DIR] -s
                                        SAVE_DIR -e ENCODINGS
                                        [-f FACE_DETECTION_METHOD] -sp
                                        SHAPE_PREDICTOR -r
                                        FACE_RECOGNITION_MODEL [-p PROTOTXT]
                                        [-m DNN_MODEL] [-c CASCADE] -l LABEL

optional arguments:
  -h, --help            show this help message and exit
  -i IMAGE, --image IMAGE
                        path to the input image
  -d IMAGE_DIR, --image_dir IMAGE_DIR
                        path to the directory containing input image(s)
  -s SAVE_DIR, --save_dir SAVE_DIR
                        path to the directory save the output image(s)
  -e ENCODINGS, --encodings ENCODINGS
                        path to save the face encodings
  -f FACE_DETECTION_METHOD, --face_detection_method FACE_DETECTION_METHOD
                        face detection model to use: `dnn`, `haar`, 'lbp' or
                        'hog
  -sp SHAPE_PREDICTOR, --shape_predictor SHAPE_PREDICTOR
                        path to facial landmark predictor
  -r FACE_RECOGNITION_MODEL, --face_recognition_model FACE_RECOGNITION_MODEL
                        path to Face Recognition Model
  -p PROTOTXT, --prototxt PROTOTXT
                        path to the Caffe Deploy prototxt file
  -m DNN_MODEL, --dnn_model DNN_MODEL
                        path to the Caffe pre-trained model
  -c CASCADE, --cascade CASCADE
                        path to the Cascade file
  -l LABEL, --label LABEL
                        path to save the labels

```




Note: You can get the trained model file for the Facial Landmark estimator from [here](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
