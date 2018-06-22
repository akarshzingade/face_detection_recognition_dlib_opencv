# Import the required python packages.
import numpy as np
import argparse
import cv2
import os
import dlib
import face_detector
import sys
import pickle

# Construct the argument parse to parse the arguments.
argp = argparse.ArgumentParser()
argp.add_argument("-d", "--dataset_dir", required=True,
	help="path to the directory containing input image(s)")
argp.add_argument("-e", "--save_encodings", required=True,
	help="path to save the face encodings")
argp.add_argument("-f", "--face_detection_method", type=str, default="lbp",
	help="face detection model to use: `dnn`, `haar`, 'lbp' or 'hog")
argp.add_argument("-sp", "--shape_predictor", required=True,
	help="path to facial landmark predictor")
argp.add_argument("-r", "--face_recognition_model", required=True,
	help="path to Face Recognition Model")
argp.add_argument("-p", "--prototxt", required=False,
	help="path to the Caffe Deploy prototxt file")
argp.add_argument("-m", "--dnn_model", required=False,
	help="path to the Caffe pre-trained model")
argp.add_argument("-c", "--cascade", required=False,
	help="path to the Cascade file")
argp.add_argument("-l", "--label", required=True,
	help="path to save the labels")
args = vars(argp.parse_args())

if not os.path.exists(args["face_recognition_model"]):
    print "The Face Recognition model file does not exist!"
    sys.exit()

if not os.path.exists(args["shape_predictor"]):
    print "The Shape Predictor file does not exist!"
    sys.exit()

# Create the save directory if it doesn't exist.
if not os.path.exists(args["save_encodings"]):
	os.mkdir(args["save_encodings"])


if args['face_detection_method'] == 'dnn':
	if args['prototxt'] == None:
		print "[Error] You have to provide the prototxt file!"
		sys.exit()
	if args['dnn_model'] == None:
		print "[Error] You have to provide the model file!"
		sys.exit()
	detector = cv2.dnn.readNetFromCaffe(args["prototxt"], args["dnn_model"])
	get_bounding_box = face_detector.face_bounding_box_dnn

if args['face_detection_method'] == 'haar' or args['face_detection_method'] == 'lbp':
	if args['cascade'] == None:
		print "[Error] You have to provide the prototxt file!"
		sys.exit()
	detector = cv2.CascadeClassifier(args['cascade'])
	get_bounding_box = face_detector.face_bounding_box_cascade

if args['face_detection_method'] == 'hog':
	detector = dlib.get_frontal_face_detector()
	get_bounding_box = face_detector.face_bounding_box_hog


def crop_face(image,face_rect):
    """
    Given the image and the face's bounding box, crop out the face.
    """
    (x1,y1,x2,y2) = face_rect
    w = abs(x2-x1)
    h = abs(y2-y1)
    return image[y1:y1 + h, x1:x1 + w]

def get_encoding(image,rect,shape_predictor,face_recognition_model):
	#print image.shape,face.shape
	shapes_face = shape_predictor(image, rect)
	encoding = np.array(face_recognition_model.compute_face_descriptor(image, shapes_face, 1))
	return encoding

encodings = []
labels = []

ext = ('.jpg','.JPG','.jpeg','.JPEG','.png','.PNG')

# Load the Shaper Predictor model
shape_predictor = dlib.shape_predictor(args["shape_predictor"])

# Load the Face Recognition model
face_recognition_model = dlib.face_recognition_model_v1(args["face_recognition_model"])

print "[Info] Detecting faces from the dataset and encoding it"
for dir_name in os.listdir(args["dataset_dir"]):
	dir_path = os.path.join(args["dataset_dir"],dir_name)
	if os.path.isdir(dir_path):
		for image_name in os.listdir(dir_path):
			image_path = os.path.join(dir_path, image_name)
			if image_name.endswith(ext):
				image = cv2.imread(image_path)
				rects = get_bounding_box(image,detector)
				for rect in rects:
					(x1,y1,x2,y2) = rect
					rect = dlib.rectangle(left=x1, top=y1, right=x2, bottom=y2)
					encodings.append(get_encoding(image,rect,shape_predictor,face_recognition_model))
					labels.append(dir_name)
					
print "[Info] Finished detecting and encoding faces from the dataset!"
print len(encodings)
f = open(os.path.join(args["save_encodings"],'dlib_face_encoding'), "wb")
f.write(pickle.dumps(encodings))
f.close()

f = open(os.path.join(args["save_encodings"],'dlib_face_encoding_labels'), "wb")
f.write(pickle.dumps(labels))
f.close()