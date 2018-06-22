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
argp.add_argument("-i", "--image", required=False,
	help="path to the input image")
argp.add_argument("-d", "--image_dir", required=False,
	help="path to the directory containing input image(s)")
argp.add_argument("-s", "--save_dir", required=True,
	help="path to the directory save the output image(s)")
argp.add_argument("-e", "--encodings", required=True,
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
if not os.path.exists(args["save_dir"]):
	os.mkdir(args["save_dir"])

# Create the save directory if it doesn't exist.
if not os.path.exists(args["encodings"]):
	os.mkdir(args["encodings"])

# List all the input image paths
if args["image"] != None:
	all_images_path = [args["image"]]
else:
	ext = ('.jpg','.JPG','.jpeg','.JPEG','.png','.PNG')
	all_images_path = []
	for file_name in os.listdir(args["image_dir"]):
		if file_name.endswith(ext):
			all_images_path.append(os.path.join(args["image_dir"],file_name))


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

# Load the Shaper Predictor model
shape_predictor = dlib.shape_predictor(args["shape_predictor"])

# Load the Face Recognition model
face_recognition_model = dlib.face_recognition_model_v1(args["face_recognition_model"])

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

def euclidean(vec1, vec2):
    """
    Calculate the euclidean distance.
    """
    assert len(vec1) == len(vec2)
    dist = (sum([(vec1[idx] - vec2[idx])**(2) for idx in range(len(vec1))]))**(0.5)
    return dist


print("[INFO] Loading encodings...")
encodings = pickle.loads(open(args["encodings"], "rb").read())
print("[INFO] Loaded encodings...")

print("[INFO] Loading labels...")
labels = pickle.loads(open(args["label"], "rb").read())
print("[INFO] Loaded labels...")

for image_path in all_images_path:
	image = cv2.imread(image_path)
	rects = get_bounding_box(image,detector)
	for rect in rects:
		(x1,y1,x2,y2) = rect
		rect = dlib.rectangle(left=x1, top=y1, right=x2, bottom=y2)
		encoding = get_encoding(image,rect,shape_predictor,face_recognition_model)
		matches = [] 
		for enc in encodings:
			if euclidean(enc,encoding) <= 0.7:
				matches.append(True)
			else:
				matches.append(False)

		if True in matches:
			matchIdx = [i for (i, is_true) in enumerate(matches) if is_true]
			counts = {}
			for i in matchIdx:
				label = labels[i]
				counts[label] = counts.get(label, 0) + 1

			predicted_label = max(counts, key=counts.get)
			cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
			cv2.putText(image, label, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
	cv2.imwrite(os.path.join(args["save_dir"],image_path.split('/')[-1]),image)