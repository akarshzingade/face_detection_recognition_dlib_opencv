# Import the required python packages.
import numpy as np
import argparse
import cv2
import os
import face_detector

# Construct the argument parse to parse the arguments.
argp = argparse.ArgumentParser()
argp.add_argument("-i", "--image", required=False,
	help="path to the input image")
argp.add_argument("-d", "--image_dir", required=False,
	help="path to the directory containing input image(s)")
argp.add_argument("-s", "--save_dir", required=True,
	help="path to the directory save the output image(s)")
argp.add_argument("-m", "--load_model", required=True,
	help="path to the face recognition model file")
argp.add_argument("-r", "--face_recognition_method", type=str, default="lbph",
	help="face recognition model to use: `eigen`, `fisher` or 'lbph")
argp.add_argument("-f", "--face_detection_method", type=str, default="lbp",
	help="face detection model to use: `dnn`, `haar`, 'lbp' or 'hog")
argp.add_argument("-l", "--label", required=True,
	help="path to the labels")
argp.add_argument("-p", "--prototxt", required=False,
	help="path to the Caffe Deploy prototxt file")
argp.add_argument("-dm", "--dnn_model", required=False,
	help="path to the Caffe pre-trained model")
argp.add_argument("-c", "--cascade", required=False,
	help="path to the Cascade file")
args = vars(argp.parse_args())

if args["image"] == None and args["image_dir"] == None:
	print "[Error]: You have to either pass the path to an image or a directory containing the image(s)"

if args["image"] != None and args["image_dir"] != None:
	print "[Error]: Please pass either pass the path to an image or a directory containing the image(s)"

# Create the save directory if it doesn't exist.
if not os.path.exists(args["save_dir"]):
	os.mkdir(args["save_dir"])

# List all the input image paths
if args["image"] != None:
	all_images_path = [args["image"]]
else:
	ext = ('.jpg','.JPG','.jpeg','.JPEG','.png','.PNG')
	all_images_path = []
	for file_name in os.listdir(args["image_dir"]):
		if file_name.endswith(ext):
			all_images_path.append(os.path.join(args["image_dir"],file_name))

with open(os.path.join(args['label']),'r') as f:
    text = f.read().strip()
    people = text.split(',')


if args["face_recognition_method"] == 'lbph':
	try:
		face_recognizer = cv2.face.createLBPHFaceRecognizer()
	except:
		face_recognizer = cv2.face.LBPHFaceRecognizer_create()
elif args["face_recognition_method"] == 'fisher':
	try:
		face_recognizer = cv2.face.createFisherFaceRecognizer()
	except:
		face_recognizer = cv2.face.FisherFaceRecognizer_create()
else: 
	try:
		face_recognizer = cv2.face.createEigenFaceRecognizer()
	except:
		face_recognizer = cv2.face.EigenFaceRecognizer_create()

if args['face_detection_method'] == 'dnn':
	if args['prototxt'] == None:
		print "[Error] You have to provide the prototxt file!"
		sys.exit()
	if args['model'] == None:
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

# Load the face recognition model.
print("[INFO] Loading the model")
try:
	face_recognizer.load(args['load_model'])
except:
	face_recognizer.read(args['load_model'])
print("[INFO] Finished loading the model!")

for image_path in all_images_path:
	image = cv2.imread(image_path)
	rects = get_bounding_box(image,detector)
	for rect in rects:
		face = crop_face(image,rect)
		(x1,y1,x2,y2) = rect
		# LBPH Face Recognition requires grayscale inputs
		if args["face_recognition_method"] == 'lbph':
			face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
		prediction = face_recognizer.predict(face)
		label = people[prediction[0]]
		cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
		cv2.putText(image, label, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
	cv2.imwrite(os.path.join(args["save_dir"],image_path.split('/')[-1]),image)