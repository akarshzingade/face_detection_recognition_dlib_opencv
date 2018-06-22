# Import the required python packages.
import numpy as np
import argparse
import cv2
import os
import dlib
import face_detector
import sys

# Construct the argument parse to parse the arguments.
argp = argparse.ArgumentParser()
argp.add_argument("-d", "--dataset_dir", required=True,
	help="path to the directory containing input image(s)")
argp.add_argument("-s", "--save_model", required=True,
	help="path to save the trained face recognizer")
argp.add_argument("-f", "--face_detection_method", type=str, default="lbp",
	help="face detection model to use: `dnn`, `haar`, 'lbp' or 'hog")
argp.add_argument("-r", "--face_recognition_method", type=str, default="lbph",
	help="face recognition model to use: `eigen`, `fisher` or 'lbph")
argp.add_argument("-p", "--prototxt", required=False,
	help="path to the Caffe Deploy prototxt file")
argp.add_argument("-m", "--dnn_model", required=False,
	help="path to the Caffe pre-trained model")
argp.add_argument("-c", "--cascade", required=False,
	help="path to the Cascade file")
argp.add_argument("-l", "--label", required=True,
	help="path to save the labels")
args = vars(argp.parse_args())

# Create the save directory if it doesn't exist.
if not os.path.exists(args["save_model"]):
	os.mkdir(args["save_model"])

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


people = []
len_people = 0
faces = []
labels = []
ext = ('.jpg','.JPG','.jpeg','.JPEG','.png','.PNG')

def crop_face(image,face_rect):
    """
    Given the image and the face's bounding box, crop out the face.
    """
    (x1,y1,x2,y2) = face_rect
    w = abs(x2-x1)
    h = abs(y2-y1)
    return image[y1:y1 + h, x1:x1 + w]

print "[Info] Detecting faces from the dataset"
for dir_name in os.listdir(args["dataset_dir"]):
	dir_path = os.path.join(args["dataset_dir"],dir_name)
	if os.path.isdir(dir_path):
		people.append(dir_name)
		len_people += 1
		for image_name in os.listdir(dir_path):
			image_path = os.path.join(dir_path, image_name)
			if image_name.endswith(ext):
				image = cv2.imread(image_path)
				image = cv2.resize(image,(300,300))
				rects = get_bounding_box(image,detector)
				for rect in rects:
					face = crop_face(image,rect)

					# LBPH Face Recognition requires grayscale inputs
					if args["face_recognition_method"] == 'lbph':
						face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
						
					face = cv2.resize(face,(64,64))	
					faces.append(face)
					labels.append(len_people)
print "[Info] Finished detecting faces from the dataset!"

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


print "[Info] Training the Face Recognition model"
# Train the face recognizer with the faces detected in our dataset.
face_recognizer.train(faces, np.array(labels))
print "[Info] Finished training the Face Recognition model!"

print "[Info] Saving the Face Recognition model"
try:
	face_recognizer.save(os.path.join(args['save_model'],args["face_recognition_method"]+'_trained_face_recognizer'))
except:
	face_recognizer.write(os.path.join(args['save_model'],args["face_recognition_method"]+'_trained_face_recognizer'))
print "[Info] Saved the Face Recognition model!"

with open(os.path.join(args['label'],'labels.txt'),'w') as f:
    f.write( ','.join(people))







