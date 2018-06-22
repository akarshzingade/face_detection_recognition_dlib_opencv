# Import the required python packages.
import numpy as np
import argparse
import cv2
import os
import dlib

# Construct the argument parse to parse the arguments.
argp = argparse.ArgumentParser()
argp.add_argument("-i", "--image", required=False,
	help="path to the input image")
argp.add_argument("-d", "--image_dir", required=False,
	help="path to the directory containing input image(s)")
argp.add_argument("-s", "--save_dir", required=True,
	help="path to the directory save the output image(s)")
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


# Load the DNN model.
print("[INFO] Loading the Face Detector.")
face_detector = dlib.get_frontal_face_detector()
print("[INFO] Finished loading the Face Detector.")

def face_bounding_box(image,face_detector):
	# Get the bounding boxes for all the detected faces
	rects = face_detector(image,1)

	boxes = []

	# loop over the bounding boxes
	for face_rect in rects:
		y1 = face_rect.top()
		x1 = face_rect.left()
		y2 = face_rect.bottom()
		x2 = face_rect.right()
		#w = abs(x2-x1)
		#h = abs(y2-y1)
		boxes.append((x1,y1,x2,y2))

	return boxes


for image_path in all_images_path:
	image = cv2.imread(image_path)
	rects = face_bounding_box(image,face_detector)
	for rect in rects:
		(x1, y1, x2, y2) = rect
		cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

	cv2.imwrite(os.path.join(args["save_dir"],image_path.split('/')[-1]),image)
		

print "Done!"