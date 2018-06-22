# Import the required python packages.
import numpy as np
import argparse
import cv2
import os

# Construct the argument parse to parse the arguments.
argp = argparse.ArgumentParser()
argp.add_argument("-i", "--image", required=False,
	help="path to the input image")
argp.add_argument("-d", "--image_dir", required=False,
	help="path to the directory containing input image(s)")
argp.add_argument("-s", "--save_dir", required=True,
	help="path to the directory save the output image(s)")
argp.add_argument("-c", "--cascade", required=True,
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


# Load the DNN model.
print("[INFO] Loading the Cascade")
cascade = cv2.CascadeClassifier(args['cascade'])
print("[INFO] Finished loading the Cascade")

def face_bounding_box(image,cascade):
	# Convert the image to grayscale as the face detector expects a grayscale image. 
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# Use detectMultiScale to detecting faces at multiple scales.
	faces = cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5); 
	# loop over the detected faces
	boxes = []
	for face in faces:
		(x, y, w, h) = face
		boxes.append((x, y, x+w, y+h))
			
	return boxes

cascade = cv2.CascadeClassifier(args['cascade'])
for image_path in all_images_path:
	image = cv2.imread(image_path)
	rects = face_bounding_box(image,cascade)
	for rect in rects:
		(x1, y1, x2, y2) = rect
		cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

	cv2.imwrite(os.path.join(args["save_dir"],image_path.split('/')[-1]),image)
		

print "Done!"