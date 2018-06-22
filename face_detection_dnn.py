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
argp.add_argument("-p", "--prototxt", required=True,
	help="path to the Caffe Deploy prototxt file")
argp.add_argument("-m", "--model", required=True,
	help="path to the Caffe pre-trained model")
argp.add_argument("-c", "--confidence_threshold", type=float, default=0.5,
	help="threshold for probability to filter not-so-confident detections")
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
print("[INFO] Loading the DNN model")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
print("[INFO] Finished loading the DNN model")

def face_bounding_box(image,net):
	(h, w) = image.shape[:2]
	# The following line resizes the image to 300x300 and normalizes each channel with mean values 
	# (104.0, 117.0, 123.0) respectively.
	blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
		(300, 300), (104.0, 117.0, 123.0))

	net.setInput(blob)
	detections = net.forward()
	boxes = []
	# loop over the detected faces
	for i in range(0, detections.shape[2]):
		# extract the confidence for the detected face
		confidence = detections[0, 0, i, 2]
	 
		# filter out detection whose confidence is less than confidence_threshold
		if confidence > args["confidence_threshold"]:
			# compute the bounding box for the detected face.
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(x1, y1, x2, y2) = box.astype("int")
			boxes.append((x1, y1, x2, y2))
			
	return boxes

for image_path in all_images_path:
	image = cv2.imread(image_path)
	rects = face_bounding_box(image,net)
	for rect in rects:
		(x1, y1, x2, y2) = rect
		cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

	cv2.imwrite(os.path.join(args["save_dir"],image_path.split('/')[-1]),image)
		

print "Done!"