# Import the required python packages.
import numpy as np
import cv2

def face_bounding_box_dnn(image,net):
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

def face_bounding_box_cascade(image,cascade):
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

def face_bounding_box_hog(image,face_detector):
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