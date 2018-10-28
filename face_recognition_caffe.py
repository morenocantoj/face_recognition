import cv2
import sys
import ntpath
import os
import numpy as np
import argparse

# Get user supplied values
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# For set the output later
fileName = ntpath.basename(args["image"])

# Load serialized model from path
print("Loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# Load image and construct an input blob for the image resizing it to 300x300 px
image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
	(300, 300), (104.0, 177.0, 123.0))

# Pass the blog through network to obtain the detections and predictions
print("Computing object detections...")
net.setInput(blob)
detections = net.forward()

for i in range(0, detections.shape[2]):
	# Extract the confidence (i.e., probability) associated with the prediction
	confidence = detections[0, 0, i, 2]

	# Filter out weak detections by ensuring the `confidence` is
	# greater than the minimum confidence
	if confidence > args["confidence"]:
		# Compute the (x, y)-coordinates of the bounding box for the object
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		# Draw the bounding box of the face along with the probability
		text = "{:.2f}%".format(confidence * 100)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(image, (startX, startY), (endX, endY),
			(0, 0, 255), 2)
		cv2.putText(image, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

# Show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)
