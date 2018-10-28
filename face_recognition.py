import cv2
import sys
import ntpath
import os

# Get user supplied values
imagePath = sys.argv[1]
fileName = ntpath.basename(imagePath)
cascPath = "haarcascade_frontalface_default.xml"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image and convert it to gray
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image (change scaleFactor if needed)
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.2,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.CASCADE_SCALE_IMAGE
)

print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Faces found", image)

# Split complete filename for extracting base name and extension
fileParts = os.path.splitext(fileName)

# Save image with detected faces
cv2.imwrite("output/" + fileParts[0] + "_faced" + fileParts[1], image)
