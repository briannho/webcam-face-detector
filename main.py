# bin/bash/python3

"""
@author: Brian Ho
"""

import os

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

from webcam_face_detector import WebcamFaceDetector

def process_image(image, face_detector, model):
    """
        Process each image and calculate its embedding
    """
    image = image.convert('RGB')

    # detect face from image
    x1, y1, width, height = face_detector.detect(image)[0][0].astype(int)
    x1, y1 = abs(x1), abs(y1)

    # crop image to get just the face and normalize
    face = np.asarray(image)[y1:height, x1:width]
    resized_face = Image.fromarray(face).resize((160,160))
    final_face = np.asarray(resized_face) / 255

    # calculate embedding
    face_tensor = torch.from_numpy(final_face.astype(np.float32)
                                  ).reshape(1,3,160,160)
    embedding = model(face_tensor)

    return embedding

def main():
	# initialize models
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	mtcnn = MTCNN()
	resnet = InceptionResnetV1(pretrained='vggface2',
	                           device=device)
	resnet.eval()

	# process each image and treat them as verified identities
	known_embeddings = []
	known_names = []
	listdir = os.listdir('train_data/')

	for person in listdir:
	    pics = os.listdir("train_data/" + person)
	    
	    for pic in pics:
	        image = Image.open(f"train_data/{person}/{pic}")
	        embedding = process_image(image=image,
	                                  face_detector=mtcnn,
	                                  model=resnet)

	        known_embeddings.append(embedding)
	        known_names.append(person)

	# instantiate face detector
	wfd = WebcamFaceDetector(face_detector=mtcnn,
	                         model=resnet,
	                         known_embeddings=known_embeddings,
	                         known_names=known_names)

	# start webcam
	cap = cv2.VideoCapture(0)

	# initialize vars
	face_boxes = []
	names = []
	process_frame = True

	while True:
	    ret, frame = cap.read()

	    # try except block to prevent code break when a face is not detected
	    try:
	        # process every other frame to save time on classifying
	        if process_frame:
	            # Resize frame of video to 1/4 size for faster processing
	            small_frame = cv2.resize(frame,
	                                     (0, 0),
	                                     fx=0.25,
	                                     fy=0.25)

	            # Convert from BGR to RGB
	            rgb_small_frame = small_frame[:, :, ::-1]

	            # detect face box
	            boxes, _ = wfd.detect_face(rgb_small_frame)
	            face_boxes = boxes
	            names = []

	            # for each face, predict identity
	            for box in zip(boxes): # left top right bottom 
	                name = wfd.classify(frame, box)
	                names.append(name)

	        process_frame = not process_frame

	        # draw box and identity around each face
	        for box, name in zip(face_boxes, names):
	            wfd.draw_box(frame, box, name)

	    except:
	        pass

	    # Show the frame
	    cv2.imshow('Video', frame)

	    if cv2.waitKey(1) & 0xFF == ord('q'):
	        break

	cap.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()