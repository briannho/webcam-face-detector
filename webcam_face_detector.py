# bin/bash/python3

"""
@author: Brian Ho
"""

import numpy as np
import os

import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

class WebcamFaceDetector(object):
    """
    Face classifier class
    """

    def __init__(self, face_detector, model, known_embeddings, known_names):
        self.face_detector = face_detector
        self.model = model
        self.known_embeddings = known_embeddings
        self.known_names = known_names

    def draw_box(self, frame, box, name):
        """
            Draw a box around detected face with predicted identity
        """
        # draw box around face
        x1, y1, width, height = box.astype(int) * 4
        x1, y1 = abs(x1), abs(y1)
        cv2.rectangle(frame,
                      (x1, y1),
                      (width, height),
                      (0, 0, 255),
                      2)

        # add name label below box
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.rectangle(frame,
                      (x1, height - 35),
                      (width, height),
                      (0, 0, 255),
                      cv2.FILLED)
        cv2.putText(frame,
                    name,
                    (x1 + 6, height - 6),
                    font,
                    1.0,
                    (255, 255, 255),
                    1)

        return frame

    def detect_face(self, frame):
        """
            Return the bounding boxes of the faces and probabilities
        """
        boxes, probs = self.face_detector.detect(frame)
        return boxes, probs


    def classify(self, frame, box):
        """
            Predict the identity of a face
        """
        # get coordinates of bounding box
        x1, y1, width, height = box[0].astype(int) * 4
        x1, y1 = abs(x1), abs(y1)

        # crop an image of just the face and normalize
        face = np.asarray(frame)[y1:height, x1:width]
        resized_face = Image.fromarray(face).resize((160,160))
        final_face = np.asarray(resized_face) / 255 

        # calculate embedding of face
        face_tensor = torch.from_numpy(final_face.astype(np.float32)
                                      ).reshape(1,3,160,160)
        embedding = self.model(face_tensor)

        # compare embedding to known embeddings and select best match
        name = 'unknown'
        min_dist = 100
        for i, known_embedding in enumerate(self.known_embeddings):
            dist = np.linalg.norm(embedding.detach().numpy() - 
                                  known_embedding.detach().numpy()
                                 )
            
            if (dist < 0.7) and (dist < min_dist):
                name = self.known_names[i]
                min_dist = dist

        return name