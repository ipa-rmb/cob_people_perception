#!/usr/bin/env python
"""OpenfaceWrapper.py.

This module is a high level interface for OpenFace library.

Attributes:
    file_dir(string): Directory of OpenFace in your PC.

    dlib_face_predictor(string): Name of the dlib face keypoint finder model

    network_model(string): Name of the OpenFace feature extractor model

    img_dimension(int): Dimension that all images are normalized to.

    align(openface.AlignDlib): Face aligner

    net(openface.TorchNeuralNet): Neural Network for feature extraction

Author:
    Cagatay Odabasi -- cagatay.odabasi@ipa.fraunhofer.de
"""

import os

# OpenCV
import cv2

# numpy
import numpy as np

# OpenFace
import openface

class OpenFaceWrapper(object):
    """OpenFaceWrapper provides multiple and single face detections as well
    as feature extraction from the faces by using Openface API
    """
    def __init__(self, openface_directory="/home/cag/openface/openface/"):
        super(OpenFaceWrapper, self).__init__()
        self.file_dir = os.path.dirname(openface_directory)
        model_dir = os.path.join(self.file_dir, 'models')
        dlibmodel_dir = os.path.join(model_dir, 'dlib')
        openfacemodel_dir = os.path.join(model_dir, 'openface')

        self.dlib_face_predictor = os.path.join(dlibmodel_dir,
            "shape_predictor_68_face_landmarks.dat")
        self.network_model = os.path.join(openfacemodel_dir,
            'nn4.small2.v1.t7')
        self.img_dimension = 96

        self.align = openface.AlignDlib(self.dlib_face_predictor)
        self.net = openface.TorchNeuralNet(self.network_model, \
            self.img_dimension)

        self.im = np.zeros((480, 640, 3), np.uint8)

    def detect_face(self, im):
        """
        Detects a single face in a RGB image

        Args:
            im(numpy.ndarray): Image

        Returns:
            face: cropped face image in Numpy array format
            bounding_box: bounding box of the face in dlib.Rectangle format
            landmarks: list of x,y tuples
        """

        landmarks = None

        # Detect and get the bounding box of image
        bounding_box = self.align.getLargestFaceBoundingBox(im)

        if bounding_box is not None:
            landmarks = self.align.findLandmarks(im, bounding_box)

        # Align the face
        face = self.align.align(self.img_dimension, im, bounding_box,
            landmarks=landmarks,
            landmarkIndices=openface.AlignDlib.
            OUTER_EYES_AND_NOSE)

        return (face, bounding_box, landmarks)

    def find_landmarks(self, image, bounding_box):
        """
        Finds the landmarks of a given face

        Args:
            image(numpy.ndarray): rgb image
            bounding_box(dlib.rectangle): bounding box of face

        Returns:
            landmarks(tuple): landmark coordinates
        """
        try:
            return self.align.findLandmarks(image, bounding_box)
        except Exception as e:
            return None

    def extract_features(self, face):
        """
        Extracts the features using deep net of a given face

        Args:
            face: face image in Numpy array format

        Returns:
            features: (128,1) feature vector in Numpy.Array format

        """
        return self.net.forward(face)

    def align_face(self, face, bounding_box):
        """
        Aligns the face using Eyes and Nose

        Args:
            face: Face image in Numpy format
            bounding box: (dlib.Rectangle) bounding box of face

        Returns:
            aligned face: in Numpy format
        """
        # Align the face
        return self.align.align(self.img_dimension, face, bounding_box,
            landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

    def calculate_distance(self, features_1, features_2):
        """
        Calculates angle between two face feature vectors

        Args:
            features_1: (128,1) feature vector of face 1 in Numpy.Array format
            features_2: (128,1) feature vector of face 2 in Numpy.Array format

        Returns:
            distance: distance between vectors in float

        """

        return  np.arccos(np.dot(features_1, features_2))

    def estimate_head_rotation(self, landmarks, image):
        """
        Estimates the rough rotation of face using the landmarks of faces.
        This functions is acquired from
        http://www.learnopencv.com/head-pose-estimation-using-opencv-and-dlib/
        and modified to fit in this class.

        Args:
            landmarks: list of (x,y) tuples (generated from dlib)
            image: image opencv

        Returns:
            estimated rotation vector: (3.1) numpy.array()

        """

        size = image.shape

        #2D image points. If you change the image, you need to change vector
        image_points = np.array([
                                    landmarks[30],     # Nose tip
                                    landmarks[8],     # Chin
                                    landmarks[36],     # Left eye left corner
                                    landmarks[45],     # Right eye right corne
                                    landmarks[48],     # Left Mouth corner
                                    landmarks[54]   # Right mouth corner
                                ], dtype="double")

        # 3D model points.
        model_points = np.array([
                                (0.0, 0.0, 0.0),         # Nose tip
                                (0.0, -330.0, -65.0),    # Chin
                                (-225.0, 170.0, -135.0), # Left eye left corner
                                (225.0, 170.0, -135.0),  # Right eye right corne
                                (-150.0, -150.0, -125.0),# Left Mouth corner
                                (150.0, -150.0, -125.0)  # Right mouth corner
                                ])


        # Camera internals

        focal_length = size[1]
        center = (size[1]/2, size[0]/2)
        camera_matrix = np.array(
                                 [[focal_length, 0, center[0]],
                                 [0, focal_length, center[1]],
                                 [0, 0, 1]], dtype="double"
                                 )

        dist_coeffs = np.zeros((4, 1)) # Assuming no lens distortion

        (success, rotation_vector, translation_vector) = \
            cv2.solvePnP(model_points, \
            image_points, camera_matrix, dist_coeffs, flags=cv2.CV_ITERATIVE)

        # Project a 3D point (0, 0, 1000.0) onto the image plane.
        # We use this to draw a line sticking out of the nose

        (nose_end_point2D, jacobian) = \
            cv2.projectPoints(np.array([(0.0, 0.0, 500.0)]), \
            rotation_vector, translation_vector, camera_matrix, dist_coeffs)

        #for pts in image_points:
        #    cv2.circle(image, (int(pts[0]), int(pts[1])), 3, (0, 0, 255), -1)


        pts_1 = int(image_points[0][0]), int(image_points[0][1])
        pts_2 = int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1])

        #cv2.line(image, pts_1, pts_2, (255, 0, 0), 2)

        # Display image
        cv2.imshow("Output", image)
        cv2.waitKey(20)

        return rotation_vector
