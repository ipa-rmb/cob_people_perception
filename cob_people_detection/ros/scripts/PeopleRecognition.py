#!/usr/bin/env python

#import os
import itertools
import pickle

# OpenFace
#import openface
import dlib

# scikit learn
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier#, KDTree, BallTree
from sklearn import svm
#from sklearn import svm

from scipy import spatial

# OpenCV
import cv2

# numpy
import numpy as np

from cob_perception_msgs.msg import  DetectionArray, Rect#, Mask, Detection
from cob_perception_msgs.msg import ColorDepthImageArray

import util
from util import Timer

#from OpenfaceWrapper import OpenFaceWrapper


################################################################################
# People Recognition
#######################People Recognition#######################################

class PeopleRecognition(object):
    """Class for people recognition from 3D camera"""
    def __init__(self, openface_wrapper, pub, hard_threshold=0.70, \
        soft_threshold=0.99, number_of_feature_threshold=20, \
        min_abs_pitch=2.9, max_abs_pitch=3.4, max_abs_yaw=0.40, \
        max_abs_roll=1, \
        dbscan_eps=0.60, dbscan_min_samples=5):
        super(PeopleRecognition, self).__init__()
        self.openface_wrapper = openface_wrapper
        self.hard_threshold = hard_threshold
        self.soft_threshold = soft_threshold
        self.number_of_feature_threshold = number_of_feature_threshold
        self.min_abs_pitch = min_abs_pitch
        self.max_abs_pitch = max_abs_pitch
        self.max_abs_yaw = max_abs_yaw
        self.max_abs_roll = max_abs_roll
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.pub = pub

        self.classifier = KNeighborsClassifier(n_neighbors=5)
        self._people_list = [0]
        self._people_features_list = []
        self.detection_array = DetectionArray()


    def assign_names_to_people_list(self, names):
        """
        Assigns name to each id in the people list

        Args:
            names(dictionary):  0 : cagatay, 1 : john_doe, etc..

        Returns:
        """

        for idx, people in enumerate(self._people_list):

            try:
                self._people_list[idx] = names[str(people)]
            except:
                print "Error: The label cannot be updated!"
            else:
                self._people_list[idx] = names[str(people)]
        print self._people_list

    def delete_entries_with_label(self, label):
        """
        Deletes the entries from people_list with a certain label, it also
        deletes the classifier

        Args:
            label(string): label of the entry to be deleted

        Returns:
        """

        # Find the idices that will be deleted
        items_to_delete = []
        for idx, people in enumerate(self._people_list):

            if str(people) == label:
                items_to_delete.append(idx)

        # Delete the items
        for idx in sorted(items_to_delete, reverse=True):
            del self._people_list[idx]
            del self._people_features_list[idx]

        # Reset the classifier
        self.classifier = KNeighborsClassifier(n_neighbors=5)


    def cluster(self):
        """
        Applies clustering on X without knowing number of
        classes. Then,
        trains and returns a multiclass classifier.

        Args:
            No input args

        Returns:
            classifier object

        """

        if len(self._people_features_list) > 1:
            # Create the training set
            X = np.vstack(self._people_features_list)

            # Clustering
            #db = DBSCAN(eps=0.7).fit(X)
            db = DBSCAN(self.dbscan_eps,self.dbscan_min_samples, \
                algorithm='ball_tree').fit(X)

            # Labels of X
            y = db.labels_

            # Update the people list
            self.update_people_list(map(int, y))

            # Get a NN query
            self.classifier.fit(X, y)

        else:
            print "Error: There is only one sample!"

    def update_people_list(self, label_array):
        """
        Updates the _people_list with given input argument

        Args:
            label_array: labels of _people_features_list in list
             of integers

        Returns:
            No output args
        """
        self._people_list = label_array

    def infer(self, face_feature):
        """
        Classifies and returns the label of the sample

        Args:
            label_array(list of str): labels of
             _people_features_list

        Returns:
            label
        """

        try:
            return self.classifier.\
                predict_proba(face_feature.reshape(1,-1)).tolist()
        except Exception as e:
            print e


    def save_people_database(self):
        """
        Saves the current human recognition information as
         human_database.p

        Args:
            No input args

        Returns:
            No output args
        """
        pickle.dump([self._people_list, self._people_features_list],  \
            open("human_database.p", "wb"))

    def load_people_database(self, filename="human_database.p"):
        """
        Loads the trained human information pickle file

        Args:
            filename(str): name of the trained pickle,
             default=human_database.p

        Returns:
            No output args
        """
        self._people_list, self._people_features_list = \
            pickle.load(open(filename, "rb"))

    def add_features_2_dataset(self, features, label):
        """
        Loads the trained human information pickle file

        Args:
            features: (128D) feature vector
            label: (string) label of the feature

        Returns:
            No output args
        """
        if self._people_list.count(label) < \
            self.number_of_feature_threshold or \
            label is -1:

            self._people_features_list.append(features)
            self._people_list.append(label)

    def add_face_detection_2_messsage(self, msg_detection, bounding_box=None):

        r = Rect()

        if bounding_box:
            r.x = \
                msg_detection.head_detection.x + bounding_box.left()
            r.y = \
                msg_detection.head_detection.y + bounding_box.top()
            r.width = bounding_box.width()
            r.height = bounding_box.height()

            msg_detection.face_detections[0] = r
        else:
            r.x = msg_detection.head_detection.x
            r.y = msg_detection.head_detection.y
            r.width = msg_detection.head_detection.width
            r.height = msg_detection.head_detection.height

            msg_detection.face_detections.append(r)

        return msg_detection

    def threshold_based_recognition(self, data, converter, is_learning_active):

        # Create a ColorDepthImageArray which is a detection container
        color_depth_image_array = ColorDepthImageArray()

        # header is the same
        color_depth_image_array.header = data.header

        # List of depth images
        depth_list = []

        # Create the outgoing message: DetectionArray
        detection_list_msg = DetectionArray()
        detection_list_msg.header = data.header

        # Get all detections
        for idx, detection in enumerate(data.head_detections):

            # Convert rgb and depth images into OpenCV mat format
            im = converter(detection.color_image, type="bgr8")
            detection.depth_image.encoding = "32FC3"
            depth_image = converter(detection.depth_image)

            face = None
            bounding_box = None

            # check if a face is found
            if len(detection.face_detections) > 0:

                # create a bounding box object
                bounding_box = dlib.rectangle(detection.face_detections[0].x,\
                    detection.face_detections[0].y,\
                    detection.face_detections[0].width +
                    detection.face_detections[0].x, \
                    detection.face_detections[0]. height +
                    detection.face_detections[0].y)

                # align the face
                face = self.openface_wrapper.align_face(im, bounding_box)

            # get the biggest face in the head area and the bounding_box
            landmarks = self.openface_wrapper.find_landmarks(im, bounding_box)

            # if a face is detected
            if face is not None:

                # Estimate rotation vector
                rotation_vector = \
                    self.openface_wrapper.estimate_head_rotation(landmarks,\
                    im)

                # Create an empty Detection
                detection_msg = \
                    util.create_recognition_message("face")

                # extract features
                features = self.openface_wrapper.extract_features(face)

                print self.infer(features)

                print self._people_list

                # Check that if the program is its first turn
                if not self._people_features_list and \
                    self._people_list:

                    label = "Unknown"

                    if is_learning_active:
                        label = 0
                        # add the first people to the list
                        self._people_list[0] = label

                        self._people_features_list.append(np.array(features))

                    # Add message of current people to the detections list
                    color_depth_image_array.head_detections.append(\
                        detection \
                        )
                    # Add depth image to list
                    depth_list.append(depth_image)

                    # add face bounding box to incoming message
                    detection = self.add_face_detection_2_messsage(detection, \
                        bounding_box)

                    # update the recognition message
                    detection_msg = \
                        util.update_recognition_message(detection_msg,\
                        detection, depth_image, str(label))

                    # add detection message object to list
                    detection_list_msg.detections.append(detection_msg)

                else:

                    # flag for new people
                    is_people_found = False

                    # flag for low confidence people
                    low_confidence_flag = False

                    # list for measurements
                    measurement_list = []

                    # Storage for the index of new people
                    #new_people_index = 0

                    # Storage for the new people features
                    new_people_features = []

                    # iterate through the people list
                    # and compare the feature vectors
                    for p, f in itertools.izip(self._people_list, \
                        self._people_features_list):

                        # Calculate the distance between current
                        # face and database
                        distance =\
                            self.openface_wrapper.calculate_distance(\
                            features, f)

                        #print "Distance  {}".format(distance)

                        # add distance to list
                        measurement_list.append(distance)

                        # thresholding for a new people
                        if distance > self.hard_threshold and distance < \
                            self.soft_threshold:
                            low_confidence_flag = True
                        elif distance > self.soft_threshold:
                            new_people_features = features
                        else:
                            # this means a people met before is recognized
                            is_people_found = True

                    # Get the absolute value of rotation vector
                    rotation_vector = abs(rotation_vector)

                    print rotation_vector

                    if is_people_found:
                        # Get the closest measurement
                        idx = measurement_list.index(min(measurement_list))

                        print "You are {} with distance of {}" \
                            .format(self._people_list[idx], \
                            min(measurement_list))

                        # add face bounding box to incoming message
                        detection = self.add_face_detection_2_messsage(\
                            detection, \
                            bounding_box)

                        # update the recognition message
                        detection_msg =\
                            util.update_recognition_message(\
                            detection_msg, \
                            detection, depth_image,\
                            str(self._people_list[idx]))

                        # Populate the dataset of existing people if the list is
                        # not saturated or it is not unknown: -1

                        if is_learning_active:
                            self.add_features_2_dataset(features, \
                                self._people_list[idx])


                    elif low_confidence_flag:
                        # face is not recognized with high confidence

                        # Get the closer measurement
                        idx = measurement_list.index(min(measurement_list))

                        print "I am not sure!,\
                            You may be {} with a distance of {}" \
                            .format(self._people_list[idx],\
                                 min(measurement_list))

                        detection = self.add_face_detection_2_messsage(\
                            detection, \
                            bounding_box)

                        detection_msg =\
                        util.update_recognition_message(\
                            detection_msg, \
                            detection, depth_image, \
                            "Unknown")

                    elif rotation_vector[2] < self.max_abs_yaw \
                     and \
                        rotation_vector[0] > self.min_abs_pitch and \
                        rotation_vector[0] < self.max_abs_pitch and \
                        rotation_vector[1] < self.max_abs_roll and \
                        bounding_box.width() > 65:

                        # initialize the new label with unknown label
                        new_label = "Unknown"

                        if is_learning_active:

                            new_label = max(self._people_list)+1

                            # Add new people to the list
                            self._people_list.append(new_label)

                            self._people_features_list.append(\
                                np.array(new_people_features))

                            print "People list"
                            print self._people_list

                        detection = self.add_face_detection_2_messsage(\
                            detection, \
                            bounding_box)

                        detection_msg = util.update_recognition_message(\
                            detection_msg, \
                            detection, depth_image, \
                            str(new_label))

                    else:
                        print "I cannot recognize you!. Please look directly to\
                        my camera!"

                        detection =  \
                            self.add_face_detection_2_messsage(\
                            detection, \
                            bounding_box)

                        detection_msg =\
                            util.update_recognition_message(detection_msg, \
                            detection, depth_image,\
                            "Unknown")

                    # Add message to the list
                    detection_list_msg.detections.append(detection_msg)

                    # Add message of current people to the detections list
                    color_depth_image_array.head_detections.append(\
                        detection \
                        )
                    # add depth image to the list
                    depth_list.append(depth_image)

            else:
                # if there is no face
                # send a message with just head bounding box

                detection = self.add_face_detection_2_messsage(detection)

                detection_msg = \
                    util.create_recognition_message("head")

                detection_msg = util.update_recognition_message(detection_msg, \
                    detection, depth_image)

                detection_list_msg.detections.append(detection_msg)

        return detection_list_msg

    def look(self, data, converter, is_learning_active):
        """
        Main logic of online people recognition.
        This method shoulde be called at every iteration.
         It also publishes the message.

        Args:
            data(ColorDepthImageArray.msg): ROS msg coming
             from head detector
            converter(util.convert_img_to_cv): Converter
             from img to cv
            is_learning_active(bool): Flag for enabling learning

        Returns:
            No output args
        """
        # check if there are any heads in the scene
        number_of_heads_detected = len(data.head_detections)
        detection_list_msg = None
        if number_of_heads_detected > 0:

            # Train the main classifier if there are more than 1 people
            if len(self._people_list) > 1 and is_learning_active is True:
                self.cluster()

            # run the learning loop
            detection_list_msg = \
                self.threshold_based_recognition(data, \
                converter, is_learning_active)
        else:
            detection_list_msg = DetectionArray()
            detection_list_msg.header = data.header

        self.pub.publish(detection_list_msg)


class SupervisedPeopleRecognition(object):
    """docstring for SupervisedPeopleRecognition."""
    def __init__(self):
        super(SupervisedPeopleRecognition, self).__init__()
        self.clf = svm.SVC()
        self.clf.probability = True

    def train(self, X, y):
        self.clf.fit(X, y)

    def infer(self, X):
        return self.clf.predict(X.reshape(1, -1))[0]

    def look(self, converter, data, openface_wrapper):
        """
        Main logic of people recognition.
        This method shoulde be called at every iteration.

        Args:
            data(ColorDepthImageArray.msg): ROS msg coming
             from head detector
            converter(util.convert_img_to_cv): Converter
             from img to cv
            openface_wrapper(OpenFaceWrapper): class instance

        Returns:
            detection_list_msg(DetectionArray)
        """
        # Create the outgoing message: DetectionArray
        detection_list_msg = DetectionArray()
        detection_list_msg.header = data.header

        # Get all detections
        for idx, detection in enumerate(data.head_detections):

            # Convert rgb and depth images into OpenCV mat format
            im = converter(detection.color_image, type="bgr8")
            detection.depth_image.encoding = "32FC3"
            depth_image = converter(detection.depth_image)

            face = None
            bounding_box = None

            # check if a face is found
            if len(detection.face_detections) > 0:

                # create a bounding box object
                bounding_box = dlib.rectangle(detection.face_detections[0].x,\
                    detection.face_detections[0].y,\
                    detection.face_detections[0].width +
                    detection.face_detections[0].x, \
                    detection.face_detections[0]. height +
                    detection.face_detections[0].y)

                # align the face
                face = openface_wrapper.align_face(im, bounding_box)

            # get the biggest face in the head area and the bounding_box
            #landmarks = openface_wrapper.find_landmarks(im, bounding_box)

            # if a face is detected
            if face is not None:
                # Create an empty Detection
                detection_msg = \
                    util.create_recognition_message("face")

                # extract features
                features = openface_wrapper.extract_features(face)

                # get the prediction
                prediction = self.infer(features)

                # add face bounding box to incoming message
                detection = self.add_face_detection_2_messsage(\
                    detection, \
                    bounding_box)

                # update the recognition message
                detection_msg =\
                    util.update_recognition_message(\
                    detection_msg, \
                    detection, depth_image,\
                    prediction)

            else:
                # if there is no face
                # send a message with just head bounding box

                detection = self.add_face_detection_2_messsage(detection)

                detection_msg = \
                    util.create_recognition_message("head")

                detection_msg = util.update_recognition_message(detection_msg, \
                    detection, depth_image)

            # append all detections in one message
            detection_list_msg.detections.append(detection_msg)

        return detection_list_msg

    def add_face_detection_2_messsage(self, msg_detection, bounding_box=None):

        r = Rect()

        if bounding_box:
            r.x = \
                msg_detection.head_detection.x + bounding_box.left()
            r.y = \
                msg_detection.head_detection.y + bounding_box.top()
            r.width = bounding_box.width()
            r.height = bounding_box.height()

            msg_detection.face_detections[0] = r
        else:
            r.x = msg_detection.head_detection.x
            r.y = msg_detection.head_detection.y
            r.width = msg_detection.head_detection.width
            r.height = msg_detection.head_detection.height

            msg_detection.face_detections.append(r)

        return msg_detection

################################################################################
# Model Saver
#######################Model Saver##############################################

class ModelSaver(object):
    """docstring for ModelSaver."""
    def __init__(self, model_directory, filename="human_database.p"):
        super(ModelSaver, self).__init__()
        self.model_directory = model_directory
        self.filename = filename

    def save_people_database(self, online_people_recognition):
        """
        Saves the current human recognition information to the disk

        Args:
            online_people_recognition(PeopleRecognition object)

        Returns:
            No output args
        """
        pickle.dump([online_people_recognition._people_list,
            online_people_recognition._people_features_list],  \
            open(self.model_directory + self.filename, "wb"))

    def load_people_database(self,\
        online_people_recognition):
        """
        Loads the trained human information pickle file

        Args:
            filename(str): name of the trained pickle,
                 default=human_database.p

        Returns:
            online_people_recognition(PeopleRecognition object)

        """
        try:
            online_people_recognition._people_list,\
                online_people_recognition._people_features_list = \
                pickle.load(open(self.model_directory + self.filename, "rb"))
        except Exception as exception:
            print "Could not find the model, so created a new one! Error: {}".\
                format(exception)

        return online_people_recognition
