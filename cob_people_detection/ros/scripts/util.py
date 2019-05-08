#!/usr/bin/env python
"""
Different utility functions to support online face recognition node
under cob_people_detection package.

Author:
    Cagatay Odabasi -- cagatay.odabasi@ipa.fraunhofer.de
    Timer class is taken from: https://stackoverflow.com/a/5849861/3244341
"""

import time

# OpenCV
#import cv2

#numpy
#import numpy as np

# OpenCV Bridge
from cv_bridge import CvBridge, CvBridgeError

# Messages
from cob_perception_msgs.msg import Detection#, DetectionArray, Rect, Mask
#from cob_perception_msgs.msg import ColorDepthImageArray
#from std_msgs.msg import Header
#from sensor_msgs.msg import Image

################################################################################
# UTILS
################################################################################

# convert sensor_msgs/image to OpenCV image with bridge
# this function is for RGB images
def convert_img_to_cv(sensor_img, type=None):
    """
    convert sensor_msgs/image to OpenCV image with bridge, but just for
    RGB images

    Args:
        type(string): bgr8
        sensor_img(Image std_msgs)

    Returns:
        (np.array) image
    """
    bridge = CvBridge()

    # if type is set, rgb conversion is made, otherwise depth
    if type:
        try:
            return bridge.imgmsg_to_cv2(sensor_img, type)
        except CvBridgeError as e:
            print e
    else:
        try:
            return bridge.imgmsg_to_cv2(sensor_img)
        except CvBridgeError as e:
            print e

def prepare_cartesian_message_4_face_detector(msg, head, depth_image):
    temp_message = Detection()

    temp_message.pose.pose.orientation.x = 0
    temp_message.pose.pose.orientation.y = 0
    temp_message.pose.pose.orientation.z = 0
    temp_message.pose.pose.orientation.w = 1

    temp_message.detector = "face"

    msg.mask.roi.x = head.face_detections[0].x
    msg.mask.roi.y = head.face_detections[0].y
    msg.mask.roi.width = head.face_detections[0].width
    msg.mask.roi.height = head.face_detections[0].height

    # calculate center
    coord_x = depth_image.shape[0]/2
    coord_y = depth_image.shape[0]/2

    msg.pose.pose.position.x = depth_image[coord_x, coord_y][0]
    msg.pose.pose.position.y = depth_image[coord_x, coord_y][1]
    msg.pose.pose.position.z = depth_image[coord_x, coord_y][2]

    return temp_message

def create_recognition_message(detector):
    """
    Create empty recognition message and put the label of the detector:
    face or head

    Args:
        detector(string): face or head detector

    Returns:
        (Detection()) message
    """
    temp_message = Detection()

    temp_message.pose.pose.orientation.x = 0
    temp_message.pose.pose.orientation.y = 0
    temp_message.pose.pose.orientation.z = 0
    temp_message.pose.pose.orientation.w = 1

    temp_message.detector = detector

    return temp_message

def crop_image(image, bounding_box):
    """
    Crops the input image according to the given bounding box

    Args:
        image(np.array): input image to be cropped
        bounding_box(dlib.rect): bounding box

    Returns:
        (np.array) Cropped image
    """
    # Create a dlib rectangle
    return image[bounding_box.y:bounding_box.y + bounding_box.height,
        bounding_box.x:bounding_box.x + bounding_box.width, :]

def update_recognition_message(msg, head, depth_image, label=None):
    """
    Update the reconition message according to the arguments given,
    if there is a label, this means that a face is detected; however,
    if the label is None, then this will be a head message.

    Args:
        msg (Detection()): message to be updated
        head (ColorDepthImage()): message containing head detection
        depth_image(np.array): depth image of head
        label(string): label of the face found


    Returns:
        (Detection()) message
    """
    msg.mask.roi.x = head.face_detections[0].x
    msg.mask.roi.y = head.face_detections[0].y
    msg.mask.roi.width = head.face_detections[0].width
    msg.mask.roi.height = head.face_detections[0].height

    # calculate center
    coord_x = depth_image.shape[0]/2
    coord_y = depth_image.shape[0]/2

    msg.pose.pose.position.x = depth_image[coord_x, coord_y][0]
    msg.pose.pose.position.y = depth_image[coord_x, coord_y][1]
    msg.pose.pose.position.z = depth_image[coord_x, coord_y][2]

    if label is not None:
        msg.label = label

        msg.header = head.color_image.header

        msg.detector = "face"

    else:
        msg.header = head.color_image.header

        msg.label = "UnknownHead"

        msg.detector = "head"

    return msg

class Timer(object):
    """
    Timer class to measure elapsed time

    Example:
        with Timer():
            do_something()

        when do_something() finishes the time will be printed.

    """
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print '[%s]' % self.name,
        print 'Elapsed: %s' % (time.time() - self.tstart)
