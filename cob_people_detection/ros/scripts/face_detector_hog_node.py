#!/usr/bin/env python
"""

A ROS node to detect faces. First, it requires head point clouds, then
the node can be used to find the faces inside the head areas using Openface and
dlib APIs.

Author:
    Cagatay Odabasi -- cagatay.odabasi@ipa.fraunhofer.de

"""

import os

# ROS
import rospy

# numpy
import numpy as np

from cob_perception_msgs.msg import Detection, DetectionArray, Rect#, Mask
from cob_perception_msgs.msg import ColorDepthImageArray

import util
from util import Timer

from OpenfaceWrapper import OpenFaceWrapper

class FaceDetectorNode(object):
    """Main class for FaceDetectorNode"""
    def __init__(self):
        super(FaceDetectorNode, self).__init__()

        # get the parameters from parameter server
        (reason_about_3dface_size, face_size_max, face_size_min,\
            max_face_depth, display_timing, openface_directory) =\
            self.get_parameters()

        self.reason_about_3dface_size = reason_about_3dface_size
        self.face_size_max = face_size_max
        self.face_size_min = face_size_min
        self.max_face_depth = max_face_depth
        self.display_timing = display_timing
        self.openface_directory = openface_directory

        # Subscribe to head positions
        rospy.Subscriber("~head_positions", \
            ColorDepthImageArray, self.head_callback)

        # Advertise the result of Face Recognizer
        self.pub_cartesian = \
            rospy.Publisher('face_detector/face_detections_cartesian', \
            DetectionArray, queue_size=1)
        self.pub_position = \
            rospy.Publisher('face_detector/face_positions', \
            ColorDepthImageArray, queue_size=1)

        self.openface_wrapper = \
            OpenFaceWrapper(self.openface_directory)

    def face_3d_reasoning(self, face_depth_image, bounding_box, detection, is_active):
        """
        This method finds the biggest face in each head image coming from head
        detector node

        Args:
            face_depth_image(np.ndarray): depth cloud
            bounding_box(dlib.rectangle): bounding box of the detected face
            detection(ColorDepthImage): head detection message object
            is_active(bool): flag controlling the activation of this function

        Returns:
            is_valid_face (bool)
        """

        is_valid_face = False

        if is_active is True:
            # Get the face bounding box coords
            left = bounding_box.left()
            right = bounding_box.right()
            top = bounding_box.top()
            bottom = bounding_box.bottom()

            # Calculate face width
            face_width = abs(np.nanmax(face_depth_image[top : bottom, \
                left : right, 0]) -\
                np.nanmin(face_depth_image[top : bottom, left : right, 0]))

            # Calculate median depth
            median_depth = np.median(face_depth_image[top : bottom, \
                left : right, 2])

            # Check the face
            if face_width > self.face_size_min \
                and \
                face_width < self.face_size_max \
                and \
                median_depth < self.max_face_depth:
                is_valid_face = True
            else:
                is_valid_face = False
        else:
            is_valid_face = True

        # Check if face image goes outside the head image
        if bounding_box.left() < 0 or bounding_box.top() < 0 or\
            bounding_box.left() + bounding_box.width() > \
            detection.head_detection.width or\
            bounding_box.top() + bounding_box.height() > \
            detection.head_detection.height:
            is_valid_face = False

        return is_valid_face

    def find_faces_in_head_images(self, data, converter):
        """
        This method finds the biggest face in each head image coming from head
        detector node

        Args:
            data(ColorDepthImageArray.msg): ROS msg coming from head detector
            converter(util.convert_img_to_cv): Converter from img to cv

        Returns:
            (ColorDepthImageArray, DetectionArray) tuple
        """

        # Create a ColorDepthImageArray which is a detection container
        color_depth_image_array = ColorDepthImageArray()

        # header is the same
        color_depth_image_array.header = data.header

        # Create the outgoing message: DetectionArray
        detection_list_msg = DetectionArray()
        detection_list_msg.header = data.header

        # Get all detections
        for idx, detection in enumerate(data.head_detections):

            # Convert rgb and depth images into OpenCV mat format
            image = converter(detection.color_image, type="bgr8")
            detection.depth_image.encoding = "32FC3"
            depth_image = converter(detection.depth_image)

            # get the biggest face in the head area and the bounding_box
            face, bounding_box, _ = self.openface_wrapper.detect_face(image)

            # if a face is detected
            if face is not None:

                # Check if it is a valid face in 3D coords
                valid_face = self.face_3d_reasoning(depth_image,\
                    bounding_box, detection, self.reason_about_3dface_size)

                if valid_face is True:
                    # Create a rectangle
                    rectangle = Rect()

                    # Fill the rectangle with face positions
                    rectangle.x = bounding_box.left()
                    rectangle.y = bounding_box.top()
                    rectangle.width = bounding_box.width()
                    rectangle.height = bounding_box.height()

                    # Add face detection to the message
                    data.head_detections[idx].face_detections.append(rectangle)

                    # prepare message for cartesian coordinates
                    d_msg = Detection()
                    d_msg = \
                        util.prepare_cartesian_message_4_face_detector(d_msg, \
                        detection, depth_image)

                    # Append the detection
                    detection_list_msg.detections.append(d_msg)

        return (data, detection_list_msg)

    def get_parameters(self):
        """
        Gets the necessary parameters from parameter server

        Args:

        Returns:
            tuple of parameters(bool, double, double, double, bool)

        """
        reason_about_3dface_size = False # TODO: There is a problem with this
        # function that needs to be solved.
        #rospy.get_param("~reason_about_3dface_size")
        face_size_max = rospy.get_param("~face_size_max_m")
        face_size_min = rospy.get_param("~face_size_min_m")
        max_face_depth = rospy.get_param("~max_face_z_m")
        display_timing = rospy.get_param("~display_timing")
        openface_directory = os.path.expanduser("~/openface/")

        return (reason_about_3dface_size, face_size_max, face_size_min,\
            max_face_depth, display_timing, openface_directory)

    def head_callback(self, data):
        """Callback for head images """

        msg_pos = None
        msg_cart = None

        if self.display_timing is True:
            with Timer("Face Detection"):
                msg_pos, msg_cart =\
                    self.find_faces_in_head_images(data,\
                        util.convert_img_to_cv)
        else:
            msg_pos, msg_cart =\
                self.find_faces_in_head_images(data, \
                    util.convert_img_to_cv)

        # publish the messages
        self.pub_position.publish(msg_pos)
        self.pub_cartesian.publish(msg_cart)

def main():
    """Main function """
    # init the node
    rospy.init_node('face_detector_hog', anonymous=False)

    # start the node
    FaceDetectorNode()

    # spin until someone presses ctrl + c
    rospy.spin()

if __name__ == '__main__':
    main()
