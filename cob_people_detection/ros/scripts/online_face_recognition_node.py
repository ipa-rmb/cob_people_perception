#!/usr/bin/env python
"""
A ROS node to learn unknown faces online and save them in a database as a
pickle. The system first aligns the face images according to keypoints
detected with dlib, then FaceNet face feature extractor network is applied
to extract features. These features are 128D long vector whicha become
input to a DBSCAN clustering algortihm. In this way, an id is assigned to
the each entry in database.

Author:
    Cagatay Odabasi -- cagatay.odabasi@ipa.fraunhofer.de
"""

# ROS
import rospy

from std_srvs.srv import Empty, SetBool, SetBoolRequest

from cob_perception_msgs.msg import DetectionArray# Detection, Rect, Mask
from cob_perception_msgs.msg import ColorDepthImageArray

import util
from util import Timer

from OpenfaceWrapper import OpenFaceWrapper
from PeopleRecognition import PeopleRecognition
from PeopleRecognition import ModelSaver

import actionlib

from cob_people_detection.msg import updateDataAction, addDataAction, \
                                     getDetectionsAction, loadModelAction, \
                                     deleteDataAction

################################################################################
# Action Servers
################################################################################
class UpdateDataServer(object):
    def __init__(self, node):
        super(UpdateDataServer, self).__init__()
        self.server = actionlib.SimpleActionServer(\
            '/face_capture/update_data_server',\
            updateDataAction, self.execute, False)
        self.node = node
        self.server.start()

    def execute(self, goal):
        # unsubscribe from face positions
        self.node.sub_face.unregister()

        # Call the service to turn off recognition

        rospy.wait_for_service('/online_face_recognizer/toggle_online_learning')

        toggle = rospy.ServiceProxy(\
            '/online_face_recognizer/toggle_online_learning', SetBool)

        req = SetBoolRequest()

        req.data = False

        toggle(req)

        names = dict()

        names[goal.old_label] = goal.new_label

        # assign names to ids
        self.node.online_people_recognition.assign_names_to_people_list(names)

        # Subscribe to the face positions again
        self.node.sub_face = rospy.Subscriber("~face_detector/face_positions",\
            ColorDepthImageArray, self.node.face_callback)

        self.server.set_succeeded()
class DeleteDataServer(object):
    def __init__(self, node):
        super(DeleteDataServer, self).__init__()
        self.server = actionlib.SimpleActionServer(\
            '/face_capture/delete_data_server',\
            deleteDataAction, self.execute, False)
        self.node = node
        self.server.start()

    def execute(self, goal):
        # unsubscribe from face positions
        self.node.sub_face.unregister()

        # Call the service to turn off recognition

        rospy.wait_for_service('/online_face_recognizer/toggle_online_learning')

        toggle = rospy.ServiceProxy(\
            '/online_face_recognizer/toggle_online_learning', SetBool)

        req = SetBoolRequest()

        req.data = False

        toggle(req)

        self.node.online_people_recognition.\
            delete_entries_with_label(goal.label)

        # Subscribe to the face positions again
        self.node.sub_face = rospy.Subscriber("~face_detector/face_positions",\
            ColorDepthImageArray, self.node.face_callback)

        self.server.set_succeeded()
class AddDataServer(object):
    def __init__(self, node):
        super(AddDataServer, self).__init__()
        self.server = actionlib.SimpleActionServer(\
            '/face_capture/add_data_server',\
            addDataAction, self.execute, False)
        self.node = node
        self.server.start()

    def execute(self, goal):
        self.server.set_succeeded()
class LoadModelServer(object):
    def __init__(self, node):
        super(LoadModelServer, self).__init__()
        self.server = actionlib.SimpleActionServer(\
            '/face_recognizer/load_model_server',\
            loadModelAction, self.execute, False)
        self.node = node
        self.server.start()

    def execute(self, goal):
        self.server.set_succeeded()
class GetDetectionsServer(object):
    def __init__(self, node):
        super(GetDetectionsServer, self).__init__()
        self.server = actionlib.SimpleActionServer(\
            '/face_recognizer/get_detections_server',\
            getDetectionsAction, self.execute, False)
        self.node = node
        self.server.start()

    def execute(self, goal):
        self.server.set_succeeded()
################################################################################

################################################################################
# Main part of program
################################################################################

class OnlineFaceRecognizerNode(object):
    """docstring for OnlineFaceRecognizerNode."""
    def __init__(self):
        super(OnlineFaceRecognizerNode, self).__init__()
        self.enable_online_face_recognition = None
        self.max_abs_pitch = None
        self.min_abs_pitch = None
        self.max_abs_yaw = None
        self.max_abs_roll = None
        self.display_timing = None
        self.soft_threshold = None
        self.hard_threshold = None
        self.number_of_features_per_people = None
        self.dbscan_eps = None
        self.dbscan_min_samples = None
        self.openface_directory = None
        self.model_directory = None

        self.online_people_recognition = None
        self.openface_wrapper = None

        self.sub_image = None
        self.sub_face = None

        # init the node
        rospy.init_node('online_people_recognition', anonymous=False)

        self.get_parameters()

        # Model Saver object
        self.model_saver = ModelSaver(self.model_directory)

        # Advertise the result of Face Recognizer
        self.pub = rospy.Publisher('face_recognizer/face_recognitions', \
            DetectionArray, queue_size=1)

        # Create the service for assigning names to ids
        rospy.Service('assign_names_console', Empty, self.assign)

        # create openface_wrapper class
        openface_wrapper = OpenFaceWrapper(self.openface_directory)

        # create PeopleRecognition
        self.online_people_recognition = PeopleRecognition(openface_wrapper, \
            self.pub,\
            self.hard_threshold, \
            self.soft_threshold, self.number_of_features_per_people, \
            self.min_abs_pitch, self.max_abs_pitch, self.max_abs_yaw, \
            self.max_abs_roll, \
            self.dbscan_eps, self.dbscan_min_samples)


        # load the database
        self.online_people_recognition = \
            self.model_saver.load_people_database(\
            self.online_people_recognition)

        # Subscribe to the face positions
        self.sub_face = rospy.Subscriber("~face_detector/face_positions",\
            ColorDepthImageArray, self.face_callback)

        #Run the servers
        UpdateDataServer(self)
        DeleteDataServer(self)
        LoadModelServer(self)
        GetDetectionsServer(self)
        AddDataServer(self)

        # spin until the end of world or someone presses ctrl + c
        rospy.spin()

        # ctrl + c is pressed, so save the model
        self.model_saver.save_people_database(self.online_people_recognition)


    def get_parameters(self):
        """
        Gets the necessary parameters from parameter server

        Args:

        Returns:

        """

        self.enable_online_face_recognition =\
            rospy.get_param("~enable_online_face_recognition")
        self.max_abs_pitch = rospy.get_param("~max_abs_pitch")
        self.min_abs_pitch = rospy.get_param("~min_abs_pitch")
        self.max_abs_yaw = rospy.get_param("~max_abs_yaw")
        self.max_abs_roll = rospy.get_param("~max_abs_roll")
        self.display_timing = rospy.get_param("~display_timing")
        self.soft_threshold = rospy.get_param("~soft_threshold")
        self.hard_threshold = rospy.get_param("~hard_threshold")
        self.number_of_features_per_people = \
            rospy.get_param("~number_of_features_per_people")
        self.dbscan_eps = rospy.get_param("~eps")
        self.dbscan_min_samples = rospy.get_param("~min_samples")
        self.openface_directory = rospy.get_param("~openface_directory")
        self.model_directory = rospy.get_param("~model_directory")

        # if offline recognition is selected, shutdown this node
        self.shutdown(rospy.get_param("~recognition_method"))

    def shutdown(self, recognition_method):
        """
        Shuts down the node if recognition method is not 4
        """
        if not recognition_method is 4:
            rospy.signal_shutdown("Offline recognition is selected!")

    def face_callback(self, data):
        """ Callback for face images(RGBD)
        """
        # Get the parameter for enabling the training
        self.enable_online_face_recognition =\
            rospy.get_param("~enable_online_face_recognition")

        # main logic that recognites people in an online manner
        if self.display_timing is True:
            with Timer("Recognition "):
                self.online_people_recognition.look(data, \
                    util.convert_img_to_cv,\
                    self.enable_online_face_recognition)
        else:
            self.online_people_recognition.look(data, util.convert_img_to_cv,\
                self.enable_online_face_recognition)

    def assign(self, arg):
        """
        Service function for assigning names to the ids in the database.
        """

        # unsubscribe from face positions
        self.sub_face.unregister()

        # get unique labels
        unique_labels = set(self.online_people_recognition._people_list)

        # name storage for real people names
        names = dict()

        # get the inputs and asign them to id list
        for label in unique_labels:
            var = raw_input("Who is {}? ".format(label))

            names[label] = var

        # Call the service

        rospy.wait_for_service('/online_face_recognizer/toggle_online_learning')

        toggle = rospy.ServiceProxy(\
            '/online_face_recognizer/toggle_online_learning', SetBool)

        req = SetBoolRequest()

        req.data = False

        toggle(req)

        # assign names to ids
        self.online_people_recognition.assign_names_to_people_list(names)

        # Subscribe to the face positions again
        self.sub_face = rospy.Subscriber("~face_detector/face_positions",\
            ColorDepthImageArray, self.face_callback)


def main():
    """ main function
    """
    node = OnlineFaceRecognizerNode()

if __name__ == '__main__':
    main()
