#!/usr/bin/env python

""" A ROS service to enable or disable the online face learning

This service sould be called in a launch file, since the default value for
learning can be false. So, the system cannot learn new people.

Example:
    To enable the learning:

        $ rosservice call /online_face_recognizer/
            toggle_online_learning "data: true"

    To disable the learning:

        $ rosservice call /online_face_recognizer/
            toggle_online_learning "data: false"

Author:
    Cagatay Odabasi -- cagatay.odabasi@ipa.fraunhofer.de

"""

from std_srvs.srv import SetBool, SetBoolResponse
import rospy

def toggle(req):
    """
    Applies clustering on X without knowing number of
    classes. Then,
    trains and returns a multiclass classifier.

    Args:
        Request object(SetBool)

    Returns:
        Response object(SetBool)

    """
    # Get the paramater
    param_name = rospy.search_param("enable_online_face_recognition")
    v = rospy.get_param(param_name)

    resp = SetBoolResponse()

    if v is True and req.data is False:
        rospy.set_param(param_name, False)
        resp.success = True
        resp.message = "Online learning is disabled!"
    elif v is False and req.data is True:
        rospy.set_param(param_name, True)
        resp.success = True
        resp.message = "Online learning is enabled!"
    elif v is True and req.data is True:
        resp.success = False
        resp.message = "Online learning is already enabled!"
    else:
        resp.success = False
        resp.message = "Online learning is already disabled!"

    return resp

def toggle_server():
    """
    Main function of the module.

    Args:

    Returns:

    """
    rospy.init_node('toggle_online_learning_server')
    rospy.Service('toggle_online_learning', SetBool, toggle)
    rospy.spin()

if __name__ == "__main__":
    toggle_server()
