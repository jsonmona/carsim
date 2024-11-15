#! /usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np

from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist

from drive import Drive


def warping(image):
    source = np.float32([[140, 100], [0, 480], [500, 100], [640, 480]])
    destination = np.float32([[0, 0], [0, 480], [480, 0], [480, 480]])
    
    M = cv2.getPerspectiveTransform(source, destination)
    
    warp_image = cv2.warpPerspective(image, M, (480, 480), flags=cv2.INTER_LINEAR)

    return warp_image


class RobotControlNode():
    def __init__(self):
        self.drive = Drive()
        self.drive.do_movement = self.update_movement
        self.bridge = CvBridge()
        rospy.init_node('robot_control_node', anonymous=False)
        #rospy.Subscriber('/main_camera/image_raw/compressed', CompressedImage, self.camera_callback)
        rospy.Subscriber('/main_camera/image_raw', CompressedImage, self.camera_callback)
        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

    
    def camera_callback(self, data):
        image = self.bridge.compressed_imgmsg_to_cv2(data, desired_encoding="bgr8")
        image = warping(image)
        self.drive.on_camera(image)
        cv2.waitKey(1)
        

    def update_movement(self, speed, rot):
        movement = Twist()
        movement.linear.x = speed
        movement.angular.z = rot
        self.pub.publish(movement)
        


if __name__ == "__main__":
    if not rospy.is_shutdown():
        RobotControlNode()
        rospy.spin()
