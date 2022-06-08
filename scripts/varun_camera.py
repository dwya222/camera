#!/usr/bin/env python3

import cv2
from PIL import Image
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

DEBUG = False

def main():
    rospy.init_node('image_pub')
    br = CvBridge()
    video = cv2.VideoCapture('/dev/video0')
    image_pub = rospy.Publisher('/logi_image', Image, queue_size=10)

    for _ in range(30):
        _,frame = video.read()

    while not rospy.is_shutdown():
        _,frame = video.read()
        out_image = br.cv2_to_imgmsg(frame)
        image_pub.publish(out_image)

if __name__ == '__main__':
    main()
