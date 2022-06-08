#!/usr/bin/env python3

import cv2
import pyrealsense2 as rs
from dt_apriltags import Detector
import numpy as np
import numpy.linalg as L
from random import randint
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

RGB_CAMERA_WIDTH = 1280
RGB_CAMERA_HEIGHT = 720
D_CAMERA_WIDTH = 640
D_CAMERA_HEIGHT = 480
BASE_TAG_SIZE = 0.143
TARGET_TAG_SIZE = 0.026

DEBUG = False

def main():
    rospy.init_node('image_pub')
    br = CvBridge()
    image_pub = rospy.Publisher('/realsense_image', Image, queue_size=10)
    # Setup OpenCV for debugging
    if DEBUG:
        cv2.namedWindow("Color", cv2.WINDOW_AUTOSIZE)
    #cv2.namedWindow("Depth", cv2.WINDOW_AUTOSIZE)

    # Initialize RealSense
    cfg = rs.config()

    cfg.enable_stream(rs.stream.color, RGB_CAMERA_WIDTH, RGB_CAMERA_HEIGHT, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, D_CAMERA_WIDTH, D_CAMERA_HEIGHT, rs.format.z16, 30)

    pipeline = rs.pipeline()
    pipeline.start(cfg)

    profile = pipeline.get_active_profile()
    rgb_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
    rgb_intrinsics = rgb_profile.get_intrinsics()

    FOCAL_X = rgb_intrinsics.fx
    FOCAL_Y = rgb_intrinsics.fy
    PRINCIPAL_X = rgb_intrinsics.ppx
    PRINCIPAL_Y = rgb_intrinsics.ppy

    # Let camera warm up for some frames
    for _ in range(30):
        pipeline.wait_for_frames()

    while not rospy.is_shutdown():
        frames = pipeline.wait_for_frames()
        color = frames.get_color_frame()
        depth = frames.get_depth_frame()
        if not color or not depth:
            continue

        color_img = np.asanyarray(color.get_data())
        out_image = br.cv2_to_imgmsg(color_img)

        image_pub.publish(out_image)

if __name__ == '__main__':
    main()
