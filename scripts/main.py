#!/usr/bin/env python3

import cv2
import pyrealsense2 as rs
from dt_apriltags import Detector
import numpy as np
import numpy.linalg as L
from random import randint
import rospy
from geometry_msgs.msg import Point

RGB_CAMERA_WIDTH = 1280
RGB_CAMERA_HEIGHT = 720
D_CAMERA_WIDTH = 640
D_CAMERA_HEIGHT = 480
BASE_TAG_SIZE = 0.143
TARGET_TAG_SIZE = 0.026

DEBUG = True
#DEBUG = False

if DEBUG:
    print("WARNING: DEBUG mode currently enabled!\nPublisher does not work in DEBUG mode.")

def main():
    rospy.init_node('point_pub')
    point_publisher = rospy.Publisher('/point_command', Point, queue_size=1)
    point_goal = Point()
    # Setup OpenCV for debugging
    cv2.namedWindow("Color", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Depth", cv2.WINDOW_AUTOSIZE)

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

    # Setup AprilTags Detector
    at_detector = Detector(families='tag36h11',
                           nthreads=1,
                           quad_decimate=1.0,
                           quad_sigma=0.0,
                           refine_edges=1,
                           decode_sharpening=0.25,
                           debug=0)

    target_detector = Detector(families='tagStandard41h12',
                                nthreads=1,
                                quad_decimate=1.0,
                                quad_sigma=0.0,
                                refine_edges=1,
                                decode_sharpening=0.25,
                                debug=0)

    base_hmat = None
    target_hmat = None
    target_vec = None

    # Let camera warm up for some frames
    for _ in range(30):
        pipeline.wait_for_frames()

    while not rospy.is_shutdown():
        frames = pipeline.wait_for_frames()
        color = frames.get_color_frame()
        depth = frames.get_depth_frame()
        if not color or not depth:
            continue

        # Convert images to numpy arrays
        depth_img = np.asanyarray(depth.get_data())
        color_img = np.asanyarray(color.get_data())

        # AprilTags only allows for grayscale, so convert here
        gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

        camera_params = (FOCAL_X, FOCAL_Y, PRINCIPAL_X, PRINCIPAL_Y)
        base_tags = at_detector.detect(gray, True, camera_params, BASE_TAG_SIZE)

        for tag in base_tags:
            if DEBUG:
                pass#print("Found base tag!")
            base_hmat = hmat(tag.pose_R, tag.pose_t)
            for idx in range(len(tag.corners)):
                cv2.line(color_img, tuple(tag.corners[idx-1, :].astype(int)),
                tuple(tag.corners[idx, :].astype(int)), (0, 255, 0))

        target_tags = target_detector.detect(gray, True, camera_params, TARGET_TAG_SIZE)
        for tag in target_tags:
            if DEBUG:
                pass#print("Found target tag!")
            target_hmat = hmat(tag.pose_R, tag.pose_t)
            target_vec = np.append(tag.pose_t, 1)
            for idx in range(len(tag.corners)):
                cv2.line(color_img, tuple(tag.corners[idx-1, :].astype(int)),
                tuple(tag.corners[idx, :].astype(int)), (0, 255, 0))

        if DEBUG:
            cv2.imshow("Color", color_img)
            #cv2.imshow("Depth", ctr_img)
            k = cv2.waitKey(1)

            if k == 27:
                cv2.destroyAllWindows()
                break

        try:
            final_vert = L.inv(base_hmat) @ target_vec # np.dot(...)
        except Exception as e:
            rospy.logerr(f"Final vert calculation failed. Exception traceback: {e}")
            continue

        if DEBUG:
            print(f"Base Transform:\n{base_hmat}")
            print(f"Target Transform:\n{target_hmat}")
            print(f"Final Vert: {final_vert}")

        # final position offset for location of april tag and depth of box
        point_goal.x = .14 - final_vert[2] - .05
        point_goal.y = final_vert[0] - .05
        point_goal.z = .125 - final_vert[1]
        # David's tuned model 9/7/22 3:30 pm
        # point_goal.x = .137 - final_vert[2] - 0.05
        # point_goal.y = final_vert[0] - 0.05
        # point_goal.z = .115 - final_vert[1]  - 0.025
        # x = -final_vert[2]
        # y = final_vert[0]
        # z = -final_vert[1]
        # Model 1
        # point_goal.x = -1.50988*x + 0.521876*y - 0.88391*z + 0.8105088622167836
        # point_goal.y = 0.116555*x + 0.967043*y - 0.0122555*z - 0.019142235446133513
        # point_goal.z = 0.835344*x - 0.60735 *y - 0.195295*z - 0.862472468184561
        # Model 2
        # point_goal.x = -0.988231*x + 0.118553*y - 0.169785*z + 0.20453288331398603
        # point_goal.y = 0.0793662*x + 0.981345*y - 0.202772*z + 0.05431255401200416
        # point_goal.z = 0.120759*x  - 0.0506639*y  - 0.533495*z - 0.095935334741368
        # Model 3
        # point_goal.x = 0.961062*x + 0.10059*y - 0.0185353*z + 0.1368620622
        # point_goal.y = -0.107118*x + 0.962997*y + 0.0104095*z -0.01481089
        # point_goal.z = -0.0581172*x  - 0.009249*y  + 0.96769*z + 0.060088+0.08

        if DEBUG:
            print(f"Vertex: {final_vert}")
            print(f"Point Goal: {point_goal}")
        # else:
            point_publisher.publish(point_goal)


def hmat(R, t):
    out = np.hstack((R, t))
    out = np.vstack((out, [0,0,0,1]))
    return out

if __name__ == '__main__':
    main()
