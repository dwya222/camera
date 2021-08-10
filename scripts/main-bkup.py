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
TAG_SIZE = 0.143

DEBUG = True

if DEBUG:
    print("WARNING: DEBUG mode currently enabled!\nPublisher does not work in DEBUG mode.")

def main():
    point_publisher = rospy.Publisher('/point_command', Point, queue_size=10)
    rospy.init_node('point_pub')
    point_goal = Point()
    # Setup OpenCV for debugging
    #cv2.namedWindow("Color", cv2.WINDOW_AUTOSIZE)
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

    pc = rs.pointcloud()
    center_xs = []
    center_ys = []
    final_verts = []

    # Setup AprilTags Detector
    at_detector = Detector(families='tag36h11',
                           nthreads=1,
                           quad_decimate=1.0,
                           quad_sigma=0.0,
                           refine_edges=1,
                           decode_sharpening=0.25,
                           debug=0)
    tag_hmat = None

    # Let camera warm up for some frames
    for _ in range(30):
        pipeline.wait_for_frames()

    while True:
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
        binary = binarize_img(color_img)

        camera_params = (FOCAL_X, FOCAL_Y, PRINCIPAL_X, PRINCIPAL_Y)
        tags = at_detector.detect(gray, True, camera_params, TAG_SIZE)

        for tag in tags:
            if DEBUG:
                print("Found tag!")
            tag_hmat = np.hstack((tag.pose_R, tag.pose_t))
            tag_hmat = np.vstack((tag_hmat, [0,0,0,1]))
            for idx in range(len(tag.corners)):
                cv2.line(color_img, tuple(tag.corners[idx-1, :].astype(int)),
                tuple(tag.corners[idx, :].astype(int)), (0, 255, 0))

        ctr_img, rect = find_largest_blob(binary)

        if rect:
            center_x = int(rect[0] + (rect[2] / 2))
            center_y = int(rect[1] + (rect[3] / 2))
            if DEBUG:
                cv2.circle(color_img, (center_x, center_y), 10, (255, 0, 0), 2)
            center_xs.append(int(D_CAMERA_WIDTH / RGB_CAMERA_WIDTH * center_x))
            center_ys.append(int(D_CAMERA_HEIGHT / RGB_CAMERA_HEIGHT * center_y))

        if len(center_xs) == 0 or len(center_ys) == 0:
            avg_x = 0
            avg_y = 0
        else:
            avg_x = np.average(center_xs)
            avg_y = np.average(center_ys)

        # Delete to keep buffers from getting too large
        if len(center_xs) > 50:
            del center_xs[0]
        if len(center_ys) > 50:
            del center_ys[0]
        if len(final_verts) > 50:
            del final_verts[0]

        #print(f"Target Value: ({avg_x}, {avg_y})")

        points = pc.calculate(depth)
        v = points.get_vertices()
        verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
        center_pixel = int(avg_y) * D_CAMERA_WIDTH + int(avg_x)
        vert = np.append(verts[center_pixel], 1)

        final_vert = L.inv(tag_hmat) @ vert # np.dot(...)
        final_verts.append(final_vert)

        final_vert = np.average(final_verts, axis=0)

        if DEBUG:
            cv2.imshow("Color", color_img)
            cv2.imshow("Depth", ctr_img)
            k = cv2.waitKey(1)

            if k == 27:
                cv2.destroyAllWindows()
                break

        point_goal.x = .14 + final_vert[2]
        point_goal.y = final_vert[0]
        point_goal.z = .1 + final_vert[1]

        if DEBUG:
            print(f"Vertex: {final_vert}")
            print(f"Point Goal: {point_goal}")
        else:
            point_publisher.publish(point_goal)


def binarize_img(color_img, mask_size=7):
    low_H, low_S, low_V = (0, 120, 50)
    high_H, high_S, high_V = (5, 255, 255)
    erosion_kernel = np.ones((mask_size, mask_size), np.uint8)
    dilation_kernel = np.ones((mask_size*3, mask_size*3), np.uint8)

    hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
    frame_threshold = cv2.inRange(hsv, (low_H, low_S, low_V), (high_H, high_S, high_V))
    eroded = cv2.erode(frame_threshold, erosion_kernel)
    dilation = cv2.dilate(eroded, dilation_kernel)

    return dilation


def find_largest_blob(bin_img):
    largest_area = 0
    largest_contour_idx = 0
    bounding_rect = None

    contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area > largest_area:
            largest_area = area
            largest_contour_index = i
            bounding_rect = cv2.boundingRect(cnt)

    color = (randint(0,255),randint(0,255),randint(0,255))
    ctr_img = cv2.drawContours(bin_img, contours, largest_contour_idx, color, 3)

    return (ctr_img, bounding_rect)


if __name__ == '__main__':
    main()
