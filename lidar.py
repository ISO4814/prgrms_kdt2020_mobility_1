#!/usr/bin/env python
#-*- coding: utf-8 -*-

import rospy
import cv2, math, time
import numpy as np

from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge
from xycar_motor.msg import xycar_motor

frame = np.empty(shape=[0])
cap = cv2.VideoCapture(0)
obstacle = False
distance = np.empty(shape=[0])
cur_lane = "mid"

# =============== Hough Transform =================
def calculate_lines(img, lines):
    global left_line, right_line, both_line

    left = []
    right = []

    try:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            # print("x1: {}, y1: {}, x2: {}, y2: {}".format(x1, y1, x2, y2))
            if x1 != x2:
                parameters = np.polyfit((x1, x2), (y1, y2), 1)
                # print("slope: {}".format(parameters[0]))
            else:
                continue
            slope = parameters[0]
            y_intercept = parameters[1]
            if x2 < 320:
                left.append((slope, y_intercept))
            elif 320 < x2:
                right.append((slope, y_intercept))

        if len(left) != 0:
            left_avg = np.average(left, axis=0)
            left_line = calculate_coordinates(img, left_avg)
        elif len(left) == 0:
            left_line = np.array([0, 0, 0, 0])

        if len(right) != 0:
            right_avg = np.average(right, axis=0)
            right_line = calculate_coordinates(img, right_avg)
        elif len(right) == 0:
            right_line = np.array([0, 0, 0, 0])

        if len(left) != 0 or len(right) != 0:
            both = left + right
            both_avg = np.average(both, axis=0)               # [slope_avg, y_intercept_avg]
            both_line = calculate_coordinates(img, both_avg)  # [x1, y1, x2, y2]
        elif len(left) == 0 and len(right) == 0:
            both_line = np.array([320, 480, 320, 280])

        return np.array([left_line, right_line]), np.array([both_line])
    
    except TypeError:
        pass

def calculate_coordinates(img, parameters):
    global height

    slope, intercept = parameters

    y1 = height
    y2 = int(y1 - 200)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    
    return np.array([x1, y1, x2, y2])

def visualize_direction(img, lines):
    lines_visualize = np.zeros_like(img)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            try:
                cv2.line(lines_visualize, (x1+(320-x1), y1), (x2+(320-x1), y2), (0, 0, 255), 5)
            except OverflowError:
                pass

    return lines_visualize

def visualize_lines(img, lines):
    lines_visualize = np.zeros_like(img)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            try:
                cv2.line(lines_visualize, (x1, y1), (x2, y2), (0, 255, 0), 5)
            except OverflowError:
                pass

    return lines_visualize

# =============== Image Processing =================
def perspective_img(img):
    global frame
    global height, width, mid_x
    
    point_1 = [110, height-200]
    point_2 = [10, height- 160]
    point_3 = [width-110, height-200]
    point_4 = [width-10, height-160]

    # draw area
    area = np.zeros_like(img)
    area = cv2.line(area, tuple(point_1), tuple(point_2), (255, 255, 0), 2)
    area = cv2.line(area, tuple(point_3), tuple(point_4), (255, 255, 0), 2)
    area = cv2.line(area, tuple(point_1), tuple(point_3), (255, 255, 0), 2)
    area = cv2.line(area, tuple(point_2), tuple(point_4), (255, 255, 0), 2)
    area = cv2.line(area, (320, 480), (320, 0), (255, 255, 0), 4)

    warp_src  = np.array([point_1, point_2, point_3, point_4], dtype=np.float32)
    
    warp_dist = np.array([[0,0],\
                          [0,height],\
                          [width,0],\
                          [width, height]],\
                         dtype=np.float32)

    M = cv2.getPerspectiveTransform(warp_src, warp_dist)
    Minv = cv2.getPerspectiveTransform(warp_dist, warp_src)
    warp_img = cv2.warpPerspective(img, M, (width, height), flags=cv2.INTER_LINEAR)
    

    return warp_img, M, Minv, area

def set_roi(img):
    global height, width, mid_x

    region_1 = np.array([[
        (10, height-20),
        (10, height-150),
        (200, height-240),
        (mid_x-20, height-240),
        (mid_x-200, height-20)
    ]])

    region_2 = np.array([[
        (width-10, height-20),
        (width-10, height-150),
        (width-200, height-240),
        (mid_x+20, height-240),
        (mid_x+200, height-20)
    ]])
    
    mask = np.zeros_like(img)
    left_roi = cv2.fillPoly(mask, region_1, 255)
    right_roi = cv2.fillPoly(mask, region_2, 255)
    roi = cv2.bitwise_and(img, mask)
    
    return roi

def canny_edge(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def lane_keeping(lines):
    global height, width
    global pub, motor_control
    global steer

    speed = 30
    max_steer = 30
    x1, y1, x2, y2 = lines[0]
    parameters = np.polyfit((x1, x2), (y1, y2), 1)
    #print("slope: {}".format(parameters[0]))
    angle = math.atan2(1, abs(parameters[0]))
    angle = math.degrees(angle)

    if 0 < angle <= 10:
        Kp = 3.0
    elif 10 < angle:
        Kp = 3.0

    if 0 < parameters[0]:
        steer = -angle * Kp
    elif parameters[0] < 0 :
        steer = (angle * Kp) + 7

    steer = np.clip(steer, -max_steer, max_steer)

    #print("steer: {}".format(steer))
    motor_control.angle = steer
    motor_control.speed = speed
    pub.publish(motor_control)

def img_callback(data):
    global frame, bridge

    frame = bridge.imgmsg_to_cv2(data, "bgr8")

def lidar_callback(data):
    global distance

    distance = data.ranges

def obstacle_cnt(distance):
    lidar_count_l = 0
    lidar_count_r = 0
    lidar = distance
    for r_point in lidar[185:250]:
        if r_point != 0.0 and r_point < 0.5:
            lidar_count_r += 1
    for l_point in lidar[250:315]:
        if l_point != 0.0 and l_point < 0.5:
            lidar_count_l += 1
    lidar_obs_count = lidar_count_l + lidar_count_r

    return [lidar_obs_count, lidar_count_l, lidar_count_r]

def stop_line(img):
    
    return True

def start_detect(img):

    return True


def main():
    global frame
    global height, width, mid_x
    global bridge, pub, motor_control
    global distance
    global cur_lane

    bridge = CvBridge()
    motor_control = xycar_motor()

    rospy.init_node('lane_detect')
    rospy.Subscriber("/usb_cam/image_raw", Image, img_callback, queue_size=1)
    rospy.Subscriber("/scan", LaserScan, lidar_callback, queue_size=1)
    pub = rospy.Publisher('/xycar_motor', xycar_motor, queue_size=1)

    while not rospy.is_shutdown():
        if frame.size != (640 * 480 * 3):
            continue
        height, width = frame.shape[0:2]
        mid_x = width // 2
        [lidar_obs_count, lidar_count_l, lidar_count_r] = obstacle_cnt(distance)
          
        
        if lidar_obs_count > 20:
            obstacle = True
            print("l, r", lidar_count_l, lidar_count_r)
            if cur_lane == "mid" and lidar_count_l < lidar_count_r:
                cur_lane = "left"
                t = time.time()
                #while time.time() - t < 1:
                while obstacle_cnt(distance)[0] != 0:
                    motor_control.angle = -30
                    motor_control.speed =30
                    pub.publish(motor_control)
                    lidar = distance
            elif cur_lane == "mid" and lidar_count_l > lidar_count_r:
                cur_lane = "right"
                t = time.time()
                while obstacle_cnt(distance)[0] != 0:
                #while time.time() - t < 1:
                    motor_control.angle = 37
                    motor_control.speed = 30
                    pub.publish(motor_control)
            elif cur_lane == "right":
                cur_lane = "left"
                t = time.time()
                while obstacle_cnt(distance)[0] != 0:
                #while time.time() - t < 1:
                    motor_control.angle = -30
                    motor_control.speed = 30
                    pub.publish(motor_control)
            elif cur_lane == "left":
                cur_lane = "right"
                t = time.time()
                while obstacle_cnt(distance)[0] != 0:
                #while time.time() - t < 1:
                    motor_control.angle = 37
                    motor_control.speed = 30
                    pub.publish(motor_control)
                    
        
        else:
            
            #============== image transform ==============
            canny = canny_edge(frame)

            #============== stope line =================
            if stop_line(canny):
                t = time.time()
                while time.time() - t < 5.1:
                    motor_control.angle = 0
                    motor_control.speed = 0
                    pub.publish(motor_control)
                lap_count = True
                continue
            
            if start_detect(canny) and lap_count:
                lap += 1
                lap_count = False

            # cv2.imshow('canny', canny)
            roi = set_roi(canny)
            #cv2.imshow('roi', roi)
            warp_img, M, Minv, area = perspective_img(roi)
            #============== Hough Line Transform ==============
            hough = cv2.HoughLinesP(warp_img, 1, np.pi/180, 100, np.array([]), minLineLength = 20, maxLineGap = 20)
            if hough is not None:
                lines, direction = calculate_lines(warp_img, hough)
            
            warp_img = cv2.cvtColor(warp_img, cv2.COLOR_GRAY2BGR)
            lines_visualize = visualize_lines(warp_img, lines)
            warp_img = cv2.addWeighted(warp_img, 0.9, lines_visualize, 1, 1)
            direction_visualize = visualize_direction(warp_img, direction)
            warp_img = cv2.addWeighted(warp_img, 0.9, direction_visualize, 1, 1)
            roi = cv2.addWeighted(roi, 0.9, area, 1, 1)
            
            #cv2.imshow('warp', warp_img)
            #cv2.imshow('result', roi)
            lane_keeping(direction)
            cv2.waitKey(1)

        print("cur_lane = " + cur_lane)

if __name__ == "__main__":
    try:
        main()
    finally:
        cap.release()
        cv2.destroyAllWindows()
