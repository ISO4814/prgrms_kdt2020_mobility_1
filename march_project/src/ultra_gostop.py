#!/usr/bin/env python

import rospy
import time
from xycar_motor.msg import xycar_motor
from std_msgs.msg import Int32MultiArray

motor_control = xycar_motor()
distance = [0,0,0,0,0,0,0,0]
rospy.init_node('auto_driver')

pub = rospy.Publisher('xycar_motor', xycar_motor, queue_size=1)

def callback(data):
    global distance 
    distance = data.data

def drive_go():
    global pub
    global motor_control
    motor_control.angle = 0
    motor_control.speed = 30
    pub.publish(motor_control)

def drive_stop():
    global pub
    global motor_control
    motor_control.angle = 0
    motor_control.speed = 0
    pub.publish(motor_control)
    
sub = rospy.Subscriber("xycar_ultrasonic", Int32MultiArray, callback, queue_size = 1)
while not rospy.is_shutdown():
    if distance[2] > 30:
        drive_go()
    else:
        drive_stop():
    time.sleep(0.1)
