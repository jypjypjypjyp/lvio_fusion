#! /home/jyp/.miniconda3/envs/pytorch/bin/python
# -*- coding: utf-8 -*-

import rospy
from adaptive_fusion import assistant
from std_msgs.msg import String


def publish_callback(event):
    hello_str = "hello world %s" % event.current_real.to_sec()
    # hello_str = assistant.funcname()
    rospy.loginfo(hello_str)
    pub.publish(hello_str)


if __name__ == '__main__':
    try:
        rospy.loginfo('hello_str')
        rospy.init_node('adaptive_fusion', anonymous=True)
        pub = rospy.Publisher('chatter', String, queue_size=10)
        timer = rospy.Timer(rospy.Duration(1. / 10), publish_callback)  # 10Hz
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
