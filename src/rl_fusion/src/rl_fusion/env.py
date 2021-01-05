import gym
import rospy
import numpy as np
from gym import spaces
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float64MultiArray


class LvioFusionEnv(gym.Env):
    metadata = {}
    camera_height = None
    camera_width = None
    clt_create_env = None
    clt_step = None

    def __init__(self):
        self.action_space = spaces.Box(np.array([0, 0, 0, 0, 0, 0, 0, 0]), np.array([
                                       1, 1, 1, 1, 1, 1, 1, 1]), dtype=np.float64)
        self.observation_space = spaces.Box(low=0, high=255, shape=(
            LvioFusionEnv.camera_height, LvioFusionEnv.camera_width), dtype=np.uint8)
        self.cv_bridge = CvBridge()
        self.id = LvioFusionEnv.clt_create_env()

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        visual = Float64MultiArray(data=action[:2])
        lidar_ground = Float64MultiArray(data=action[2:5])
        lidar_surf = Float64MultiArray(data=action[5:8])
        imu = Float64MultiArray(data=np.ones(9))
        resp = LvioFusionEnv.clt_step(self.id, visual, lidar_ground, lidar_surf, imu)
        obs = numpy.asarray(self.cv_bridge.imgmsg_to_cv2(resp.image, "mono8"))
        return obs, resp.reward, resp.done, {}

    def reset(self):
        self.id = LvioFusionEnv.clt_create_env()
        return self.state
