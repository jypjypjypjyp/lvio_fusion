import gym
import numpy as np
from gym import spaces
from numpy.core.numeric import Inf
from std_msgs.msg import Float32


class LvioFusionEnv(gym.Env):
    metadata = {}
    obs_rows = None
    obs_cols = None
    client_create_env = None
    client_step = None

    def __init__(self):
        self.action_space = spaces.Box(np.array([0.1, 0.1, 0.1]), np.array([10, 10, 10]), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=Inf, shape=(
            LvioFusionEnv.obs_rows, LvioFusionEnv.obs_cols), dtype=np.float32)
        self.id = LvioFusionEnv.client_create_env()

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg
        visual = Float32(data=action[0])
        lidar_ground = Float32(data=action[1])
        lidar_surf = Float32(data=action[2])
        resp = LvioFusionEnv.client_step(self.id, visual, lidar_ground, lidar_surf)
        obs = resp.obs.reshape(LvioFusionEnv.obs_rows, LvioFusionEnv.obs_cols, 3)
        return obs, resp.reward, resp.done, {}

    def reset(self):
        resp = LvioFusionEnv.client_create_env()
        self.id = resp.id
        return resp.obs
