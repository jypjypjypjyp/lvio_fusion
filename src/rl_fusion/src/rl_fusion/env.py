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
        self.action_space = spaces.Box(np.array([ 0, 0]), np.array([1000, 1000]), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=Inf, shape=(
            LvioFusionEnv.obs_rows, LvioFusionEnv.obs_cols, 3), dtype=np.float32)
        self.id = LvioFusionEnv.client_create_env()

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg
        resp = LvioFusionEnv.client_step(self.id, 1, action[0], action[1])
        obs = np.array(resp.obs)
        if obs.size ==0 or resp.done:
            obs = np.zeros(shape=(LvioFusionEnv.obs_rows, LvioFusionEnv.obs_cols, 3), dtype=np.float32)
            resp.done = True
        else:
            obs = obs.reshape(LvioFusionEnv.obs_rows, LvioFusionEnv.obs_cols, 3)
        return obs, resp.reward, resp.done, {}

    def reset(self):
        resp = LvioFusionEnv.client_create_env()
        self.id = resp.id
        obs = np.array(resp.obs).reshape(LvioFusionEnv.obs_rows, LvioFusionEnv.obs_cols, 3)
        return obs
