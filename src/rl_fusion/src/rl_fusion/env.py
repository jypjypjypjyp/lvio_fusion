import gym
from gym import spaces
import numpy as np

class FactorGraphEnv(gym.Env):
    metadata = {}

    def __init__(self):

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(8)

        self.x=[140,220,300,380,460,140,300,460]
        self.y=[250,250,250,250,250,150,150,150]

        self.rewards = dict();        #回报的数据结构为字典
        self.rewards[5] = -1.0
        self.rewards[6] = 1.0
        self.rewards[7] = -1.0

        self.map = dict()
        self.map[0] = [-1, 5, -1, 1]
        self.map[1] = [-1, -1, 0, 2]
        self.map[2] = [-1, 6, 1, 3]
        self.map[3] = [-1, -1, 2, 4]
        self.map[4] = [-1, 7, 3, -1]

        self.gamma = 0.8         #折扣因子
        self.viewer = None
        self.state = None

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg
        
        #系统当前状态
        state = self.state
        if state not in self.map:
            return state, 0, True, {}

        #状态转移
        next_state = self.map[state][action]

        if next_state == -1:
            return state, 0.0, False, {}
        else:
            self.state = next_state
            if next_state not in self.map:
                return next_state, self.rewards[next_state], True, {}
            else:
                return next_state, 0.0 , False, {}

    def reset(self):
        self.state = int(random.random() * len(self.map))
        return self.state
    