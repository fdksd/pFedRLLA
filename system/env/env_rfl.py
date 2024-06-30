'''
当更改client num时:
 - 需要修改RFL中的action_space和observation_space
 - 同时, 需要在serveravg中, 修改保存, 训练DDPG的一些阈值
'''

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np


class RFL(gym.Env):
    def __init__(self):
        self.action_space = spaces.Box(low=0, high=1, shape=[100,], dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=[128*100, 1], dtype=np.float32)

    def step(self, action):
        # assert self.action_space.contains(action)
        print("step")
        reward = 0
        done = False
        return None, reward, done, {}

    def init(self, client_num):
        self.action_space = spaces.Box(low=0, high=1, shape=[(client_num+1),], dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=[128*(client_num+1), 1], dtype=np.float32)
        
    def reset(self, client_num):
        print("reset")
        pass