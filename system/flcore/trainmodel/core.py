import numpy as np
import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F

def combined_shape(length, shape=None):  #返回一个元祖(x,y)
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape) # ()可以理解为元组构造函数，*号将shape多余维度去除

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

class MLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, nn.Tanh)

    def forward(self, obs):
        # Return output from network scaled to action space limits.
        x = self.pi(obs)
        x = F.softmax(x, dim=1)  # log(softmax(x))
        return x

class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)


class MLPActorCritic(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes=(256, 256),
                 activation=nn.ReLU, act_limit = 2.0):
        super().__init__()

        # build policy and value functions
        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).numpy()
