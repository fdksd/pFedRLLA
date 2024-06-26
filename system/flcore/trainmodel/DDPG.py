import numpy as np
from copy import deepcopy
from torch.optim import Adam
import torch
import flcore.trainmodel.core_mlp as core

# https://gitcode.com/openai/spinningup/blob/master/spinup/algos/pytorch/ddpg/

class ReplayBuffer:   # 输入为size；obs的维度(3,)：这里在内部对其解运算成3；action的维度3
    """
    A simple FIFO experience replay buffer for DDPG agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def update_topK(self, r1, k):
        begin = self.ptr - k
        base = r1*1.0 / k
        # for i in range(begin, self.ptr):
        #     base += (1.0/k * self.rew_buf[i])
        print("=================", base)
        for i in range(begin, self.ptr):
            self.rew_buf[i] = self.rew_buf[i] - base

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}

class DDPG:
    def __init__(self, obs_dim, act_dim, act_bound, actor_critic=core.MLPActorCritic, seed=0, 
                replay_size=int(1e5), gamma=0.9, polyak=0.96, pi_lr=1e-3, q_lr=1e-3, act_noise=0.03):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_bound = act_bound
        self.gamma = gamma
        self.polyak = polyak
        self.act_noise = act_noise

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.ac = actor_critic(obs_dim, act_dim, act_limit = 1.0)
        self.ac_targ = deepcopy(self.ac)

        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=pi_lr)
        self.q_optimizer = Adam(self.ac.q.parameters(), lr=q_lr)

        for p in self.ac_targ.parameters():
            p.requires_grad = False

        self.replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    def myto(self, device):
        self.ac.to(device)
        self.ac_targ.to(device)

    def compute_loss_q(self, data):   #返回(q网络loss, q网络输出的状态动作值即Q值)
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        
        q = self.ac.q(o, a)

        # Bellman backup for Q function
        with torch.no_grad():
            q_pi_targ = self.ac_targ.q(o2, self.ac_targ.pi(o2))
            backup = r + self.gamma * (1 - d) * q_pi_targ

        action_var = 0
        for aa in a:
            action_var += np.var(aa.numpy())

        # MSE loss against Bellman backup
        loss_q = ((q - backup)**2).mean() - 0.2 * action_var
        print("======================value func loss", loss_q)

        return loss_q # 这里的loss_q没加负号说明是最小化，很好理解，TD正是用函数逼近器去逼近backup，误差自然越小越好

    def compute_loss_pi(self, data):
        o = data['obs']
        # o = o.view([len(o), int(self.obs_dim/32), 32])
        q_pi = self.ac.q(o, self.ac.pi(o)) 
        return -q_pi.mean()  # 这里的负号表明是最大化q_pi,即最大化在当前state策略做出的action的Q值

    def update(self, data):
        # First run one gradient descent step for Q.
        self.q_optimizer.zero_grad()
        loss_q = self.compute_loss_q(data)
        print("=======================Q net loss", loss_q)
        loss_q.backward()
        self.q_optimizer.step()

        # Freeze Q-network so you don't waste computational effort
        # computing gradients for it during the policy learning step.
        for p in self.ac.q.parameters():
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi = self.compute_loss_pi(data)
        print("====================actor loss", loss_pi)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for p in self.ac.q.parameters():
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def get_action(self, o, noise_scale):
        # o = o.view([len(o), int(self.obs_dim/32), 32]) 
        a = self.ac.act(torch.as_tensor(o, dtype=torch.float32))
        a += noise_scale * np.random.randn(self.act_dim)
        return a
        # return 0.5 * (a + np.ones(self.act_dim))
        # return np.clip(a, self.act_bound[0], self.act_bound[1]) # space范围在[-1,1]之间
