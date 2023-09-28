import torch
import numpy as np
import time
import copy
import torch.nn as nn
from flcore.clients.clientbase import Client


class clientNew(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.mu = args.mu
        self.mu1 = args.mu

        self.global_update_pre = copy.deepcopy(list(self.model.parameters()))
        self.global_update = copy.deepcopy(list(self.model.parameters()))
        self.global_params = copy.deepcopy(list(self.model.parameters()))

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def train(self):
        start_time = time.time()
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.model.train()

        max_local_steps = self.local_steps
        if self.train_slow:
            max_local_steps = np.random.randint(1, max_local_steps // 2)


        # 计算重要度矩阵 & mask
        precision_matrices = {} #重要度
        mask = {}
        params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}# 模型的所有参数
        for n, p in params.items():
            precision_matrices[n] = p.clone().detach().fill_(0) #取zeros_like
            mask[n] = p.clone().detach().fill_(0) #取zeros_like
        # 计算重要度矩阵
        for x, y in trainloader:
            if type(x) == type([]):
                x[0] = x[0].to(self.device)
            else:
                x = x.to(self.device)
            y = y.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(x)
            loss = self.loss(output, y)
            loss.backward()
            for n, p in self.model.named_parameters():                         
                precision_matrices[n].data += p.grad.data ** 2 / len(trainloader)
        # 将重要度中非0元素全部设为1
        for n, item in precision_matrices.items():
            mask[n].data = torch.div(item.data, item.data)
            mask[n].data = torch.where(torch.isnan(mask[n].data), torch.full_like(mask[n].data, 0), mask[n].data)

        # 计算global model update
        self.global_update_pre = copy.deepcopy(self.global_update)
        for u, pre, new in zip(self.global_update, self.global_params, self.model.parameters()):
            u.data = new.data - pre.data

        for step in range(max_local_steps):
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.loss(output, y)
                loss.backward()
                
                for idx, (n, p) in enumerate(self.model.named_parameters()):
                    if p.requires_grad:
                        p.grad.data = p.grad.data + \
                        self.mu1 * mask[n] * (self.global_update[idx] - self.global_update_pre[idx]) + \
                        self.mu * precision_matrices[n] * (p.data - self.global_params[idx])
                self.optimizer.step()

        # self.model.cpu()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def set_parameters(self, model):
        for new_param, global_param, param in zip(model.parameters(), self.global_params, self.model.parameters()):
            global_param.data = new_param.data.clone()
            param.data = new_param.data.clone()
