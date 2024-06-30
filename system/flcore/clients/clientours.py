import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client
from utils.privacy import *


class clientAVG(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        self.epoch = 0
        self.step = args.lr_step # learning rate update every step

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.5)

        # differential privacy
        if self.privacy:
            check_dp(self.model)
            initialize_dp(self.model, self.optimizer, self.sample_rate, self.dp_sigma)

    def adjust_learning_rate(self):
        """Sets the learning rate to the initial LR decayed by 10 every 20 epochs"""
        self.learning_rate = self.learning_rate * (0.9 ** (self.epoch // self.step))
        if self.epoch // self.step > 0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate
            self.epoch = 0
        else:
            self.epoch += 1

    def train(self):
        trainloader = self.load_train_data()
        
        start_time = time.time()

        self.model.train()

        max_local_steps = self.local_steps
        if self.train_slow:
            max_local_steps = np.random.randint(1, max_local_steps // 2)

        for step in range(max_local_steps):
            self.adjust_learning_rate()
            for i, (x, y) in enumerate(trainloader):
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
                if self.privacy:
                    dp_step(self.optimizer, i, len(trainloader))
                else:
                    self.optimizer.step()

        # self.model.cpu()
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        if self.privacy:
            res, DELTA = get_dp_params(self.optimizer)
            print(f"Client {self.id}", f"(ε = {res[0]:.2f}, δ = {DELTA}) for α = {res[1]}")

    # def set_parameters(self, model):
    #     for new_param, old_param in zip(model.parameters(), self.model.parameters()):
    #         old_param.data = new_param.data.clone()

