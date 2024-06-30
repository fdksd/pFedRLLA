import time
import torch
import copy
from torch import nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from flcore.clients.clientours import clientAVG
from flcore.servers.serverbase import Server
from threading import Thread
import random
import math

from flcore.trainmodel.auto_encoder import autoencoder 
from utils.data_utils import format_data

from flcore.trainmodel.DDPG import *
import gym
import env


class FedDRL(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        
        self.args = args

        # select slow clients
        self.set_slow_clients()
        self.set_clients(args, clientAVG)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")
        # setting
        self.start_step = 50
        self.update_every = 10
        self.fine_tuning_step = 10

        # model init
        env = gym.make('RFL-v0')
        env.init(self.join_clients) 
        
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        act_bound = [env.action_space.low, env.action_space.high]
        self.env = env
        self.ddpg = DDPG(obs_dim, act_dim, act_bound)
        self.encoder = autoencoder(self.args.num_classes)

        # load pretrained model
        lr = 0.005
        # if self.model_exists():
        #     self.load_model()
        #     lr = 0.01

        self.Budget = []
        self.Reward = []
        self.layer = [4] * self.num_clients
        self.acc_pres = [0 for i in range(args.num_clients)]

        self.criterion = nn.MSELoss(size_average=False)
        self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr=lr, weight_decay=1e-5)
        
        self.embeddings = {}
        self.init_embedding()
 
    def init_embedding(self):
        embedding = self.encoder_train(self.global_model).detach().clone()
        for i in range(self.num_clients):
            self.embeddings[i] = copy.deepcopy(embedding)

    def get_observation(self, id):
        arr = [self.embeddings[i] for i in self.pre_selected]
        arr.append(self.embeddings[id])
        tmp = torch.stack(arr)
        return tmp.reshape(1, -1) 
        # pre_selected = self.pre_selected[:]
        # pre_selected.append(id)
        # return torch.cat([tmp.reshape(1, -1), torch.Tensor([pre_selected])], dim=1)

    def send_models1(self, i):
        assert (len(self.selected_clients) > 0)

        actions, sims = [], []
        for client in self.selected_clients:
            # personalized aggregation
            id = client.id

            o = self.get_observation(id) 
            # get the action
            if i > self.start_step:
                a = self.ddpg.get_action(o, self.ddpg.act_noise)[0]
            else:
                a = self.env.action_space.sample()

            # calculate the simmilarity
            sim = self.get_similarity(id)
            prob = self.normalization(a)

            actions.append(prob)
            sims.append(np.linalg.norm(prob-sim, ord=2))

            self.aggregate_body(id)
            self.aggregate_head(id, prob)

        return actions, sims

    ''''''
    def aggregate_body(self, id):
        if len(self.uploaded_models) <= 0: return 
        
        weights = copy.deepcopy(self.uploaded_weights)
        weights.append(self.clients[id].train_samples)
        totle_weight = sum(weights)
        for i, w in enumerate(weights):
            weights[i] = w / totle_weight
        print("+++++++++++++++++++++++", weights)

        self.global_model = copy.deepcopy(self.clients[id].model)
        for param in self.global_model.parameters():
            param.data = param.data.clone() * weights[-1]
            
        for w, client_model in zip(weights, self.uploaded_models):
            self.add_parameters(w, client_model)

        params = {n: p.data for n, p in self.global_model.named_parameters()}
        idx = 0
        for n, p in self.clients[id].model.named_parameters():
            idx += 1
            if idx <= self.layer[id]:
                p.data = params[n].clone()
    
    def aggregate_head(self, id, a):
        if len(self.uploaded_models) <= 0: return 

        self.global_model = copy.deepcopy(self.clients[id].model)
        for param in self.global_model.parameters():
            param.data = param.data.clone() * a[-1]
            
        for w, client_model in zip(a, self.uploaded_models):
            self.add_parameters(w, client_model)

        params = {n: p.data for n, p in self.global_model.named_parameters()}
        idx = 0
        for n, p in self.clients[id].model.named_parameters():
            idx += 1
            if idx > self.layer[id]:
                p.data = params[n].clone()

    def encoder_train(self, model):
        ''' auto-encoder 训练'''
        model = copy.deepcopy(model)
        params = {n: p.to("cpu") for n, p in model.named_parameters()}
        params = format_data(params)
        loss, embedding = self.encoder(params)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        model.to(self.device)
        return embedding.detach().clone()

    def encoder_emb(self, layer, model):
        ''' auto-encoder 输出embedding'''
        params = {n: p.to("cpu") for n, p in model.named_parameters()}
        params = format_data(params)
        
        if layer == '':
            embedding = self.encoder.get_embedding(params)
        else:    
            embedding = self.encoder.get_layer_embedding(int(layer/2), params)

        model.to(self.device)
        return embedding.detach().clone()

    def get_train_acc(self, client):
        cl, ns, ct = client.validation_metrics()
        return ct*1.0 / ns

    def sample_prob(self, mean, std):
        prob = []
        for m, s in zip(mean, std):
            tmp = torch.empty(1).normal_(mean=m,std=s)
            prob.append(tmp)
        return np.array(prob)

    def normalization(self, data):
        totle_weight = data.sum()
        data = data / totle_weight
        return data

    def get_similarity(self, id):
        sim = np.array([]) 
        emb = self.embeddings[id].reshape(1,-1)
        for pre_id in self.pre_selected:
            emb_c = self.embeddings[pre_id].reshape(1,-1)
            norm1 = torch.norm((emb-emb_c), p=1)
            norm1 = torch.cosine_similarity(emb, emb_c) + norm1
            sim = np.append(sim, norm1)
        sim = np.append(sim, 0) # n+1个权重
        

        min, max = sim.min(), sim.max()
        sim = [(i-min+0.0001)/(max-min+0.0001) for i in sim]
        sim = np.array([math.exp(-i) for i in sim])
        return self.normalization(sim)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        self.uploaded_weights = []
        tot_samples = 0
        self.uploaded_ids = []
        self.uploaded_models = []
        for client in self.selected_clients:
            self.uploaded_weights.append(client.train_samples)
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_models.append(client.model)
    

    def train(self):
        ddpg = copy.deepcopy(self.ddpg)

        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.pre_selected = [c.id for c in self.selected_clients]
            self.selected_clients = self.select_clients()  

            if i == 0:
                self.pre_selected = [c.id for c in self.selected_clients]
            
            actions, sims = self.send_models1(i)
            acc_afts = [self.get_train_acc(c) for c in self.selected_clients]

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            r1 = 0
            tmp_reward = 0
            for idx, client in enumerate(self.selected_clients):
                # 计算reward
                print("============action", actions[idx])
                r1 += math.exp(acc_afts[idx]-1)
                if self.acc_pres[client.id] == 0:
                    self.acc_pres[client.id] = acc_afts[idx]

                r = (math.exp(acc_afts[idx]-1)-1) + (acc_afts[idx]-self.acc_pres[client.id]) - 2*sims[idx]
                tmp_reward += r
                print("* acc_pre: {:.4f}, acc_aft: {:.4f}, sim_diff: {:.4f}, exp: {:.4f}, totle reward: {:.4f}".format(
                    self.acc_pres[client.id], acc_afts[idx], sims[idx], math.exp(acc_afts[idx]-1), r
                ))
                self.acc_pres[client.id] = acc_afts[idx]

                # current state
                o = self.get_observation(client.id) 

                # 更新embeddings数组，并计算新的state
                if i>100 and (i+1)%self.fine_tuning_step != 0:
                    self.embeddings[client.id] = self.encoder_emb('', client.model)
                else:
                    self.embeddings[client.id] = self.encoder_train(client.model)

                # 保存记录
                if i > 0:
                    o2 = self.get_observation(client.id) # new state after choosing action
                    ddpg.replay_buffer.store(o, actions[idx], r, o2, False)
            ddpg.replay_buffer.update_topK(r1, self.join_clients)
            self.Reward.append(tmp_reward / len(self.selected_clients))

            if i >= self.start_step and (i+1) % self.update_every == 0:
                for _ in range(5):
                    batch = ddpg.replay_buffer.sample_batch(32)
                    ddpg.update(data=batch)

            if i%self.eval_gap == 0:
                self.evaluate_one_step()
            self.receive_models()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

        print("\nBest global accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))
        print("\n============ reward =============")
        print(self.Reward)

        self.ddpg = copy.deepcopy(ddpg)
        self.save_global_model()
        self.save_results()
