import time
import torch
import copy
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from threading import Thread
import random
import math

from flcore.trainmodel.auto_encoder import autoencoder 
from utils.data_utils import format_data

from flcore.trainmodel.DDPG import *
import gym
import env


class FedCos(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        
        self.args = args

        # select slow clients
        self.set_slow_clients()
        self.set_clients(args, clientAVG)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.start_step = 8
        self.update_every = 8
        self.fine_tuning_step = 10
        self.encoder = autoencoder()
        lr = 0.001
        # if self.model_exists():
        #     self.load_model()
        #     lr = 0.0001
        self.Budget = []
        self.layer = [self.args.mu] * self.num_clients
        self.fc_avg = [0] * self.num_clients

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr=lr, weight_decay=1e-5)
        
        self.embeddings = {}
        self.init_embedding()
 
    def init_embedding(self):
        embedding = self.encoder_train(self.global_model).detach().clone()
        for i in range(self.num_clients):
            self.embeddings[i] = copy.deepcopy(embedding)

    def get_observation(self, id):
        arr = [self.embeddings[c.id] for c in self.pre_selected]
        arr.append(self.embeddings[id])
        tmp = torch.stack(arr)
        return tmp.reshape(1, -1)

    '''
    def aggregate_parameters1(self, id, a):
        a = a.flatten()
        Epsilon = 0.6
        idx_sorted = np.argsort(-a) # -a中元素从小到大的下标list

        top_k = int(self.num_clients * self.join_ratio)
        num = int(top_k * 0.8)
        random_num = top_k - num

        print("==================tar id", id, "selected", idx_sorted[:top_k])

        selected_idx = idx_sorted[:num]
        flag = np.random.random()
        if flag > 0.6:
            e_idx = np.random.choice(idx_sorted[num:], random_num)
            selected_idx = np.append(selected_idx, e_idx)
        else:
            selected_idx = idx_sorted[:top_k]

        # print("client ", id, "aggregation polciy is", a, "\n", idx_sorted)
        totle_weight = a[id]
        for param in self.clients[id].model.parameters():
            param.data = param.data.clone() * a[id]
        for i in selected_idx:
            if i == id: continue
            totle_weight += a[i]
            params = {n: p for n, p in self.clients[i].model.named_parameters() if p.requires_grad}
            for n, p in self.clients[id].model.named_parameters():
                if p.requires_grad == True:
                    p.data += (a[i] * params[n])
        for n, p in self.clients[id].model.named_parameters():
            if p.requires_grad == True:
                p.data = p.data / totle_weight
    '''

    def aggregate_parameters1(self, id, a):
        assert (len(self.uploaded_models) > 0)
        base = a.max() - a.min() if a.max() > 0 else -a.min()
        totle_weight = a.sum() + base * len(a)
        a = (a + np.array([base]*len(a)) ) / totle_weight
        print("==================action", a)

        self.global_model = copy.deepcopy(self.clients[0].model)
        for param in self.global_model.parameters():
            param.data.zero_()
            
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
        params = {n: p.to("cpu") for n, p in model.named_parameters() if p.requires_grad}
        params = format_data(params)
        output, embedding = self.encoder(params)
        loss = self.criterion(output[0], params[0])
        for i in range(1, 2):
            loss += self.criterion(output[i], params[i])
        print("=====================auto-encorder loss", loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        model.to(self.device)
        return embedding.detach().clone()

    def encoder_emb(self, layer, model):
        ''' auto-encoder 输出embedding'''
        params = {n: p.to("cpu") for n, p in model.named_parameters() if p.requires_grad}
        params = format_data(params)
        
        if layer == '':
            embedding = self.encoder.get_embedding(params)
        else:    
            embedding = self.encoder.get_layer_embedding(int(layer/2), params)

        model.to(self.device)
        return embedding.detach().clone()

    def get_train_acc(self, client):
        # cl, ns, ct = client.train_metrics()
        cl, ns, ct = client.validation_metrics()
        return ct*1.0 / ns

    def cmp_emb(self):
        rets = []
        # tar = self.embeddings[0].reshape(1, -1)
        # for i in range(1, self.num_clients):
        #     tmp = self.embeddings[i].reshape(1, -1)
        #     cos_sim = torch.cosine_similarity(tar, tmp)
        #     norm2 = torch.norm((tar-tmp), p=2)
        #     rets.append((cos_sim, norm2))
        # print("OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO", rets)
        tar = self.encoder_emb("fc", self.clients[0].model)
        for idx, c in enumerate(self.clients):
            tmp = self.encoder_emb("fc", c.model)
            norm2 = torch.norm((tar-tmp), p=2)
            rets.append(norm2)
        print("OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOfc laywe", rets)

        rets = []
        tar = self.encoder_emb("conv", self.clients[0].model)
        for idx, c in enumerate(self.clients):
            tmp = self.encoder_emb("conv", c.model)
            norm2 = torch.norm((tar-tmp), p=2)
            rets.append(norm2)
        print("OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOconv laywe", rets)

    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.pre_selected = copy.deepcopy(self.selected_clients)
            self.selected_clients = self.select_clients()
            self.send_models()
            
            for client in self.selected_clients:
                # personalized aggregation
                id = client.id
                if i>0:
                    a = np.array([]) 
                    emb = self.embeddings[id].reshape(1,-1)
                    for c in self.pre_selected:
                        print(c.id, end=" ")
                        emb_c = self.embeddings[c.id].reshape(1,-1)
                        cos_c = torch.cosine_similarity(emb, emb_c)
                        a = np.append(a, cos_c)
                    self.aggregate_parameters1(client.id, a)

                client.train()
                acc_aft = self.get_train_acc(client)
                print("===========acc aft", acc_aft)
                
                # if i > self.start_step:
                #     if fc_value < conv_value and self.layer[id] > 2:
                #         self.layer[id] = self.layer[id]-1
                #     elif fc_value > conv_value and self.layer[id] < 6:
                #         self.layer[id] = self.layer[id]+1

                # 更新embeddings数组，并计算新的state
                if i%self.fine_tuning_step != 0:
                    self.embeddings[client.id] = self.encoder_emb('', client.model)
                else:
                    self.embeddings[client.id] = self.encoder_train(client.model)

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()
            
            self.receive_models()
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

        print("\nBest global accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.draw()
        # self.save_global_model()

    # 对样本进行预处理并画图
    def plot_embedding(self, data, label, title):
        x_min, x_max = np.min(data, 0), np.max(data, 0)
        data = (data - x_min) / (x_max - x_min)		# 对数据进行归一化处理
        fig = plt.figure()		# 创建图形实例
        ax = plt.subplot(111)		# 创建子图
        # 遍历所有样本
        print("=======", data)
        for i in range(data.shape[0]):
            # 在图中为每个数据点画出标签
            plt.text(data[i, 0], data[i, 1], str(label[i]), color=plt.cm.Set1(label[i] / 10),
                    fontdict={'weight': 'bold', 'size': 7})
        plt.xticks()		# 指定坐标的刻度
        plt.yticks()
        plt.title(title, fontsize=14)
        # 返回值
        return fig

    def draw(self):
        embs = np.array([i.reshape(1, -1).tolist()[0] for i in self.embeddings.values()])
        lables = np.array([0 if i<10 else 1 for i in range(self.num_clients)])

        ts = TSNE(n_components=2, init='pca', random_state=0)
        results = ts.fit_transform(embs)

        # 调用函数，绘制图像
        fig = self.plot_embedding(results, lables, 't-SNE Embedding of digits')
        # 显示图像
        plt.savefig("./a.png")

        params = []
        for c in self.clients:
            aa = {n: p for n, p in c.model.named_parameters()}
            tmp = torch.cat([p.clone().reshape(1, -1) for idx, p in enumerate(aa.values())], dim=1)
            params.append(tmp.reshape(1, -1).tolist()[0])
        params = np.array(params)
        
        results = ts.fit_transform(params)
        # 调用函数，绘制图像
        fig = self.plot_embedding(results, lables, 't-SNE params of digits')
        # 显示图像
        plt.savefig("./b.png")


