import torch
import os
import numpy as np
import pandas as pd
# import h5py
import copy
import time
import random

from utils.data_utils import read_client_data

class Server(object):
    def __init__(self, args, times):
        # Set up the main attributes
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.global_rounds = args.global_rounds
        self.local_steps = args.local_steps
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.join_clients = int(self.num_clients * self.join_ratio)
        self.algorithm = args.algorithm
        self.time_select = args.time_select
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.save_folder_name = args.save_folder_name
        self.top_cnt = 100

        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []

        self.ft_test_acc = []
        self.ft_test_auc = []
        self.std_ft_test_acc = []
        self.std_ft_test_auc = []

        self.times = args.times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate

        self.Budget = []

    def set_clients(self, args, clientObj):
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=train_slow, 
                            send_slow=send_slow)
            self.clients.append(client)

    # random select slow clients
    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients

    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(
            self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(
            self.send_slow_rate)

    def select_clients(self):
        selected_clients = list(np.random.choice(self.clients, self.join_clients, replace=False))
        
        # selected_clients = list(np.array(self.clients).take([0, 1, 2, 5, 6, 10, 11, 15, 16])) # cifar10
        # selected_clients = list(np.array(self.clients).take([0, 1, 2, 5, 6, 9, 10, 13, 14])) # cifar10

        return selected_clients

    def send_models(self):
        assert (len(self.selected_clients) > 0)

        for client in self.selected_clients:
            client.set_parameters(self.global_model)

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
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples
                
    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)
        
        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()
            
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    # def add_parameters1(self, w, client_model):
    #     params = {n: p for n, p in client_model.named_parameters()}
    #     for n, p in self.global_model.named_parameters():
    #         if "fc" in n:
    #             p.data += params[n].clone() * w

    def save_global_model(self):
        model_path = os.path.join("models", 'Cifar')
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path1 = os.path.join(model_path, self.algorithm + "_encoder3.28" + ".pt")
        model_path2 = os.path.join(model_path, self.algorithm + "_drl3.28" + ".pt")
        torch.save(self.encoder, model_path1)
        torch.save(self.ddpg, model_path2)

    def load_model(self):
        model_path = os.path.join("models", 'Cifar')
        model_path1 = os.path.join(model_path, self.algorithm + "_encoder3.28" + ".pt")
        model_path2 = os.path.join(model_path, self.algorithm + "_drl3.28" + ".pt")
        assert (os.path.exists(model_path1))
        assert (os.path.exists(model_path2))
        self.encoder = torch.load(model_path1)
        self.ddpg = torch.load(model_path2)

    def model_exists(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_encoder100" + ".pt")
        return os.path.exists(model_path)
        
    def save_results(self):
        self.algorithm = self.algorithm + '_frac' + str(self.args.join_ratio) + '_lr' + str(self.args.local_learning_rate) + \
                        '_le' + str(self.args.local_steps) + '_bs' + str(self.args.batch_size) + '_l' + str(self.args.lamda) 
        algo = self.dataset + "_" + self.algorithm
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.rs_test_acc)):
            algo = algo + "_" + self.goal + "_" + str(self.times)
            file_path = result_path + "{}.csv".format(algo)

            data = []
            # cols = ['test_acc', 'test_auc', 'train_loss']
            cols = ['test_acc', 'test_auc', 'time']
            for idx, test_acc in enumerate(self.rs_test_acc):
                data.append([test_acc, self.rs_test_auc[idx], self.Budget[idx]])
            if self.ft_test_acc != []:
                cols.extend(['ft_test_acc', 'ft_test_auc', 'std_ft_test_acc', 'std_ft_test_auc'])
                for idx, ft_test_acc in enumerate(self.ft_test_acc):
                    data[idx].extend([ft_test_acc, self.ft_test_auc[idx], self.std_ft_test_acc[idx], self.std_ft_test_auc[idx]])
            
            data = pd.DataFrame(np.array(data), columns=cols)
            data.to_csv(file_path, mode='a', index=False)

    def save_item(self, item, item_name):
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        torch.save(item, os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def load_item(self, item_name):
        return torch.load(os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def test_metrics(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
        # for c in self.selected_clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [0]

        return ids, num_samples, tot_correct, tot_auc

    def train_metrics(self):
        num_samples = []
        tot_correct = []
        losses = []
        
        self.loss_term = []
        for c in self.selected_clients:
            cl, ns, ct = c.train_metrics()
            if (ns == 0):
                print("client ", c.id, " cl=", cl, " ns=", ns, " ct=", ct);
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(cl*1.0)
            if c in self.selected_clients:
                self.loss_term.append(cl*1.0/ns)

        ids = [0]

        return ids, num_samples, losses, tot_correct

    def evaluate(self, acc=None, loss=None):
        stats = self.test_metrics()
        # stats_train = self.train_metrics()
        
        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        test_auc = sum(stats[3])*1.0 / sum(stats[1])
        # self.train_acc = sum(stats_train[3])*1.0 / sum(stats[1])
        # train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]
        
        if acc == None:
            self.rs_test_acc.append(test_acc)
            self.rs_test_auc.append(np.std(accs))
        else:
            acc.append(test_acc)
        
        # if loss == None:
        #     self.rs_train_loss.append(train_loss)
        # else:
        #     loss.append(train_loss)

        # print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accurancy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))

    def print_(self, test_acc, test_auc, train_loss):
        print("Average Test Accurancy: {:.4f}".format(test_acc))
        print("Average Test AUC: {:.4f}".format(test_auc))
        print("Average Train Loss: {:.4f}".format(train_loss))

    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        for acc_ls in acc_lss:
            if top_cnt != None and div_value != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                if find_top:
                    pass
                else:
                    return False
            elif div_value != None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True

    def evaluate_one_step(self):
        # models_temp = []
        # for c in self.selected_clients:
        #     models_temp.append(copy.deepcopy(c.model))
        #     c.train_one_step()

        stats = self.test_metrics()

        # set local model back to client for training process
        # for i, c in enumerate(self.selected_clients):
        #     c.clone_model(models_temp[i], c.model)

        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        test_auc = sum(stats[3])*1.0 / sum(stats[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]
        
        self.ft_test_acc.append(test_acc)
        self.ft_test_auc.append(test_auc)
        self.std_ft_test_acc.append(np.std(accs))
        self.std_ft_test_auc.append(np.std(aucs))
        print("Average FT Test Accurancy: {:.4f}".format(test_acc))