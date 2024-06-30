import copy
import torch
import torch.nn as nn
import numpy as np
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data


class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        self.model = copy.deepcopy(args.model)
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # integer
        self.save_folder_name = args.save_folder_name

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_steps = args.local_steps

        # check BatchNorm
        self.has_BatchNorm = False
        for layer in self.model.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break

        self.train_slow = kwargs['train_slow']
        self.send_slow = kwargs['send_slow']
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        self.privacy = args.privacy
        self.dp_sigma = args.dp_sigma
        self.sample_rate = self.batch_size / self.train_samples

        self.participate_num = 1
        self.validation_data = None


    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        nid = self.id
        # if self.id in [5, 6]:
        #     nid = 3 if self.id == 5 else 1 # cifar100
            # nid = 2 if self.id == 5 else 1 # cifar10
        train_data = read_client_data(self.dataset, nid, is_train=True)
        # if self.id in [5, 6]:
        #     tmp = read_client_data(self.dataset, nid+5, is_train=True) #cifar100
        #     # tmp = read_client_data(self.dataset, nid+4, is_train=True)
        #     train_data.extend(tmp)
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=True)

    def load_validation_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        nid = self.id
        # if self.id in [5, 6]:
        #     nid = 3 if self.id == 5 else 1
            # nid = 2 if self.id == 5 else 1
        validation_data = read_client_data(self.dataset, nid, is_train=True, is_validation=True)
        # if self.id in [5, 6]:
        #     tmp = read_client_data(self.dataset, nid+5, is_train=True, is_validation=True)
        #     # tmp = read_client_data(self.dataset, nid+4, is_train=True, is_validation=True)
        #     validation_data.extend(tmp)
        return DataLoader(validation_data, batch_size, drop_last=True, shuffle=True)

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        nid = self.id
        # if self.id in [5, 6]:
        #     nid = 3 if self.id == 5 else 1
            # nid = 2 if self.id == 5 else 1
        test_data = read_client_data(self.dataset, nid, is_train=False)
        # if self.id in [5, 6]:
        #     tmp = read_client_data(self.dataset, nid+5, is_train=False)
        #     # tmp = read_client_data(self.dataset, nid+4, is_train=False)
        #     test_data.extend(tmp)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=True)
        
    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        
        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                y_true.append(label_binarize(y.detach().cpu().numpy(), classes=np.arange(self.num_classes)))

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = 0
        # auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        
        return test_acc, test_num, auc

    def train_metrics(self):
        trainloader = self.load_train_data(1)
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        train_acc = 0
        train_num = 0
        loss = 0
        for x, y in trainloader:
            if type(x) == type([]):
                x[0] = x[0].to(self.device)
            else:
                x = x.to(self.device)
            y = y.to(self.device)
            output = self.model(x)

            train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            train_num += y.shape[0]
            loss += self.loss(output, y).item() * y.shape[0]


        return loss, train_num, train_acc

    ''''''
    def validation_metrics(self):
        validation_loader = self.load_validation_data(batch_size=8)
        self.model.eval()

        train_acc = 0
        train_num = 0
        loss = 0
        for x, y in validation_loader:
            if type(x) == type([]):
                x[0] = x[0].to(self.device)
            else:
                x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(x)

            train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            train_num += y.shape[0]
            loss += self.loss(output, y).item() * y.shape[0]
            

        return loss, train_num, train_acc
    
    def train_one_step(self):
        trainloader = self.load_train_data()
        
        self.model.train()

        max_local_steps = 1

        for step in range(max_local_steps):
            # self.adjust_learning_rate()
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

    # def get_next_train_batch(self):
    #     try:
    #         # Samples a new batch for persionalizing
    #         (x, y) = next(self.iter_trainloader)
    #     except StopIteration:
    #         # restart the generator if the previous generator is exhausted.
    #         self.iter_trainloader = iter(self.trainloader)
    #         (x, y) = next(self.iter_trainloader)

    #     if type(x) == type([]):
    #         x = x[0]
    #     x = x.to(self.device)
    #     y = y.to(self.device)

    #     return x, y


    def save_item(self, item, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(item, os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    def load_item(self, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        return torch.load(os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    # @staticmethod
    # def model_exists():
    #     return os.path.exists(os.path.join("models", "server" + ".pt"))