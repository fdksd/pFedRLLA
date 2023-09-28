import os
import datetime
import copy
import numpy as np

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST

class autoencoder(nn.Module):
    def __init__(self, num_classes, in_features=3, dim=1600):
        super(autoencoder, self).__init__()
        
        self.encoder = nn.LSTM(32*32, 128, num_layers=2, batch_first=True)
        self.decoder = nn.LSTM(128, 32*32, num_layers=2, batch_first=True)

        self.criterion = nn.MSELoss()
 
    def forward(self, inputs):
        f1, s, v = torch.pca_lowrank(inputs[0].reshape(32, -1), q=32, center=True, niter=4)
        f2, _, _ = torch.pca_lowrank(inputs[1].reshape(32, -1), q=32, center=True, niter=4)
        # f3, _, _ = torch.pca_lowrank(inputs[2].reshape(32, -1), q=32, center=True, niter=4)
        # f4, _, _ = torch.pca_lowrank(inputs[3].reshape(32, -1), q=32, center=True, niter=4)
        # x = torch.stack((torch.stack((f1, f2, f3, f4)), )) 
        x = torch.stack((torch.stack((f1, f2)), )) 
        x = x.view(1, 2, -1)

        embedding, (n, c) = self.encoder(x)
        x, (n, c) = self.decoder(embedding)
        
        loss = self.criterion(x[0][0].reshape(32, -1), f1)
        loss += self.criterion(x[0][1].reshape(32, -1), f2)
        # loss += self.criterion(x[0][2].reshape(32, -1), f3)
        # loss += self.criterion(x[0][3].reshape(32, -1), f4)

        emb = embedding[0][0]
        for i in range(1, embedding.size()[1]):
            emb = (emb + embedding[0][i])
        return loss, emb

    def get_embedding(self, inputs):
        with torch.no_grad():
            f1, _, _ = torch.pca_lowrank(inputs[0].reshape(32, -1), q=32, center=True, niter=4)
            f2, _, _ = torch.pca_lowrank(inputs[1].reshape(32, -1), q=32, center=True, niter=4)
            # f3, _, _ = torch.pca_lowrank(inputs[2].reshape(32, -1), q=32, center=True, niter=4)
            # f4, _, _ = torch.pca_lowrank(inputs[3].reshape(32, -1), q=32, center=True, niter=4)

            # x = torch.stack((torch.stack((f1, f2, f3, f4)), )) 
            x = torch.stack((torch.stack((f1, f2)), )) 
            x = x.view(1, 2, -1)

            embedding, (n, c) = self.encoder(x)
        
        emb = embedding[0][0]
        for i in range(1, embedding.size()[1]):
            emb = (emb + embedding[0][i])
        return emb
