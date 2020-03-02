import itertools as it
from collections import OrderedDict
import time
import random
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils 
import torchvision.models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from helpers import *
from models import *

# Global variables
set_seed(0)
use_cuda = torch.cuda.is_available() # True if cuda is available
criterion = nn.NLLLoss()

def train(n_epochs, train_set, valid_set, network, params):
    m = RunManager()
    for run in RunBuilder.get_runs(params):
        network = network.cuda() if use_cuda else network
        train_loader = torch.utils.data.DataLoader(train_set, num_workers=0, batch_size=run.batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_set, num_workers=0, batch_size=run.batch_size, shuffle=True)
        loaders = OrderedDict(train=train_loader, valid=valid_loader)
        optimizer = optim.Adam(network.parameters(), lr=run.lr)
        
        m.begin_run(run, network, loaders, stop_early=True, save_best_model=False)
        network.train() # keep grads
        for epoch in range(5):
            m.begin_epoch()
            
            # Train
            for batch_idx, (images, labels) in enumerate(loaders['train']):
                
                images, labels = images.cuda(), labels.cuda()
                optimizer.zero_grad()
                preds = network(images)
                loss = criterion(preds, labels)
                loss.backward()
                optimizer.step()
                
                m.track_loss(loss, 'train')
                m.track_num_correct(preds, labels, 'train')
            
            # Validation
            network.eval() # skips dropout and batch_norm 
            for batch_idx, (images, labels) in enumerate(loaders['valid']):

                images, labels = images.cuda(), labels.cuda()
                preds = network(images)
                loss = criterion(preds, labels)

                m.track_loss(loss, 'valid')
                m.track_num_correct(preds, labels, 'valid')
                
            m.end_epoch()
            if m._get_early_stopping():
                break
            
        m.end_run()
        
    m.save_results('results')

if __name__ == "__main__":

    transform = transforms.Compose([transforms.ToTensor()])

    dataset = datasets.FashionMNIST(
                        root='./data',
                        train=True,
                        download=True,
                        transform=transform)

    train_set, valid_set = torch.utils.data.random_split(dataset, [50000, 10000])

    params = OrderedDict(
            lr=[0.01, 0.001],
            batch_size=[128, 256],
            patience=[1]
    )

    train(5, train_set, valid_set, Tester(), params)

    

    