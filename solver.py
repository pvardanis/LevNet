# import itertools as it
from collections import OrderedDict
# import time
# import random
import torch
# import numpy as np
# import pandas as pd
import torch.nn as nn
# import torch.nn.functional as F
import torchvision.utils 
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from helpers import *
from models import *

# Global variables
set_seed(0)

class Solver(RunManager): # Solver() inherits from RunManager()
    def __init__(self, train_set, valid_set, test_set, config=None, params_dict=None):
        super().__init__()

        # Always same
        self.criterion = nn.NLLLoss()
        self.networks = OrderedDict(Tester=Tester)
        self.optimizers = OrderedDict(Adam=optim.Adam)

        if config:
            # Data Loaders
            self.train_set = train_set
            self.valid_set = valid_set
            self.test_set = test_set
            self.num_workers = config.num_workers

            # Model Settings
            self.model_type = config.model_type
            self.image_size = config.image_size
            self.input_ch = config.input_ch
            self.output_ch = config.output_ch
            self.optimizer = config.optimizer
            
            # Optimizer hyper-parameters
            self.lr = config.lr
            self.beta1 = config.beta1
            self.beta2 = config.beta2
            self.momentum = config.momentum

            # Training settings
            self.num_epochs = config.num_epochs
            self.num_epochs_decay = config.num_epochs_decay
            self.batch_size = config.batch_size
            self.stop_early = config.early_stopping
            self.save_best_model = config.save_best_model
            

        if params_dict: # parameters for training loop (config is more general, so doesn't have to be called in a RunManager instance)
            self.params = params_dict

        # Global settings
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def build_model(self):
        pass

    def train(self):
        
        for run in RunBuilder.get_runs(self.params):
            network = self.networks[self.model_type]().to(self.device) # for each run create a new instance of the network
            train_loader = torch.utils.data.DataLoader(self.train_set, num_workers=self.num_workers, batch_size=run.batch_size, shuffle=True)
            valid_loader = torch.utils.data.DataLoader(self.valid_set, num_workers=self.num_workers, batch_size=run.batch_size, shuffle=True)
            loaders = OrderedDict(train=train_loader, valid=valid_loader)
            optimizer = self.optimizers[self.optimizer](network.parameters(), lr=run.lr)
            
            self.begin_run(run, network, loaders)
            network.train() # keep grads
            for epoch in range(self.num_epochs):
                self.begin_epoch()
                # Train
                for batch_idx, (images, labels) in enumerate(loaders['train']):
                    images, labels = images.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    preds = network(images)
                    loss = self.criterion(preds, labels)
                    loss.backward()
                    optimizer.step()
                    
                    self.track_loss(loss, 'train')
                    self.track_num_correct(preds, labels, 'train')
                
                # Validation
                network.eval() # skips dropout and batch_norm 
                for batch_idx, (images, labels) in enumerate(loaders['valid']):
                    images, labels = images.to(self.device), labels.to(self.device)
                    preds = network(images)
                    loss = self.criterion(preds, labels)

                    self.track_loss(loss, 'valid')
                    self.track_num_correct(preds, labels, 'valid')
                    
                self.end_epoch()
                if self._get_early_stopping():
                    break
                
            self.end_run()
        self.save_results('results')

    