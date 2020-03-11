from collections import OrderedDict
import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision.utils 
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision.models
import os
from helpers import *
from models import *
from barbar import Bar
from IPython.display import clear_output
from torchsummary import summary
                
# Global variables
set_seed(0)

class Solver(object): 
    def __init__(self, train_set, valid_set, test_set, config=None):
        super().__init__()

        # Global settings
        if config.tpu:
            import torch_xla
            import torch_xla.core.xla_model as xm 

            self.device = xm.xla_device()
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Loss/Optimizer
        if config.loss == 'mse':
            self.criterion = nn.MSELoss()
        elif config.loss == 'l1':
            self.criterion = nn.L1Loss()
        elif config.loss == 'mod':
            self.criterion = MSEWrap 
        elif config.loss == 'atan':
            self.criterion = Atan 
        elif config.loss == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss()
        elif config.loss == 'cos':
            self.criterion = Cosine

        self.lr_scheduler = config.lr_scheduler

        self.optimizers = OrderedDict(adam=optim.Adam, sgd=optim.SGD)
        self.schedulers = OrderedDict(reduce_lr=optim.lr_scheduler.ReduceLROnPlateau, step_lr=optim.lr_scheduler.StepLR)
        
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
        self.optimizer = config.optimizers # list of optimizers
        
        # Create params dictionary
        self.params = OrderedDict()
        
        # Optimizer hyper-parameters
        self.params['optimizer'] = config.optimizers
        self.params['lr'] = config.lr
        self.betas = [config.beta1, config.beta2] # Adam
        self.momentum = config.momentum # SGD

        # Training settings
        self.num_epochs = config.num_epochs
        self.num_epochs_decay = config.num_epochs_decay
        self.params['batch_size'] = config.batch_size
        self.stop_early = config.early_stopping
        self.params['patience'] = config.patience 
        self.save_best_model = config.save_best_model

        print(self.params)
                
    def build_model(self):

        if self.model_type in ['vgg-16', 'vgg-16-bn']:
            
            assert self.image_size == 224, "ERROR: Wrong image size."

            model = torchvision.models.vgg16(pretrained=False) if self.model_type == 'vgg-16' else torchvision.models.vgg16_bn(pretrained=False)
            num_features = model.classifier[6].in_features
            features = list(model.classifier.children())[:-3] # Remove last layer and non-linearity
            features.extend([nn.Dropout(p=0.5), nn.Linear(num_features, self.output_ch * 2)]) # Add our layer with output_ch
            model.classifier = nn.Sequential(*features) # Replace the model classifier
            
            for param in model.features.parameters(): # disable grad for trained layers
                param.requires_grad = False

            # summary(model.to('cuda'), (3, 224, 224))
            return model

        elif self.model_type == 'tester': return Tester()
        elif self.model_type == 'levnet': return LevNet()
    
    def train(self):
        m = RunManager(self.save_best_model, self.stop_early)
        if global_vars.console: 
            global_vars.cls() 
        else: 
            clear_output(wait=True)

        for run in RunBuilder.get_runs(self.params):
            network = self.build_model().to(self.device, dtype=torch.float) # this returns a new instance of the network .to(self.device)
            train_loader = torch.utils.data.DataLoader(self.train_set, num_workers=self.num_workers, batch_size=run.batch_size, shuffle=True)
            valid_loader = torch.utils.data.DataLoader(self.valid_set, num_workers=self.num_workers, batch_size=run.batch_size, shuffle=True)
            loaders = OrderedDict(train=train_loader, valid=valid_loader)
            
            if run.optimizer == 'adam':
                optimizer = self.optimizers[run.optimizer](network.parameters(), lr=run.lr, betas=self.betas)
            elif run.optimizer == 'sgd':
                optimizer = self.optimizers[run.optimizer](network.parameters(), lr=run.lr, momentum=self.momentum)

            if self.lr_scheduler: scheduler = self.schedulers['reduce_lr'](optimizer, patience=run.patience, \
                                                                        threshold=0.01,  verbose=1) # assign new lr_scheduler
            
            m.begin_run(run, network, loaders)
            for epoch in range(self.num_epochs):
                # Update lr to dataframe
                if self.lr_scheduler: 
                    print(scheduler.get_last_lr())  

                # Train
                network.train() # keep grads
                m.begin_epoch()
                print('\nEpoch {}'.format(epoch+1))
                print('\nTrain:\n')
                for batch_idx, (images, labels) in enumerate(Bar(loaders['train'])):
                    images, labels = images.to(self.device, dtype=torch.float), labels.to(self.device, dtype=torch.float) # labels is a tensor of (512, 128) values if we use MyVgg
                    optimizer.zero_grad()
                    preds = network(images) # returns a dictionary 512 outputs of 128 values (512, 128) if MyVgg is used
                    # Sum all losses from all classifiers if MyVgg is used

                    if isinstance(self.criterion, nn.CrossEntropyLoss):
                        loss = 0
                        for output, target in zip(preds.values(), labels): # comparing (128, 1) vs. (128, 1) vectors 
                            loss += self.criterion(output, target)
                    else:
                        loss = self.criterion(preds, labels)
                    loss.backward()
                    optimizer.step()
                    
                    m.track_loss(loss, 'train')
                    if isinstance(network, MyVgg): m.track_num_correct(preds, labels, 'train')
                                    
                # Validation
                print('\nValid:\n')
                network.eval() # skips dropout and batch_norm 
                with torch.no_grad():
                    for batch_idx, (images, labels) in enumerate(Bar(loaders['valid'])):
                        images, labels = images.to(self.device, dtype=torch.float), labels.to(self.device, dtype=torch.float) # labels is an array of 512 values if we use MyVgg
                        preds = network(images)
                        # Sum all losses from all classifiers if MyVgg is used
                        if isinstance(self.criterion, nn.CrossEntropyLoss):
                            loss = 0
                            for output, target in zip(preds.values(), labels): # comparing (128, 1) vs. (128, 1) vectors 
                                loss += self.criterion(output, target)
                        else:
                            loss = self.criterion(preds, labels)

                        m.track_loss(loss, 'valid')
                        if isinstance(network, MyVgg): m.track_num_correct(preds, labels, 'valid')
                    
                scheduler.step(loss) # update lr_scheduler
                m.end_epoch()
                if m._get_early_stop():
                    break
                
            m.end_run()
        m.save_results('results')

