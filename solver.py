from collections import OrderedDict
import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision.utils 
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision.models
from helpers import *
from models import *

# Global variables
set_seed(0)

class Solver(object): 
    def __init__(self, train_set, valid_set, test_set, config=None):
        super().__init__()

        # Global settings
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Always same
        self.criterion = nn.NLLLoss().to(self.device)
        self.optimizers = OrderedDict(adam=optim.Adam, sgd=optim.SGD)

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
        if self.stop_early: self.params['patience'] = config.patience 
        self.save_best_model = config.save_best_model

        print(self.params)
                
    def build_model(self):

        if self.model_type in ['vgg-16', 'vgg-19']:
            
            assert self.image_size == 224, "ERROR: Wrong image size."
            model = torchvision.models.vgg16(pretrained=True) if self.model_type == 'vgg-16' else torchvision.models.vgg19(pretrained=True)
            if self.input_ch != 3:
                first_conv_layer = [nn.Conv2d(self.input_ch, 3, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)]
                first_conv_layer.extend(list(model.features))  
                model.features= nn.Sequential(*first_conv_layer)  

            model.classifier[-1] = nn.Linear(4096, 1000)
            model.classifier.add_module('7', nn.ReLU())
            model.classifier.add_module('8', nn.Dropout(p=0.5, inplace=False))
            model.classifier.add_module('9', nn.Linear(1000, self.output_ch))
            model.classifier.add_module('10', nn.LogSoftmax(dim=1))
            
            for param in model.features[1:].parameters(): # disable grad for trained layers
                param.requires_grad = False

            # make sure that params are freezed the correct way            
            # for param in model.features[1:].parameters(): # disable grad for trained layers
            #     if param.requires_grad == True: print("ERROR: params not freezed in features!")
            
            # for param in model.features[0].parameters():
            #     if param.requires_grad == False: print("ERROR: params freezed in first_conv_layer")

            # for param in model.classifier.parameters():
            #     if param.requires_grad == False: print("ERROR: params freezed in classifier")

            # x = torch.randn(1, 1, 224, 224) # (256, 256, 3)
            # output = model(x)
            # print(output.shape)
            return model.to(self.device)

        elif self.model_type == 'tester': return Tester().to(self.device)
        elif self.model_type == 'levnet': return LevNet().to(self.device)

    def train(self):
        m = RunManager(self.save_best_model, self.stop_early)
        for run in RunBuilder.get_runs(self.params):
            network = self.build_model() # this returns a new instance of the network .to(self.device)
            train_loader = torch.utils.data.DataLoader(self.train_set, num_workers=self.num_workers, batch_size=run.batch_size, shuffle=True)
            valid_loader = torch.utils.data.DataLoader(self.valid_set, num_workers=self.num_workers, batch_size=run.batch_size, shuffle=True)
            loaders = OrderedDict(train=train_loader, valid=valid_loader)
            
            if run.optimizer == 'adam':
                optimizer = self.optimizers[run.optimizer](network.parameters(), lr=run.lr, betas=self.betas)
            elif run.optimizer == 'sgd':
                optimizer = self.optimizers[run.optimizer](network.parameters(), lr=run.lr, momentum=self.momentum)

            m.begin_run(run, network, loaders)
            for epoch in range(self.num_epochs):
                # Train
                network.train() # keep grads
                m.begin_epoch()
                for batch_idx, (images, labels) in enumerate(loaders['train']):
                    images, labels = images.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    preds = network(images)
                    loss = self.criterion(preds, labels)
                    loss.backward()
                    optimizer.step()
                    
                    m.track_loss(loss, 'train')
                    m.track_num_correct(preds, labels, 'train')
                                    
                # Validation
                network.eval() # skips dropout and batch_norm 
                for batch_idx, (images, labels) in enumerate(loaders['valid']):
                    images, labels = images.to(self.device), labels.to(self.device)
                    preds = network(images)
                    loss = self.criterion(preds, labels)

                    m.track_loss(loss, 'valid')
                    m.track_num_correct(preds, labels, 'valid')
                    
                m.end_epoch()
                if m._get_early_stop():
                    break
                
            m.end_run()
        m.save_results('results')

    