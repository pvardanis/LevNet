import numpy as np
import pandas as pd
from collections import OrderedDict
from collections import namedtuple
from itertools import product
import time
from datetime import datetime
import os
import random
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

from IPython.display import clear_output

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

class RunBuilder():
    '''
    Returns

    List of different runs for all combination of params.
    '''
    @staticmethod
    def get_runs(params):

        Run = namedtuple('Run', params.keys()) # subclass Run() tuple with parameter keys as items

        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs

class RunManager():
    def __init__(self):

        self.epoch_count = 0 # it could've been even more abstract than this (i.e. define Epoch())
        self.epoch_loss = 0
        self.epoch_num_correct = 0
        self.epoch_start_time = None
        
        self.run_params = None # runs returned from RunBuilder()
        self.run_count = 0
        self.run_data = [] # parameter values and results of each epoch
        self.run_start_time = None

        self.network = None # model for each run
        self.loaders = None # dataloaders (train & validation) that's being used for each run
        self.tb = OrderedDict(train=None, valid=None) # tensorboard instance

    def begin_run(self, run, network, loaders, stop_early=True, save_best_model=False):

        self.run_start_time = time.time()
        self.run_params = run
        self.run_count += 1

        self.network = network
        self.loaders = loaders
        self.tb['train'] = SummaryWriter(comment=f'-{run}-train')
        self.tb['valid'] = SummaryWriter(comment=f'-{run}-valid')

        if stop_early:
            assert run.patience
            self.stop_early = True
            self.early_stopping = EarlyStopping(patience=run.patience, path=self.tb['valid'].get_logdir()) # initialize early stopping and pass the path to save the best model
        else:
            self.stop_early = False

        self.save_best_model = save_best_model

        images_train, labels_train = next(iter(self.loaders['train']))
        images_train, labels_train = images_train.cuda(), labels_train.cuda()
        grid_train = torchvision.utils.make_grid(images_train).cuda()

        images_valid, labels_valid = next(iter(self.loaders['valid']))
        images_valid, labels_valid = images_valid.cuda(), labels_valid.cuda()
        grid_valid = torchvision.utils.make_grid(images_valid).cuda()

        self.tb['train'].add_image('images_train', grid_train) # not necessarily needed
        self.tb['valid'].add_image('images_valid', grid_valid) 
        self.tb['train'].add_graph(self.network, images_train)

    def end_run(self):

        self.tb['train'].close()
        self.tb['valid'].close()
        self.epoch_count = 0

    def begin_epoch(self):

        self.epoch_start_time = time.time()
        self.epoch_count += 1
        self.epoch_loss = OrderedDict(train=0, valid=0)
        self.epoch_num_correct = OrderedDict(train=0, valid=0)

    def end_epoch(self):
        
        loss_train = self.epoch_loss['train'] / len(self.loaders['train'].dataset)
        accuracy_train = self.epoch_num_correct['train'] / len(self.loaders['train'].dataset)

        loss_valid = self.epoch_loss['valid'] / len(self.loaders['valid'].dataset)
        accuracy_valid = self.epoch_num_correct['valid'] / len(self.loaders['valid'].dataset)
        
        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time

        self.tb['train'].add_scalar('Loss', loss_train, self.epoch_count)
        self.tb['train'].add_scalar('Accuracy', accuracy_train, self.epoch_count)
        self.tb['valid'].add_scalar('Loss', loss_valid, self.epoch_count)
        self.tb['valid'].add_scalar('Accuracy', accuracy_valid, self.epoch_count)

        for name, param in self.network.named_parameters():
            self.tb['train'].add_histogram(name, param, self.epoch_count)
            self.tb['train'].add_histogram(f'{name}.grad', param.grad, self.epoch_count)

        results = OrderedDict()
        results["run"] = self.run_count
        results["epoch"] = self.epoch_count
        results["train loss"] = loss_train
        results["train accuracy"] = accuracy_train
        results["valid loss"] = loss_valid
        results["valid accuracy"] = accuracy_valid 
        results["epoch duration"] = epoch_duration
        results["run duration"] = run_duration

        for k, v in self.run_params._asdict().items(): 
            if k != 'patience': # no need to add patience in columns
                results[k] = v
        self.run_data.append(results)

        df = pd.DataFrame.from_dict(self.run_data)
        clear_output(wait=True) # update cell output for each epoch
        display(df)

        # always call early_stopping at the end of each epoch
        self.early_stopping(loss_valid, self.network, save=self.save_best_model)
                
    def track_loss(self, loss, data='train'):
        self.epoch_loss[data] += loss.item() * self.loaders[data].batch_size

    def track_num_correct(self, preds, labels, data='train'):
        self.epoch_num_correct[data] += self._get_num_correct(preds, labels)

    @torch.no_grad()
    def _get_num_correct(self, preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()

    def _get_early_stopping(self):
        return self.early_stopping.early_stop if self.stop_early else False

    def save_results(self, filename):
        
        # setup filename and directory
        current_dir = os.getcwd() # get current working directory
        datetime_ = datetime.now().strftime('%d-%m-%Y_%H-%M-%S') # add date and time to filename
        file_path = current_dir + f'/results/{filename}_{datetime_}.pkl' # path to save 
        directory = os.path.dirname(file_path)
        
        # if no folder named 'results' then create one
        try:
            os.stat(directory)
        except:
            os.mkdir(directory)       

        pd.DataFrame.from_dict(self.run_data).to_pickle(file_path) # save to pickle

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience=1, verbose=False, delta=0, path=None):
        """
        Inputs
        
            patience (int): How long to wait after last time validation loss improved (default=1).
            verbose (bool): If True, prints a message for each validation loss improvement (default=False).
            delta (float): Minimum change in the monitored quantity to qualify as an improvement (default=0.).
            path (str): Path to save the checkpoint.
        
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path 

    def __call__(self, val_loss, model, save=True):
        """
        Every time an instance is called, the state is updated.
        """
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if save: self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''
        Saves model when validation loss decreases.
        '''
        torch.save(model.state_dict(), self.path+'/checkpoint.pt')
        self.val_loss_min = val_loss

