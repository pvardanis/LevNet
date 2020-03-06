import global_vars
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
import matplotlib.pyplot as plt

from IPython.display import display
from IPython.display import clear_output

pd.set_option('display.max_columns',1000)
pd.set_option('display.max_rows',1000)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed) # for kernel restarts

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

class RunManager(object):
    def __init__(self, save_best_model, stop_early):
        self.save_best_model = save_best_model
        self.stop_early = stop_early
        
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

    def begin_run(self, run, network, loaders):
        
        self.early_stopping = None
        self.run_start_time = time.time()
        self.run_params = run
        self.run_count += 1

        self.network = network
        self.loaders = loaders
        self.tb['train'] = SummaryWriter(comment=f'-{run}-train')
        self.tb['valid'] = SummaryWriter(comment=f'-{run}-valid')

        if self.stop_early: 
            assert run.patience, "ERROR: You forgot to add patience."
            self.early_stopping = EarlyStopping(patience=run.patience, path=self.tb['valid'].get_logdir()) # initialize early stopping and pass the path to save the best model
        
        if global_vars.tensorboard: # add graph and images to SummaryWriter()
            images_train, labels_train = next(iter(self.loaders['train']))
            images_train, labels_train = images_train.cuda(), labels_train.cuda()
            grid_train = torchvision.utils.make_grid(images_train).cuda()

            images_valid, labels_valid = next(iter(self.loaders['valid']))
            images_valid, labels_valid = images_valid.cuda(), labels_valid.cuda()
            grid_valid = torchvision.utils.make_grid(images_valid).cuda()

            if not global_vars.colab: # we don't want to store everything in colab, otherwise it crashes
                self.tb['train'].add_image('images_train', grid_train) 
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

        if global_vars.tensorboard: # add graphs to SummaryWriter()
            self.tb['train'].add_scalar('Loss', loss_train, self.epoch_count)
            self.tb['train'].add_scalar('Accuracy', accuracy_train, self.epoch_count)
            self.tb['valid'].add_scalar('Loss', loss_valid, self.epoch_count)
            self.tb['valid'].add_scalar('Accuracy', accuracy_valid, self.epoch_count)

            if not global_vars.colab: # we don't want to store everything in colab, otherwise it crashes
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

        # always call early_stopping at the end of each epoch
        if self.stop_early:
            self.early_stopping(loss_valid, self.network, save=self.save_best_model)
            results["patience counter"] = self.early_stopping.counter
            
        for k, v in self.run_params._asdict().items(): 
            if k != 'patience': # no need to add patience in columns
                results[k] = v

        self.run_data.append(results)

        df = pd.DataFrame.from_dict(self.run_data)

        if global_vars.console:
            global_vars.cls() # clear console output
            print(df)            
        else:
            clear_output(wait=True) # update cell output for each epoch
            display(df)
                
    def track_loss(self, loss, data='train'):
        self.epoch_loss[data] += loss.item() * self.loaders[data].batch_size

    def track_num_correct(self, preds, labels, data='train'):
        self.epoch_num_correct[data] += self._get_num_correct(preds, labels)

    @torch.no_grad() # only applies to this function
    def _get_num_correct(self, preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()

    def _get_early_stop(self):
        return self.early_stopping.early_stop if self.stop_early else False

    def save_results(self, filename):
        
        # setup filename and directory
        current_dir = os.getcwd() # get current working directory
        datetime_ = datetime.now().strftime('%d-%m-%Y_%H-%M-%S') # add date and time to filename
        file_path = current_dir + f'/results/{filename}_{datetime_}.pkl' # path to save 
        directory = os.path.dirname(file_path)
        
        # if no folder named 'results' then create one
        os.makedirs(directory)
        
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

    def __call__(self, val_loss, model, save):
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

def test_model(model, dataset):
    ''' 
    Tests model output for an input image.
    '''
    loader = torch.utils.data.DataLoader(dataset, num_workers=0, batch_size=64, shuffle=True)
    images, labels = next(iter(loader))
    image = images[0].squeeze(dim=0)
    plt.imshow(image)
    plt.show()
    
    print("Input shape: {}".format(images[0].unsqueeze(dim=0).shape))
    output = model(images[0].unsqueeze(dim=0).to('cuda'))
    print(output, output.shape)