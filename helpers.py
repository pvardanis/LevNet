import pandas as pd
from collections import OrderedDict
from collections import namedtuple
from itertools import product
import time
from datetime import datetime
import os

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

from IPython.display import clear_output

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

    def begin_run(self, run, network, loaders):

        self.run_start_time = time.time()
        self.run_params = run
        self.run_count += 1

        self.network = network
        self.loaders = loaders
        self.tb['train'] = SummaryWriter(comment=f'-{run}-train')
        self.tb['valid'] = SummaryWriter(comment=f'-{run}-valid')

        images_train, labels_train = next(iter(self.loaders['train']))
        images_train, labels_train = images_train.cuda(), labels_train.cuda()
        grid_train = torchvision.utils.make_grid(images_train).cuda()

        images_valid, labels_valid = next(iter(self.loaders['valid']))
        images_valid, labels_valid = images_valid.cuda(), labels_valid.cuda()
        grid_valid = torchvision.utils.make_grid(images_valid).cuda()

        self.tb['train'].add_image('images_train', grid_train) # not necessarily needed
        self.tb['valid'].add_image('images_valid', grid_valid) # not necessarily needed
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

        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time

        loss_train = self.epoch_loss['train'] / len(self.loaders['train'].dataset)
        accuracy_train = self.epoch_num_correct['train'] / len(self.loaders['train'].dataset)

        loss_valid = self.epoch_loss['valid'] / len(self.loaders['valid'].dataset)
        accuracy_valid = self.epoch_num_correct['valid'] / len(self.loaders['valid'].dataset)

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

        for k, v in self.run_params._asdict().items(): results[k] = v
        self.run_data.append(results)

        print('almost ready')
        df = pd.DataFrame.from_dict(self.run_data)
        clear_output(wait=True) # update cell output for each epoch
        display(df)

    def track_loss(self, loss, data='train'):
        self.epoch_loss[data] += loss.item() * self.loaders[data].batch_size

    def track_num_correct(self, preds, labels, data='train'):
        self.epoch_num_correct[data] += self._get_num_correct(preds, labels)

    @torch.no_grad()
    def _get_num_correct(self, preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()

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



