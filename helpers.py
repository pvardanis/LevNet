import global_vars
import numpy as np
import pandas as pd
from collections import OrderedDict
from collections import namedtuple
from itertools import product
import time
from datetime import datetime
import os, shutil, h5py
import random
import torch
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset  # For custom data-sets
import torchvision.transforms as transforms
from PIL import Image
import torchvision
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from models import *

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

    def __getitem__(self, key):
        return getattr(self, key)
    
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
        '''
        Resets epoch_count and closes tensorboard graphs.
        '''
        self.tb['train'].close()
        self.tb['valid'].close()
        self.epoch_count = 0

    def begin_epoch(self):
        '''
        Initializes variables at the start of each epoch.
        '''
        self.epoch_start_time = time.time()
        self.epoch_count += 1
        self.epoch_loss = OrderedDict(train=0, valid=0)
        self.epoch_num_correct = OrderedDict(train=0, valid=0)

    def end_epoch(self):
        '''
        Stores train/valid loss & accuracy, epoch_duration, run_duration, patience counter at after a whole epoch is completed.
        '''
        loss_train = self.epoch_loss['train'] / len(self.loaders['train'].dataset)
        loss_valid = self.epoch_loss['valid'] / len(self.loaders['valid'].dataset)
        
        if isinstance(self.network, MyVgg): # add this if network is MyVgg
            accuracy_train = 100. * self.epoch_num_correct['train'] / len(self.loaders['train'].dataset) # accuracy for each phase classifier, tensor of (512, 1)
            accuracy_valid = 100. * self.epoch_num_correct['valid'] / len(self.loaders['train'].dataset) # accuracy for each phase classifier, tensor of (512, 1)
        

        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time

        if global_vars.tensorboard: # add graphs to SummaryWriter()
            self.tb['train'].add_scalar('Loss', loss_train, self.epoch_count)
            self.tb['valid'].add_scalar('Loss', loss_valid, self.epoch_count)
            if isinstance(self.network, MyVgg):
                self.tb['train'].add_scalar('Accuracy', torch.mean(accuracy_train), self.epoch_count) # track mean accuracy across all classifiers
                self.tb['valid'].add_scalar('Accuracy', torch.mean(accuracy_valid), self.epoch_count)
            
            if not global_vars.colab: # we don't want to store everything in colab, otherwise it crashes
                for name, param in self.network.named_parameters():
                    self.tb['train'].add_histogram(name, param, self.epoch_count)
                    self.tb['train'].add_histogram(f'{name}.grad', param.grad, self.epoch_count)

        results = OrderedDict()
        results["run"] = self.run_count
        results["epoch"] = self.epoch_count
        results["train loss"] = loss_train
        results["valid loss"] = loss_valid
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

        # Clear printed output
        if global_vars.console:
            global_vars.cls() # clear console output
            print(df)            
        else:
            clear_output(wait=True) # update cell output for each epoch
            display(df)
                
    def track_loss(self, loss, data='train'):
        '''
        Tracks loss.

        Inputs
            preds (tensor): predictions
            labels (tensor): labels
            data (str): 'train' or 'valid', to keep track of seperate losses
        '''
        self.epoch_loss[data] += loss.item() * self.loaders[data].batch_size

    def track_num_correct(self, preds, labels, data='train'):
        '''
        Tracks number of correct predictions.

        Inputs
            preds (tensor): predictions
            labels (tensor): labels
            data (str): 'train' or 'valid', to keep track of seperate scores
        '''
        self.epoch_num_correct[data] += self._get_num_correct(preds, labels) # tensor of (512, num_correct) for each classifier

    @torch.no_grad() # only applies to this function
    def _get_num_correct(self, preds, labels):
        '''
        Computes number of correct predicitons.

        Inputs
            preds (tensor): predictions
            labels (tensor): labels
        '''
        preds = preds.data.max(1, dim=1)[1] # take max indices of the (512, 128) tensor across axis=1 (maximum prediction for each phase)
        return np.sum(np.squeeze(preds.eq(labels).data.view_as(preds))).cpu().numpy()

    def _get_early_stop(self):
        '''
        Returns True if patience is overpassed, to break the training loop.
        '''
        return self.early_stopping.early_stop if self.stop_early else False

    def save_results(self, filename):
        '''
        Saves results from each run.

        Inputs
            filename (str): file name to be saved

        '''
        
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
    def __init__(self, patience=1, verbose=False, delta=0, path=None):
        """
        Early stops the training if validation loss doesn't improve after a given patience.
        
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

        Inputs
            val_loss (float): valid loss
            model (obj): model
            save (bool): True if save, else False 
        """
        score = val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score > self.best_score + self.delta:
            # TODO: Fix patience counter.
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

        Inputs
            val_loss (float): valid loss
            model (bool): model
        '''
        torch.save(model.state_dict(), self.path+'/checkpoint.pt')
        self.val_loss_min = val_loss

class CustomDataset(Dataset): # inherits from Dataset
    '''
    Custom Dataset() that reads a .h5 file with data and returns image and target vectors for the DataLoader().
    '''
    def __init__(self, path):
        self.path = path
        self.file = h5py.File(self.path+'/data.h5', 'r')
        self.transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                        [0.229, 0.224, 0.225])])
        self.counter = 0
                                        
    def __getitem__(self, index):
        image = self.file['pos_{}'.format(index)][()]
        image = self.transform(image)
        
        target = torch.from_numpy(self.file['phases_{}'.format(index)][()]).float()   
        target *= 2 * np.pi / 127. # actual value between 0 and 2pi
        
        return image, target

    def __len__(self):  # return number of samples we have
        return 12000#self.num_files

def arrange_images():
    '''
    Move position *.bmp images and phases *.csv files to seperate folders
    '''
    path = './images'
    assert os.path.isdir(path), "ERROR: Data not found!"
    
    root = os.listdir(path)
    path_pos = path + '/pos'
    path_phases = path + '/phases'    

    if not os.path.exists(path_pos):
        os.makedirs(path_pos)
    if not os.path.exists(path_phases):
        os.makedirs(path_phases)

    for image in root:
        if image.startswith('Phase'):
            shutil.move(os.path.join('./images', image), os.path.join(path_phases, image))
        if image.startswith('Position'):
            shutil.move(os.path.join('./images', image), os.path.join(path_pos, image))

    listA = os.listdir(path_pos)
    listA.sort()
    listB = os.listdir(path_phases)
    listB.sort()

def prepare_sets(path='./images', percent=.9):
    '''
    Loads a CustomDataset() from the path and returns a train set and a valid set randomly split with percent=.9 (default).

    Inputs
        path (str): path where images are stored.
        percent (float): percent of train/valid random split.

    Returns
        train_set (tensor): set with training data
        valid_set (tensor): set with valid data 

    '''
    
    dataset = CustomDataset(path)
    percent = int(len(dataset) * percent)
    train_set, valid_set = torch.utils.data.random_split(dataset, [percent, len(dataset) - percent])
    
    return train_set, valid_set 

def create_h5(path='images'):
    '''
    Converts .bpm images with positions and phases to a single h5 file. This makes computations for DataLoader() much faster.

    Inputs
        path: path where images are stored
    '''
    import glob
    # load positions images
    images = os.path.join(path, "pos")
    images = os.path.join(images, '*.bmp')
    list_images = glob.glob(images)
    # load phases images
    targets = os.path.join(path, "phases")
    targets = os.path.join(targets, "*.csv")
    list_targets = glob.glob(targets)
    
    with h5py.File('./images/data.h5', 'w') as hf:
        for i, img in enumerate(list_images):
            # images
            image = Image.open(img)
            image_set = hf.create_dataset(
                    name='pos_'+str(i),
                    data=image,
                    compression="gzip",
                    compression_opts=9)
            
        for i, tgt in enumerate(list_targets):
            # targets
            target = pd.read_csv(tgt, sep=" ", header=None) # load csv
            target = target.values.squeeze()

            # This is for MyVgg
            # target.columns = ['phase']
            # possible_values = [str(value) for value in range(128) if value not in target.phase.values] # phases not included in the dataframe
            # extra = pd.DataFrame({'phase_' + value.zfill(3): [0] * 512 for value in possible_values})
            # target = pd.get_dummies(target, columns=['phase']) # one-hot encoding converts it to a DataFrame of 512 rows and 128 columns (1 column per phase), convert it to numpy
            # target.columns = [colname.split('_')[0] + '_' + colname.split('_')[1].zfill(3) for i, colname in enumerate(target.columns)]
            # target[extra.columns] = extra
            # target = target.reindex(sorted(target.columns), axis=1)
            # target = target.to_numpy()
            target_set = hf.create_dataset(
                    name='phases_'+str(i),
                    data=target,
                    compression="gzip",
                    compression_opts=9)

def test_model(model, dataset):
    ''' 
    Tests model output for an input image.

    Inputs
        model: model to be tested
        dataset: dataset to test on   
    '''
    loader = torch.utils.data.DataLoader(dataset, num_workers=0, batch_size=64, shuffle=True)
    images, labels = next(iter(loader))
    image = images[0].squeeze(dim=0)
    plt.imshow(image)
    plt.show()
    
    print("Input shape: {}".format(images[0].unsqueeze(dim=0).shape))
    output = model(images[0].unsqueeze(dim=0).to('cuda'))
    print(output, output.shape)

# Custom losses
def MSEWrap(output, target):
    ''' 
    Custom loss function that wraps the MSE around 2pi.

    Inputs
        output: predicted phases
        target: true phases   
    '''
    loss = torch.mean(torch.fmod(output - target, 2 * np.pi) ** 2)
    return loss

def Atan(output, target):
    ''' 
    Custom loss function atan2.

    Inputs
        output: predicted phases
        target: true phases   
    '''
    return torch.mean(torch.abs(torch.atan2(torch.sin(target - output), torch.cos(target - output))))

def Cosine(output, target):
    '''
    Custom loss function with 2 losses:

    - loss_1: penalizes the area out of the unit circle 
    - loss_2: 0 if output = target

    Inputs
        output: predicted phases
        target: true phases  
    ''' 

    # Penalize if output is out of the unit circle   
    squares = output ** 2 # (x ^ 2, y ^ 2)
    loss_1 = ((squares[:, ::2] + squares[:, 1::2]) - 1) ** 2 # (x ^ 2 + y ^ 2 - 1) ** 2

    # Compute the second loss, 1 - cos
    loss_2 =  1. - torch.cos(torch.atan2(output[:, 1::2], output[:, ::2]) - target)  
    
    return torch.mean(loss_1 + loss_2)