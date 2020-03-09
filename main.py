import global_vars
import argparse
import os
from solver import Solver
from helpers import set_seed
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from collections import OrderedDict
import matplotlib.pyplot as plt
from helpers import test_model, CustomDataset, prepare_sets
import glob

set_seed(0)

def main(config):
    # train_loader = get_loader(image_path=config.train_path,
    #                         image_size=config.image_size,
    #                         batch_size=config.batch_size,
    #                         num_workers=config.num_workers,
    #                         mode='train',
    #                         augmentation_prob=config.augmentation_prob)
    # valid_loader = get_loader(image_path=config.valid_path,
    #                         image_size=config.image_size,
    #                         batch_size=config.batch_size,
    #                         num_workers=config.num_workers,
    #                         mode='valid',
    #                         augmentation_prob=0.)
    # test_loader = get_loader(image_path=config.test_path,
    #                         image_size=config.image_size,
    #                         batch_size=config.batch_size,
    #                         num_workers=config.num_workers,
    #                         mode='test',
    #                         augmentation_prob=0.)
    # set global_vars
    global_vars.console = config.console
    global_vars.tensorboard = config.tensorboard
    global_vars.colab = config.colab
    global_vars.tpu = config.tpu

    # making sure that config parameters are ok
    if config.model_type not in ['tester', 'levnet', 'vgg-16', 'vgg-16-bn']:
        print('ERROR: model_type should be selected from the available models: Tester/LevNet/VGG-16/VGG-19/Inception-v3.')
        return

    if not set(config.optimizers).issubset(['adam', 'sgd']):
        print('ERROR: optimizer should be selected from the available optimizers: Adam/SGD.')
        return

    if config.model_type == 'tester': 
        transform = transforms.Compose([#transforms.RandomRotation(30),
                                    transforms.Resize(224), # comment this if using Tester
                                    # transforms.RandomHorizontalFlip(),
                                    # transforms.RandomVerticalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5], [0.5]),
                                    ])

        dataset = datasets.FashionMNIST(
                            root='./data',
                            train=True,
                            download=True,
                            transform=transform)

        train_set, valid_set = torch.utils.data.random_split(dataset, [50000, 10000])
    else:
        train_set, valid_set = prepare_sets(config.data_path, percent=.9)

    solver = Solver(train_set, valid_set, test_set=None, config=config)
    if config.mode == 'train':
        solver.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
   
    # model hyper-parameters (optional)
    parser.add_argument('--image_size', type=int, default=224, help='w x h of the input image.')
    parser.add_argument('--input_ch', type=int, default=3, help='Number of channels of the input image. ')
    parser.add_argument('--output_ch', type=int, default=512, help='Number of output nodes.')
    
    # training hyper-parameters (optional)
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs.')
    parser.add_argument('--num_epochs_decay', type=int, default=70, help='Threshold for learning rate decay.')
    parser.add_argument('--batch_size', nargs='+', type=int, default=1, help='Batch size for each pass.') 
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for dataloader.')
    parser.add_argument('--lr', nargs='+', type=float, default=0.0002, help='Learning rate.')
    parser.add_argument('--lr_decay', type=float, default=0, help='Learning rate decay.') 
    parser.add_argument('--optimizers', nargs='+', type=str.lower, default=['adam'], help='List of optimizers: Adam/SGD.') 
    parser.add_argument('--beta1', type=float, default=0.5, help='Beta1 for Adam optimizer.')        
    parser.add_argument('--beta2', type=float, default=0.999, help='Beta2 for Adam optimizer.')          
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD.') 
    parser.add_argument('--early_stopping', action='store_true', default=False, help='Use early stopping for training.') 
    parser.add_argument('--patience', nargs='+', type=int, default=[20], help='Patience for early stopping.')
    parser.add_argument('--save_best_model', action='store_true', default=False, help='Save best model from each run.') 
    
    # datasets
    parser.add_argument('--mode', type=str.lower, default='train', help='train/test')
    parser.add_argument('--model_type', type=str.lower, default='tester', help='Tester/LevNet/VGG-16/VGG-19/Inception-v3') 
    parser.add_argument('--model_path', type=str, default='./models')
    parser.add_argument('--data_path', type=str, default='./images')
    parser.add_argument('--result_path', type=str, default='./results')

    # global_vars
    parser.add_argument('--console', action='store_true', default=False, help='True if using command prompt, False if using Jupyter.')
    parser.add_argument('--tensorboard', action='store_true', default=False, help='True if using tensorboard, else False.')
    parser.add_argument('--colab', action='store_true', default=False, help='True if using colab, else False.')
    parser.add_argument('--tpu', action='store_true', default=False, help='True if using TPU in colab, else False.')

    config = parser.parse_args()
    main(config)
