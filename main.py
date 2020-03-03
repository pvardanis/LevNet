import argparse
import os
from solver import Solver
from helpers import set_seed
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from collections import OrderedDict

set_seed(0)

def get_solver(config):

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

    transform = transforms.Compose([transforms.ToTensor()])

    dataset = datasets.FashionMNIST(
                        root='./data',
                        train=True,
                        download=True,
                        transform=transform)
    
    train_set, valid_set = torch.utils.data.random_split(dataset, [50000, 10000])

    params = OrderedDict(
            lr=[0.001],
            batch_size=[128],
            patience=[1]
    )

    solver = Solver(train_set, valid_set, test_set=None, config=config, params_dict=params)
    return solver

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
   
    # model hyper-parameters (optional)
    parser.add_argument('--image_size', type=int, default=224, help='w x h of input image')
    parser.add_argument('--input_ch', type=int, default=2, help='number of channels of input image')
    parser.add_argument('--output_ch', type=int, default=512, help='number of output nodes')
    
    # training hyper-parameters (optional)
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--num_epochs_decay', type=int, default=70, help='threshold for learning rate decay')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for each pass') 
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for dataloader')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam/SGD') # which model to train/test
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')        # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')      # momentum2 in Adam    
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD') # which model to train/test
    parser.add_argument('--lr_decay', type=float, default=0, help='learning rate decay') # which model to train/test
    parser.add_argument('--early_stopping', action='store_true', default=False, help='use early stopping for training') 
    parser.add_argument('--save_best_model', action='store_true', default=False, help='save best model from each run') 
    
    # misc
    parser.add_argument('--mode', type=str, default='train', help='train/test')
    parser.add_argument('--model_type', type=str, default='Tester', help='Tester/LevNet/VGG-16/VGG-19/Inception-v3') # which model to train/test
    parser.add_argument('--model_path', type=str, default='./models')
    parser.add_argument('--train_path', type=str, default='./dataset/train/')
    parser.add_argument('--valid_path', type=str, default='./dataset/valid/')
    parser.add_argument('--test_path', type=str, default='./dataset/test/')
    parser.add_argument('--result_path', type=str, default='./results/')

    config = parser.parse_args()
