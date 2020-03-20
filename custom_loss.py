# A collection of custom losses 
import torch
import numpy as np

def MSEWrap(output, target):
    ''' 
    Custom loss function that wraps the MSE around 2pi.

    Inputs
        output: predicted phases
        target: true phases   
    '''
    return torch.mean(torch.fmod(output - target, 2 * np.pi) ** 2)

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
    # Compute the first loss
    # Penalize if output is out of the unit circle   
    squares = output ** 2 # (x ^ 2, y ^ 2)
    loss_1 = ((squares[:, ::2] + squares[:, 1::2]) - 1) ** 2 # (x ^ 2 + y ^ 2 - 1) ** 2

    ## If tanh activation function outputs sin(theta), cos(theta), then x = cos(theta), y = sin(theta) (values from -1 to 1)
    ## but everything else remains the same
    # Compute the second loss
    loss_2 =  1. - torch.cos(torch.atan2(output[:, 1::2], output[:, ::2]) - target)  
    
    return torch.mean(loss_1 + loss_2)