import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple, OrderedDict
import torchvision.models
import torch
import torch.nn as nn

class Tester(nn.Module):
    '''
    Use this model only to test the workflow.
    '''
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)

        return x

class LevNet(nn.Module):
    '''
    Simple implementation of LevNet.
    '''
    def __init__(self):
        super().__init__()

        # Conv layers
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1) 
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1) 
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1) 
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1) 
        self.conv5 = nn.Conv2d(256, 512, 3, padding=1) 

        ## Maxpooling
        self.pool = nn.MaxPool2d(2,2)

        ## Fully connected
        self.fc1 = nn.Linear(7 * 7 * 512, 2048) 
        self.fc2 = nn.Linear(2048, 1024)
        self.dropout = nn.Dropout(0.2) 

    def forward(self, x):
        ## Define forward behavior
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        # flatten image
        x = x.view(-1, 7 * 7 * 512)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.tanh(self.fc2(x))
        x = self.dropout(x)

        return x

class MyVgg(nn.Module):
    def __init__(self, version='16', batch_norm=True, pretrained=True):
        super().__init__()
        
        vgg = namedtuple('vgg', ['version', 'batch_norm', 'pretrained'])
        combinations = {vgg('16', True, True): torchvision.models.vgg16_bn(pretrained=True),
                        vgg('16', True, False): torchvision.models.vgg16_bn(pretrained=False),
                        vgg('16', False, True): torchvision.models.vgg16(pretrained=True),
                        vgg('16', False, False): torchvision.models.vgg16(pretrained=False),
                        vgg('19', True, True): torchvision.models.vgg19_bn(pretrained=True),
                        vgg('19', True, False): torchvision.models.vgg19_bn(pretrained=False),
                        vgg('19', False, True): torchvision.models.vgg19(pretrained=True),
                        vgg('19', False, False): torchvision.models.vgg19(pretrained=False)}
        self.model = combinations[vgg(version, batch_norm, pretrained)]

        # Remove the last fc layer
        self.model.classifier = nn.Sequential(*list(self.model.classifier.children())[:-1])

        # Include seperate classifiers for each phase
        self.pc = OrderedDict() # pc: phase classifiers, 512 in total
        for classifier in range(512):
            self.pc['PC_{}'.format(classifier)] = nn.Sequential(nn.Linear(4096, 128, bias=True)) # no need for nn.Softmax(), it is encapsulated nn.BCEWithLogitsLoss()

    # Set your own forward pass
    def forward(self, img, extra_info=None):
        x = x.view(x.size(0), -1)
        pre_split = self.model(x) # before splitting to different classifiers, take the output from vgg

        outputs = OrderedDict()
        for pc in self.model.pc().values(): # iterate through all 512 classifiers
            outputs['x{}'.format] = pc(pre_split) # pass network output to all 512 classifiers

        return outputs # dictionary with the outputs from the 512 classifiers