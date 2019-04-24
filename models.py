import torch.nn as nn
import torch.nn.functional as F
import pdb
import torch
import sys

from torch.autograd import Variable
#from datasets import ImageDataset
#from genABdataset import create_dataset


class Discriminator(nn.Module):
    def __init__(self,input_nc):
        self.input_nc = input_nc
        super(Discriminator,self).__init__()

        self.cnn = nn.Sequential(
            nn.Linear(self.input_nc ,128),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Linear(128, 300),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Dropout(0.4),
            nn.Linear(300, 128),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, 1),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Dropout(0.4),
            nn.Softmax()
        )


    def forward(self, x):
        x = self.cnn(x)
        x = x.view(-1)
        return x


class Generator(nn.Module):
    def __init__(self,input_nc, output_nc):
        self.input_nc = input_nc
        self.output_nc = output_nc
        super(Generator,self).__init__()
        self.cnn = nn.Sequential(
            nn.Linear(self.input_nc,self.output_nc),
            nn.LeakyReLU(0.3, inplace=True),
            #nn.Linear(128, 256),
            #nn.LeakyReLU(0.3, inplace=True),
            #nn.Linear(256, 128),
            #nn.ReLU(0.3, inplace=True),
            #nn.Linear(300, self.output_nc)
        )


    def forward(self, x):
        x = self.cnn(x)
        return x



def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
