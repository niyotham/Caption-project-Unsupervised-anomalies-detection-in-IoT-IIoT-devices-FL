from pyexpat import features
from random import sample
import torch
from sklearn.model_selection import train_test_split
from torch import nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy  as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler 
from torch.utils.data import Subset,random_split,WeightedRandomSampler
from typing import Tuple, Dict
from sklearn.preprocessing import LabelEncoder
import warnings

class autoencoder(nn.Module):

      def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
                          nn.Linear(66, 256),nn.Dropout(0.5),
                          nn.ReLU(),
                          nn.Linear(256, 128),nn.Dropout(0.5),
                          nn.ReLU(),
                          nn.Linear(128, 64),nn.Dropout(0.5),
                          nn.ReLU(),
                          nn.Linear(64, 32),nn.Dropout(0.5),
                          nn.ReLU(),
                          nn.Linear(32, 16),nn.Dropout(0.5),
                          nn.ReLU(),
                          nn.Linear(16, 8))

        self.decoder = nn.Sequential(
                        nn.Linear(8, 16),nn.Dropout(0.5),
                        nn.ReLU(),
                        nn.Linear(16, 32),nn.Dropout(0.5),
                        nn.ReLU(),
                        nn.Linear(32, 64),nn.Dropout(0.5),
                        nn.ReLU(),
                        nn.Linear(64,128),nn.Dropout(0.5),
                        nn.ReLU(),
                        nn.Linear(128, 256),nn.Dropout(0.5),
                        nn.Linear(256,66),
                        nn.Tanh())

      def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class AE(nn.Module):
    
      def __init__(self):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
                      nn.Linear(66, 128),nn.BatchNorm1d(128),
                      nn.LeakyReLU(0.01),
                      nn.Linear(128, 64),nn.BatchNorm1d(64),
                      nn.LeakyReLU(0.01),
                      nn.Linear(64, 32))

        self.decoder = nn.Sequential(
                      nn.Linear(32, 64),nn.BatchNorm1d(64),
                      nn.LeakyReLU(0.01),
                      nn.Linear(64, 128),nn.BatchNorm1d(128),
                      nn.LeakyReLU(0.01),
                      nn.Linear(128, 66),
                      nn.Sigmoid())

      def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class D_autoencoder(nn.Module):
    
      def __init__(self):
        super(D_autoencoder, self).__init__()
        self.encoder = nn.Sequential(
                          nn.Linear(66, 128),nn.Dropout(0.2),
                          nn.ReLU(),
                          nn.Linear(128, 64),nn.Dropout(0.2),
                          nn.ReLU(),
                          nn.Linear(64, 32),nn.Dropout(0.2),
                          nn.ReLU(),
                          nn.Linear(32, 16))
                          

        self.decoder = nn.Sequential(
                        nn.Linear(16, 32),nn.Dropout(0.2),
                        nn.ReLU(),
                        nn.Linear(32, 64),nn.Dropout(0.2),
                        nn.ReLU(),
                        nn.Linear(64,128),nn.Dropout(0.2),
                        nn.ReLU(),
                        nn.Linear(128, 66),
                        nn.Tanh())

      def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
