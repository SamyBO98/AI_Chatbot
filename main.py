import os
import json
import random
import nltk
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

#Run once
#nltk.download('punkt_tab')

class ChatBot(nn.Module):
    
    #output_size = number of intentions
    def __init__(self, input_size, output_size):
        super(ChatBot, self).__init__()

        #128 neurons
        #represent string as numbers
        self.fc1 = nn.Linear(input_size,128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        #Break linearity
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        #learn a lot of patterns
        #input to 128 values with Linear function then ReLu (negative value to 0)
        x = self.relu(self.fc1(x))
        #no overfitting
        x = self.dropout(x)
        #Compress to keep essential 128 -> 64
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        #Final decision
        x = self.fc3(x)



