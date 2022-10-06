import torch
from torch import nn
import torch.nn.functional as F

class MLP_block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MLP_block, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.fc1 = nn.Linear(self.in_channel, self.out_channel)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return x

class Linear_mapping(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Linear_mapping, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.fc1 = nn.Linear(self.in_channel, self.out_channel)
    def forward(self, x):
        x = self.fc1(x)
        return x