import torch
import sys

def one_hot(label, depth):
    y = torch.zeros(label.size(0), depth)
    idx = torch.LongTensor(label).view(-1, 1)
    y.scatter_(dim = 1, index = idx, value = 1)
    return y

