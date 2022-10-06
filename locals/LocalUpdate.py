import torch
torch.cuda.current_device()
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import copy
from utils.label_utils import one_hot
import torch.nn.utils as torch_utils
# temperature
temp_min = 0.5
ANNEAL_RATE = 0.00003


class LocalUpdate(object):
    def __init__(self, args_, train_iterator, whole_train_iterator):
        self.args_ = args_
        self.train_iterator = train_iterator
        self.whole_train_iterator = whole_train_iterator
        self.client_num =len(self.train_iterator)
        self.loss_func = nn.CrossEntropyLoss()

    def train(self, net, temp, learning_rate):
        optimizer = torch.optim.SGD(net.parameters(),
                                    lr=learning_rate,
                                    momentum=0.9,
                                    weight_decay=5e-4)
        # optimizer = torch.optim.SGD(net.parameters(),
        #                             lr=learning_rate)
        epoch_loss = []
        for iter in range(self.args_.fed_local_epochs):
            whole_data = torch.Tensor([])
            whole_labels = torch.Tensor([])
            for whole_images, whole_labels, _ in self.whole_train_iterator:
                whole_data = \
                    whole_images.to(self.args_.device).type(torch.float32)
                whole_labels = \
                    one_hot(whole_labels, self.args_.n_class).to(self.args_.device)
            batch_loss = []
            batch_idx = 0
            for images, labels, _ in self.train_iterator:
                images = images.to(self.args_.device).type(torch.float32)
                labels = labels.to(self.args_.device)
                optimizer.zero_grad()
                if batch_idx % 1 == 1:
                    temp = np.maximum(temp * np.exp(-ANNEAL_RATE * batch_idx), temp_min)
                train_flag = True
                log_probs, policy_out = net(images, whole_data, whole_labels, temp, train_flag)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
                batch_idx = batch_idx + 1
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), policy_out



