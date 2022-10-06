import torch
torch.cuda.current_device()
from torch import nn
import torch.nn.functional as F
import copy

class LocalTest(object):
    def __init__(self, args_, test_iterator, policy_selection=None):
        self.args_ = args_
        self.test_iterator = test_iterator
        self.policy_selection = policy_selection
        self.loss_func = nn.CrossEntropyLoss()

    def test(self, net_g):
        batch_loss = []
        test_loss = 0
        correct = 0
        batch_idx = 0
        with torch.no_grad():
            for images, labels, _ in self.test_iterator:
                images = images.to(self.args_.device).type(torch.float32)
                labels = labels.to(self.args_.device)
                train_flag = False
                log_probs, policy_out = net_g(images, None, None, None, train_flag, self.policy_selection)
                loss = F.cross_entropy(log_probs, labels, reduction='sum')
                test_loss += loss.item()
                y_pred = log_probs.data.max(1, keepdim=True)[1]
                correct += y_pred.eq(labels.data.view_as(y_pred)).long().cpu().sum()
                batch_loss.append(loss.item())
                batch_idx = batch_idx + 1
        test_loss = test_loss / len(self.test_iterator.dataset)
        accuracy = 100.00 * correct / len(self.test_iterator.dataset)
        return accuracy, test_loss