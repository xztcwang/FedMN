import torch
torch.cuda.current_device()
from torch import nn
import torch.nn.functional as F


class LocalTestFedAvg(object):
    def __init__(self, args_, test_iterator):
        self.args_ = args_
        self.test_iterator = test_iterator
        self.loss_func = nn.CrossEntropyLoss()
    def test(self, net):
        net.eval()
        test_loss = 0
        correct = 0
        batch_loss = []
        batch_idx = 0
        with torch.no_grad():
            for images, labels, _ in self.test_iterator:
                images = images.to(self.args_.device).type(torch.float32)
                labels = labels.to(self.args_.device)
                log_probs = net(images)
                loss = F.cross_entropy(log_probs, labels, reduction='sum')
                test_loss += loss.item()
                y_pred = log_probs.data.max(1, keepdim=True)[1]
                correct += y_pred.eq(labels.data.view_as(y_pred)).long().cpu().sum()
                batch_loss.append(loss.item())
                batch_idx = batch_idx + 1
        test_loss = test_loss / len(self.test_iterator.dataset)
        accuracy = 100.00 * correct / len(self.test_iterator.dataset)
        return accuracy, test_loss