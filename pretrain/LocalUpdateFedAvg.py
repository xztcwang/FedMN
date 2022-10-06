import torch
torch.cuda.current_device()
from torch import nn

class LocalUpdateFedAvg(object):
    def __init__(self, args_, train_iterator):
        self.args_ = args_
        self.train_iterator = train_iterator
        self.loss_func = nn.CrossEntropyLoss()

    def train(self, net, learning_rate):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(),
                                    lr=learning_rate,
                                    momentum=0.9,
                                    weight_decay=5e-4)
        epoch_loss = []
        for iter in range(self.args_.pretrain_local_epochs):
            batch_loss = []
            batch_idx = 0
            n_samples = 0
            for images, labels, _ in self.train_iterator:
                images = images.to(self.args_.device)
                labels = labels.to(self.args_.device)
                n_samples += labels.size(0)
                optimizer.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
                batch_idx = batch_idx + 1
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
