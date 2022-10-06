import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as tvmodels
from collections import OrderedDict
from models.mobilenet import get_mobilenet
from models.lenet5 import LenetCNN

class PolicyNet(nn.Module):
    def __init__(self,
                 args_,
                 policy_hdim_x,
                 policy_hdim_y,
                 policy_outdim,
                 hard):
        super(PolicyNet, self).__init__()
        self.args_ = args_
        self.policy_hdim_x = policy_hdim_x
        self.policy_hdim_y = policy_hdim_y
        self.policy_outdim = policy_outdim
        self.hard = hard

        if self.args_.dataset=='cifar10' or self.args_.dataset=='cifar100':
            self.policy_model = get_mobilenet(self.policy_hdim_x)
            dataset_name = self.args_.dataset
            w_pretrain = torch.load('./save/'+dataset_name+'_pre_model.pt',
                                    map_location='cuda:' + str(self.args_.gpu))
            w_pretrain['mobilenetout.1.weight'] = w_pretrain.pop('classifier.1.weight')
            w_pretrain['mobilenetout.1.bias'] = w_pretrain.pop('classifier.1.bias')
            policynet_dict = self.policy_model.state_dict()
            policyparam_dict = {k: v for k, v in w_pretrain.items() if k in policynet_dict}
            policynet_dict.update(policyparam_dict)

        if self.args_.dataset == 'emnist' or self.args_.dataset == 'femnist':
            self.policy_model = LenetCNN(self.policy_hdim_x)
            dataset_name = self.args_.dataset
            w_pretrain = torch.load('./save/' + dataset_name + '_pre_model.pt',
                                    map_location='cuda:' + str(self.args_.gpu))
            w_pretrain['lenetout.1.weight'] = w_pretrain.pop('output.weight')
            w_pretrain['lenetout.1.bias'] = w_pretrain.pop('output.bias')
            policynet_dict = self.policy_model.state_dict()
            policyparam_dict = {k: v for k, v in w_pretrain.items() if k in policynet_dict}
            policynet_dict.update(policyparam_dict)

        self.policy_model.load_state_dict(policynet_dict)
        self.encoder_y = nn.Linear(self.args_.n_class, self.policy_hdim_y)
        self.hdim_xy = self.policy_hdim_x + self.policy_hdim_y
        self.encoder_xy = nn.Linear(self.hdim_xy, self.policy_outdim)
        self.relu = nn.ReLU()

    def forward(self, x, y, temp):
        z1 = self.policy_model(x)
        z2 = self.encoder_y(y)
        z = torch.cat([z1, z2], 1)
        z = F.normalize(z, p=2, dim=1)
        z = self.relu(z)
        z = self.encoder_xy(z)
        z_mean = torch.mean(z, dim=0)
        z_mean_gumbel = []
        for dim in range(self.policy_outdim):
            bit = F.gumbel_softmax(torch.Tensor([-z_mean[dim].item(), z_mean[dim].item()]), tau=temp, hard=self.hard)
            z_mean_gumbel.append(bit[1].unsqueeze(0))
        z_mean_gumbel_tensor = torch.cat(z_mean_gumbel)
        z_mean_gumbel_tensor = torch.nn.functional.softmax(z_mean_gumbel_tensor,dim=0)
        return z_mean_gumbel_tensor

