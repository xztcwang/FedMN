import torch
from torch import nn
from models.mobilenet import get_mobilenet
from models.lenet5 import LenetCNN
from models.mlps import MLP_block, Linear_mapping
import torch.nn.functional as F
############################################
#              Modular Network
############################################

#############################
#          Layer 0
#############################
class Layer_In_Modular(nn.Module):
    def __init__(self, args_, units_num_list, modular_hdims, dropout, layer_idx):
        super(Layer_In_Modular, self).__init__()
        self.args_ = args_
        self.units_num_list = units_num_list
        self.modular_hdims = modular_hdims
        self.dropout = dropout
        self.layer_idx = layer_idx
        if args_.dataset == 'cifar10' or args_.dataset == 'cifar100':
            encoder_block = get_mobilenet(self.modular_hdims[self.layer_idx])
            dataset_name = self.args_.dataset
            w_pretrain = torch.load('./save/' + dataset_name + '_pre_model.pt',
                                    map_location='cuda:' + str(self.args_.gpu))
            w_pretrain['mobilenetout.1.weight'] = w_pretrain.pop('classifier.1.weight')
            w_pretrain['mobilenetout.1.bias'] = w_pretrain.pop('classifier.1.bias')
        if args_.dataset == 'emnist' or args_.dataset == 'femnist':
            encoder_block = LenetCNN(self.modular_hdims[self.layer_idx])
            dataset_name = self.args_.dataset
            w_pretrain = torch.load('./save/' + dataset_name + '_pre_model.pt',
                                    map_location='cuda:' + str(self.args_.gpu))
            w_pretrain['lenetout.1.weight'] = w_pretrain.pop('output.weight')
            w_pretrain['lenetout.1.bias'] = w_pretrain.pop('output.bias')

        encodernet_dict = encoder_block.state_dict()
        encoderparam_dict = {k: v for k, v in w_pretrain.items() if k in encodernet_dict}
        encodernet_dict.update(encoderparam_dict)
        encoder_block.load_state_dict(encodernet_dict)

        self.Layer_In_List = nn.ModuleList([encoder_block])
        self.unit_num = self.units_num_list[self.layer_idx]
        for _ in range(1, self.unit_num):
            self.Layer_In_List.append(encoder_block)
        self.relu = nn.ReLU()

    def forward(self, input_tensor_list):
        output_list = []
        for idx in range(self.unit_num):
            z = input_tensor_list[idx]
            z = F.dropout(z, p=self.dropout, training=self.training)
            module_out = self.Layer_In_List[idx](z)
            module_out = self.relu(module_out)
            output_list.append(module_out)
        return output_list

#############################
#          Layer 1 to n-1
#############################
class Layer_Mid_Modular(nn.Module):
    def __init__(self, units_num_list, modular_hdims, dropout, layer_idx):
        super(Layer_Mid_Modular, self).__init__()
        self.units_num_list = units_num_list
        self.modular_hdims = modular_hdims
        self.m_modular_layers = len(units_num_list)
        self.dropout = dropout
        self.layer_idx = layer_idx
        model_block = MLP_block(self.modular_hdims[self.layer_idx-1],
                                self.modular_hdims[self.layer_idx])
        self.Layer_Mid_List = nn.ModuleList([model_block])
        self.unit_num = self.units_num_list[self.layer_idx]
        for _ in range(1, self.unit_num):
            self.Layer_Mid_List.append(model_block)
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    def forward(self, input_tensor_list):
        output_list = []
        for idx in range(self.unit_num):
            z = input_tensor_list[idx]
            z = F.dropout(z, p=self.dropout, training=self.training)
            module_out = self.Layer_Mid_List[idx](z)
            output_list.append(module_out)
        return output_list

#############################
#          Layer n
#############################
class Layer_Out_Modular(nn.Module):
    def __init__(self, units_num_list, modular_hdims, n_class, dropout, layer_idx):
        super(Layer_Out_Modular, self).__init__()
        self.units_num_list = units_num_list
        self.modular_hdims = modular_hdims
        self.m_modular_layers = len(units_num_list)
        self.n_class = n_class
        self.dropout = dropout
        self.layer_idx = layer_idx
        model_block = Linear_mapping(self.modular_hdims[self.layer_idx - 1],
                                     self.n_class)
        self.Layer_Out_List = nn.ModuleList([model_block])
        self.unit_num = self.units_num_list[self.layer_idx]
        for _ in range(1, self.unit_num):
            self.Layer_Out_List.append(model_block)
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    def forward(self, input_tensor_list):
        output_list = []
        for idx in range(self.unit_num):
            z = input_tensor_list[idx]
            z = F.dropout(z, p=self.dropout, training=self.training)
            module_out = self.Layer_Out_List[idx](z)
            output_list.append(module_out)
        return output_list














