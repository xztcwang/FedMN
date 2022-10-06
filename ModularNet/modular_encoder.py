import torch
from torch import nn
from PolicyNet.policy_stack import create_policy
import ModularNet.modular_layers as mod_layers

class Modular_Encoder(nn.Module):
    def __init__(self,
                 args_,
                 policy_hdim_x,
                 policy_hdim_y,
                 policy_outdim_list,
                 dropout,
                 hard=True):
        super(Modular_Encoder, self).__init__()
        self.args_ = args_
        self.units_num_list = args_.units_list
        self.modular_hdims = args_.modular_hdims
        self.n_class = args_.n_class
        self.m_modular_layers = len(self.units_num_list)
        self.policy_hdim_x = policy_hdim_x
        self.policy_hdim_y = policy_hdim_y
        self.policy_outdim_list = policy_outdim_list
        self.dropout = dropout
        self.hard = hard
        m_modular_layers = len(self.units_num_list)
        self.modular_layers = [[] for _ in range(m_modular_layers)]

        self.layer_in_modular = mod_layers.Layer_In_Modular(
            args_=self.args_,
            units_num_list=self.units_num_list,
            modular_hdims=self.modular_hdims,
            dropout=self.dropout,
            layer_idx=0)
        if m_modular_layers == 2:
            self.layer_out_modular = mod_layers.Layer_Out_Modular(
                units_num_list=self.units_num_list,
                modular_hdims=self.modular_hdims,
                n_class=self.n_class,
                dropout=self.dropout,
                layer_idx=1)
            self.modular_layers[0].append(self.layer_in_modular)
            self.modular_layers[1].append(self.layer_out_modular)
        elif m_modular_layers > 2:
            self.modular_layers[0].append(self.layer_in_modular)
            for midlayer_ in range(1, m_modular_layers - 1):
                self.layer_mid_modular = mod_layers.Layer_Mid_Modular(
                    units_num_list=self.units_num_list,
                    modular_hdims=self.modular_hdims,
                    dropout=self.dropout,
                    layer_idx=midlayer_)
                self.modular_layers[midlayer_].append(self.layer_mid_modular)
            self.layer_out_modular = mod_layers.Layer_Out_Modular(
                units_num_list=self.units_num_list,
                modular_hdims=self.modular_hdims,
                n_class=self.n_class,
                dropout=self.dropout,
                layer_idx=m_modular_layers - 1)
            self.modular_layers[m_modular_layers - 1].append(self.layer_out_modular)
        else:
            print("Error Stacking Layers")

        self.policy_outdim = sum(self.policy_outdim_list)
        self.policy_net = create_policy(args_=self.args_,
                                        policy_hdim_x=self.policy_hdim_x,
                                        policy_hdim_y=self.policy_hdim_y,
                                        policy_outdim=self.policy_outdim,
                                        hard=self.hard)
        self.relu = nn.ReLU()

    def forward(self, x, x_whole, y_whole, temp, train_flag, policy_trainout=None):
        if train_flag:
            policy_tensor = self.policy_net(x_whole, y_whole, temp)
            policy_list = []
            policy_start_idx = 0
            for policy_layer_idx in range(0, len(self.policy_outdim_list)):
                selected_policy = policy_tensor[
                                  policy_start_idx:
                                  policy_start_idx +
                                  self.policy_outdim_list[policy_layer_idx]]
                policy_list.append(selected_policy)
                policy_start_idx = policy_start_idx + self.policy_outdim_list[policy_layer_idx]
        else:
            policy_tensor = policy_trainout
            policy_list = policy_tensor
        policy_list[0] = torch.Tensor([1 for _ in range(self.units_num_list[0])])

        # feed into modular networks
        # data -> layer 0, using policy_list[0]:
        if self.m_modular_layers ==2:
            policy_map = policy_list[0]
            input_next_layer = []
            for j in range(self.units_num_list[0]):
                input_tensor = x * policy_map[j]
                input_next_layer.append(input_tensor)
            layer_outlist = self.modular_layers[0][0](input_next_layer)

            # layer 0 -> layer 1, using policy_list[1]:
            ############################################
            policy_map = policy_list[1].reshape(self.units_num_list[1],
                                                self.units_num_list[0])
            input_next_layer = []
            for towards_unit_idx in range(self.units_num_list[1]):
                unit_towardslink_sum = policy_map[towards_unit_idx, :].sum()
                input_tensor_sum = 0
                for from_unit_idx in range(self.units_num_list[0]):
                    input_tensor_sum = input_tensor_sum + \
                                       layer_outlist[from_unit_idx] * \
                                       policy_map[towards_unit_idx, from_unit_idx]
                if unit_towardslink_sum > 0:
                    unit_input_tensor = input_tensor_sum / unit_towardslink_sum
                else:
                    unit_input_tensor = input_tensor_sum
                input_next_layer.append(unit_input_tensor)
            layer_outlist = self.modular_layers[1][0](input_next_layer)
        elif self.m_modular_layers > 2:
            # ex. self.m_modular_layers == 3
            policy_map = policy_list[0]
            input_next_layer = []
            for j in range(self.units_num_list[0]):
                input_tensor = x * policy_map[j]
                input_next_layer.append(input_tensor)
            layer_outlist = self.modular_layers[0][0](input_next_layer)
            for mid_layer in range(1, self.m_modular_layers-1):
                policy_map = policy_list[mid_layer].reshape(self.units_num_list[mid_layer],
                                                            self.units_num_list[mid_layer-1])
                input_next_layer = []
                for towards_unit_idx in range(self.units_num_list[mid_layer]):
                    unit_towardslink_sum = policy_map[towards_unit_idx, :].sum()
                    input_tensor_sum = 0
                    for from_unit_idx in range(self.units_num_list[mid_layer-1]):
                        input_tensor_sum = input_tensor_sum + \
                                           layer_outlist[from_unit_idx] * \
                                           policy_map[towards_unit_idx, from_unit_idx]
                    if unit_towardslink_sum > 0:
                        unit_input_tensor = input_tensor_sum / unit_towardslink_sum
                    else:
                        unit_input_tensor = input_tensor_sum
                    input_next_layer.append(unit_input_tensor)
                layer_outlist = self.modular_layers[mid_layer][0](input_next_layer)

            policy_map = policy_list[self.m_modular_layers - 1].reshape(
                self.units_num_list[self.m_modular_layers - 1],
                self.units_num_list[self.m_modular_layers - 2])
            input_next_layer = []
            for towards_unit_idx in range(self.units_num_list[self.m_modular_layers - 1]):
                unit_towardslink_sum = policy_map[towards_unit_idx, :].sum()
                input_tensor_sum = 0
                for from_unit_idx in range(self.units_num_list[self.m_modular_layers - 2]):
                    input_tensor_sum = input_tensor_sum + \
                                       layer_outlist[from_unit_idx] * \
                                       policy_map[towards_unit_idx, from_unit_idx]
                if unit_towardslink_sum > 0:
                    unit_input_tensor = input_tensor_sum / unit_towardslink_sum
                else:
                    unit_input_tensor = input_tensor_sum
                input_next_layer.append(unit_input_tensor)
            layer_outlist = self.modular_layers[self.m_modular_layers - 1][0](input_next_layer)
        else:
            print("Error Modular Encoder!")

        #########output##########
        final_output_tensor = 0
        policy_map = policy_list[self.m_modular_layers]
        divider = sum(policy_map)
        for j in range(self.units_num_list[self.m_modular_layers-1]):
            final_output_tensor = final_output_tensor \
                                  + layer_outlist[j] \
                                  * policy_map[j]
            if divider > 0:
                final_output_tensor = final_output_tensor / divider

        return final_output_tensor, policy_list








