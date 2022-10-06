import torch
import copy

def policy_blocks_prepare(bits_vec_list):
    policy_blocks_ = []
    for layer in range(len(bits_vec_list)):
        layer_block_ = []
        for k in range(bits_vec_list[layer]):
            layer_block_.append(0)
        policy_blocks_.append(torch.Tensor(layer_block_))
    return policy_blocks_

def active_unit_flag_prepare(units_num_list):
    active_unit_flag_blocks = []
    for layer in range(len(units_num_list)):
        layer_active_unit_flag_ = []
        for units_ in range(units_num_list[layer]):
            layer_active_unit_flag_.append(0)
        active_unit_flag_blocks.append(layer_active_unit_flag_)
    return active_unit_flag_blocks


def param_block_initial(args_, bits_vec_list, block_param_num):
    blockactive_units_param = param_block(args_.units_list, block_param_num)
    blockactive_units_param_zero = param_blockzero(args_.units_list, block_param_num)
    policy_blocks_ = policy_blocks_prepare(bits_vec_list)
    active_unit_flag_blocks = active_unit_flag_prepare(args_.units_list)

    return blockactive_units_param,\
           blockactive_units_param_zero,\
           policy_blocks_,\
           active_unit_flag_blocks

def param_block(units_num_list, block_param_num):
    n_modular_layers = len(units_num_list)
    blockactive_units_param = []
    # layer 0
    layer0_units_param = []
    for un_idx in range(units_num_list[0]):
        layer0_units_param.append([1 for _ in range(block_param_num)])
    blockactive_units_param.append(layer0_units_param)
    # layer 1-n
    for layer_idx in range(1, n_modular_layers):
        layer_units_param = []
        for un_idx in range(units_num_list[layer_idx]):
            layer_units_param.append([1, 1])
        blockactive_units_param.append(layer_units_param)

    return blockactive_units_param


def param_blockzero(units_num_list, block_param_num):
    n_modular_layers = len(units_num_list)
    blockactive_units_param_zero = []
    # layer 0
    layer0_units_param = []
    for un_idx in range(units_num_list[0]):
        layer0_units_param.append([0 for _ in range(block_param_num)])
    blockactive_units_param_zero.append(layer0_units_param)
    # layer 1-n
    for layer_idx in range(1, n_modular_layers):
        layer_units_param = []
        for un_idx in range(units_num_list[layer_idx]):
            layer_units_param.append([0, 0])
        blockactive_units_param_zero.append(layer_units_param)

    return blockactive_units_param_zero


def unit_select(policy_list, units_num_list):
    active_unit_flag = []
    policy_map = policy_list[0]
    flag_vec = []
    for units_ in range(units_num_list[0]):
        flag_vec.append(int(policy_map[units_].item()))
    active_unit_flag.append(flag_vec)

    for layer_ in range(1, len(units_num_list)):
        policy_map = policy_list[layer_].reshape(units_num_list[layer_],
                                                 units_num_list[layer_ - 1])
        flag_vec = []
        list1 = active_unit_flag[layer_ - 1]
        for units_ in range(units_num_list[layer_]):
            list2 = policy_map[units_, :].numpy().tolist()
            product = [x * y for x, y in zip(list1, list2)]
            if sum(product) > 0:
                flag_vec.append(1)
            else:
                flag_vec.append(0)
        active_unit_flag.append(flag_vec)
    return active_unit_flag


def active_blocks(args_, policy_out, blockactive_units_param):
    #  active unit flag structure
    active_unit_flag = unit_select(policy_out, args_.units_list)
    blockactive_units_param_ = copy.deepcopy(blockactive_units_param)

    # which units in layer 0 are deactivated?
    for layer_ in range(len(args_.units_list)):
        for units_ in range(args_.units_list[layer_]):
            if active_unit_flag[layer_][units_] == 0:
                blockactive_units_param_[layer_][units_] = \
                    [0 for _ in blockactive_units_param_[layer_][units_]]

    return active_unit_flag, blockactive_units_param_

def copy_weights(net_glob):
    w_glob = net_glob.state_dict()
    w_param_num = len(w_glob.keys())
    modular_param_num = 0
    for k in w_glob.keys():
        if k.startswith('layer_'):
            modular_param_num = modular_param_num + 1
    policy_param_num = w_param_num - modular_param_num

    block_param_num=0
    for k in w_glob.keys():
        if k.startswith('layer_in_modular.Layer_In_List.0'):
            block_param_num = block_param_num+1

    layer_param_num = 0
    for k in w_glob.keys():
        if k.startswith('layer_in_modular.Layer_In_List'):
            layer_param_num = layer_param_num+1
    return w_glob, w_param_num, modular_param_num, policy_param_num, block_param_num