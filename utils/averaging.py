import copy
import torch

def fedavg_averaging(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def fedmd_averaging(w, structure_activeunits_locals, modular_param_num):
    sequence_activeunits_locals = []
    for local_idx in range(len(structure_activeunits_locals)):
        this_local = []
        for layer in range(len(structure_activeunits_locals[local_idx])):
            for unit_ in range(len(structure_activeunits_locals[local_idx][layer])):
                this_local = this_local + structure_activeunits_locals[local_idx][layer][unit_]
                # for param_ in range(len(structure_activeunits_locals[local_idx][layer][unit_])):
                #    this_local.append(structure_activeunits_locals[local_idx][layer][unit_][param_])
        sequence_activeunits_locals.append(this_local)
    sequence_activeunits_locals_tensor = torch.Tensor(sequence_activeunits_locals)

    w_avg = copy.deepcopy(w[0])
    count = 0
    for k in w[0].keys():
        if count < modular_param_num:
            w_avg[k] = copy.deepcopy(w[0][k] * sequence_activeunits_locals_tensor[0, count])
        else:
            w_avg[k] = copy.deepcopy(w[0][k])
        count = count + 1
    divider = torch.sum(sequence_activeunits_locals_tensor, dim=0)

    count = 0
    for k in w_avg.keys():
        if count < modular_param_num:
            for i in range(1, len(w)):
                w_avg[k] += w[i][k] * sequence_activeunits_locals_tensor[i, count]
            if divider[count] > 0:
                w_avg[k] = torch.div(w_avg[k], divider[count])
        else:
            for i in range(1, len(w)):
                w_avg[k] += w[i][k]
            w_avg[k] = torch.div(w_avg[k], len(w))
        count = count + 1
    return w_avg