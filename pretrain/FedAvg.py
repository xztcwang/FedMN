import torch
torch.cuda.current_device()
from pretrain.LocalUpdateFedAvg import LocalUpdateFedAvg
from pretrain.LocalTestFedAvg import LocalTestFedAvg
import copy
from utils.averaging import fedavg_averaging
from utils.dataset_prepare import *

def pretrain(args_, pre_model, train_iterators, test_iterators):

    client_num = len(train_iterators)
    w_pre_model = pre_model.state_dict()
    w_locals = [w_pre_model for _ in range(client_num)]

    for iter in range(args_.pretrain_rounds):
        m = max(int(args_.sampling_rate * client_num), 1)
        idxs_users = np.random.choice(range(client_num), m, replace=False)
        local_accuracy_list = [0 for _ in range(len(idxs_users))]
        local_test_loss_list = [0 for _ in range(len(idxs_users))]
        local_loss = []
        # training
        pre_model.train()
        for idx in idxs_users:
            local_train = LocalUpdateFedAvg(args_=args_, train_iterator=train_iterators[idx])
            w, train_loss = local_train.train(net=copy.deepcopy(pre_model).to(args_.device),
                                              learning_rate=args_.pretrain_lr)
            w_locals[idx] = copy.deepcopy(w)
            local_loss.append(train_loss)
        loss_train_avg = sum(local_loss) / client_num

        # averaging
        w_avg = fedavg_averaging(w_locals)
        pre_model.load_state_dict(w_avg)

        # test
        pre_model.eval()
        for idx in idxs_users:
            local_test = LocalTestFedAvg(args_=args_, test_iterator=test_iterators[idx])
            local_test_acc, local_test_loss = local_test.test(pre_model)
            local_accuracy_list[idx] = copy.deepcopy(local_test_acc)
            local_test_loss_list[idx] = copy.deepcopy(local_test_loss)
        acc_test_avg = sum(local_accuracy_list)/client_num
        loss_test_avg = sum(local_test_loss_list)/client_num

        print('Round {:3d}, Average training loss {:.3f}, '
              'Average test loss {:.3f}, '
              'Average test accuracy {:.3f}'.format(iter,
                                                    loss_train_avg,
                                                    loss_test_avg,
                                                    acc_test_avg))
    return pre_model.state_dict()








