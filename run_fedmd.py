import torch
torch.cuda.current_device()
from utils.args import *
from utils.dataset_prepare import *
from utils.vec_prepare import bits_vec_prepare
from ModularNet.modular_encoder import Modular_Encoder
from utils.param_block import param_block_initial, active_blocks, copy_weights
from locals.LocalUpdate import LocalUpdate
from locals.LocalTest import LocalTest
from utils.averaging import fedmd_averaging
import copy
from models.mobilenet import get_mobilenet
from pretrain.FedAvg import pretrain
from models.lenet5 import LenetCNN

def pretrain_prepare(args_,train_iterators, test_iterators):
    if args_.dataset=='cifar10' or args_.dataset=='cifar100':
        pre_model = get_mobilenet(args_.n_class).to(args_.device)
    if args_.dataset=='emnist' or args_.dataset=='femnist':
        pre_model = LenetCNN(args_.n_class).to(args_.device)
    w_pre_model = pretrain(args_, pre_model, train_iterators, test_iterators)
    torch.save(w_pre_model, './save/'+args_.dataset+'_pre_model.pt')



def model_local_train(args_, net, train_iterator, whole_train_iterator):
    local = LocalUpdate(args_=args_, train_iterator=train_iterator,
                        whole_train_iterator=whole_train_iterator)
    w_net_local, train_loss_local, policy_out_local \
        = local.train(net=copy.deepcopy(net).to(args_.device),
                      temp=args_.temperature,
                      learning_rate=args_.fedlocal_lr)
    return w_net_local, train_loss_local, policy_out_local

def model_local_test(args_, net, test_iterator, policy_local):
    local_test = LocalTest(args_=args_, test_iterator=test_iterator,
                           policy_selection=policy_local)
    test_acc_local, test_loss_local = local_test.test(net)
    return test_acc_local, test_loss_local

def run_experiment(args_):
    torch.manual_seed(args_.seed)
    args_.device = torch.device('cuda:{}'.format(args_.gpu)
                                if torch.cuda.is_available()
                                   and args_.gpu != -1 else 'cpu')
    #########################
    # dataset preparation
    #########################
    train_iterators, \
    val_iterators, \
    test_iterators, \
    whole_train_iterators, \
    client_num = data_prepare(args_)

    #########################
    # encoder pretrain
    #########################
    #pretrain_prepare(args_, train_iterators, test_iterators)

    #########################
    # create models
    #########################
    bits_vec_list = bits_vec_prepare(args_)
    net_glob = Modular_Encoder(args_=args_,
                               policy_hdim_x=args_.policy_hdim_x,
                               policy_hdim_y=args_.policy_hdim_y,
                               policy_outdim_list=bits_vec_list,
                               dropout=args_.droprate,
                               hard=False).to(args_.device)

    net_glob.train()
    # copy weights
    w_glob, w_param_num, modular_param_num, policy_param_num, block_param_num \
        = copy_weights(net_glob)

    # initialization of a 'structure' of active units parameters
    blockactive_units_param, \
    blockactive_units_param_zero, \
    policy_blocks_, \
    active_unit_flag_blocks = param_block_initial(args_,
                                                  bits_vec_list,
                                                  block_param_num)

    w_locals = [w_glob for _ in range(client_num)]
    structure_activeunits_locals \
        = [blockactive_units_param_zero for _ in range(client_num)]
    active_unit_flag_locals \
        = [active_unit_flag_blocks for _ in range(client_num)]
    final_policy_locals = [policy_blocks_ for _ in range(client_num)]


    print("client number: {}".format(client_num))
    test_acc_localmodel_mean =[]
    test_loss_localmodel_mean =[]
    for iter in range(args_.fed_rounds):
        idxs_users = np.array([_ for _ in range(client_num)])

        ####################
        # training
        ####################

        train_loss_locals = [0 for _ in range(client_num)]
        test_loss_locals = [0 for _ in range(client_num)]
        test_acc_locals = [0 for _ in range(client_num)]

        test_loss_localmodel = [0 for _ in range(client_num)]
        test_acc_localmodel = [0 for _ in range(client_num)]
        for idx in idxs_users:

            # local training
            net_glob.train()
            w_net_local, train_loss_local, policy_out_local \
                = model_local_train(args_=args_, net=net_glob,
                                    train_iterator=train_iterators[idx],
                                    whole_train_iterator=whole_train_iterators[idx])

            # local training output
            w_locals[idx] = copy.deepcopy(w_net_local)
            train_loss_locals[idx] = train_loss_local
            final_policy_locals[idx] = policy_out_local


            # local test before averaging
            net_glob.eval()
            net_glob.load_state_dict(w_net_local)
            test_acc_l, test_loss_l = \
                model_local_test(args_, net_glob, test_iterators[idx],
                                 final_policy_locals[idx])
            test_loss_localmodel[idx]=test_loss_l
            test_acc_localmodel[idx]=test_acc_l


            # updating blocks
            active_unit_flag, blockactive_units_param_ \
                = active_blocks(args_, policy_out_local, blockactive_units_param)
            structure_activeunits_locals[idx] = copy.deepcopy(blockactive_units_param_)
        w_glob = fedmd_averaging(w_locals, structure_activeunits_locals,
                                 modular_param_num)
        net_glob.load_state_dict(w_glob)
        test_acc_localmodel_mean.append(sum(test_acc_localmodel) / client_num)
        test_loss_localmodel_mean.append(sum(test_loss_localmodel) / client_num)
        ########################
        # test
        ########################

        net_glob.eval()
        for idx in idxs_users:
            test_acc_local, test_loss_local =\
                model_local_test(args_, net_glob, test_iterators[idx],
                                 final_policy_locals[idx])
            test_loss_locals[idx] = test_loss_local
            test_acc_locals[idx] = test_acc_local
        test_loss_avg = sum(test_loss_locals)/client_num
        test_acc_avg = sum(test_acc_locals)/client_num
        train_loss_avg = sum(train_loss_locals)/client_num

        print('Round {:3d}, training loss {:.3f}, average test loss {:.3f}, '
              'average test acc {:.3f}'.format(iter,train_loss_avg,
                                       test_loss_avg, test_acc_avg))

        sys.stdout.flush()




if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    args = parse_args()
    run_experiment(args)