import os
import argparse


def parse_args(args_list=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--dataset',
        help='name of dataset',
        type=str,
        default='cifar10'
    )
    parser.add_argument(
        '--sampling_rate',
        help='proportion of clients to be used at each round; default is 1.0',
        type=float,
        default=1.0
    )
    parser.add_argument('--pretrain_rounds',
                        help='pretrain rounds',
                        type=int,
                        default=146)
    parser.add_argument('--fed_rounds',
                        help='rounds of federated learning',
                        type=int,
                        default=151)
    parser.add_argument('--pretrain_local_epochs',
                        help='epochs of pretrian local updating',
                        type=int,
                        default=1)
    parser.add_argument('--fed_local_epochs',
                        help='epochs of fed local updating',
                        type=int,
                        default=1)
    parser.add_argument("--seed",
                        help='random seed',
                        type=int,
                        default=1234
    )
    parser.add_argument('--gpu',
                         help='index of GPU',
                         type=int,
                         default=0
    )
    parser.add_argument("--pretrain_lr",
                        type=float,
                        help='learning rate of pretrain',
                        default=0.07
    )
    parser.add_argument("--fedlocal_lr",
                        type=float,
                        help='learning rate of federated learning',
                        default=0.02
                        )
    parser.add_argument('--bz',
                        help='batch_size',
                        type=int,
                        default=128
    )
    parser.add_argument('--validation',
                        help='if chosen the validation part will be used instead of test part;'
                            ' make sure to use `val_frac > 0` in `generate_data.py`;',
                        action='store_true'
    )
    parser.add_argument("--n_class",
                        help='number of classes',
                        type=int,
                        default=10
    )
    parser.add_argument("--policy_hdim_x",
                        help='hidden dim of x in policy net',
                        type=int,
                        default=128
                        )
    parser.add_argument("--policy_hdim_y",
                        help='hidden dim of y in policy net',
                        type=int,
                        default=64
                        )
    parser.add_argument("--modular_hdims",
                        help='modular layer out dim',
                        type=int,
                        default=[512,256]
                        )
    parser.add_argument("--units_list",
                        help='unit number in the modular networks',
                        type=list,
                        default=[2,2,2]
                        )
    parser.add_argument("--temperature",
                        help='temperature initial',
                        type=float,
                        default=1.0
                        )
    parser.add_argument("--droprate",
                        help='dropout rate',
                        type=float,
                        default=0.0
                        )

    if args_list:
        args = parser.parse_args(args_list)
    else:
        args = parser.parse_args()

    return args
