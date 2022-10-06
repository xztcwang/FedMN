import sys
from utils.args import *
from utils.utils import *

def data_prepare(args_):
    data_dir = get_data_dir(args_.dataset)

    print("==> Clients initialization..")
    sys.stdout.flush()
    train_iterators, val_iterators, test_iterators, whole_train_iterators = \
        get_loaders(
            type_=LOADER_TYPE[args_.dataset],
            root_path=os.path.join(data_dir, "train"),
            batch_size=args_.bz,
            is_validation=args_.validation
        )
    client_num = len(train_iterators)
    return train_iterators, \
           val_iterators, \
           test_iterators, \
           whole_train_iterators, \
           client_num
