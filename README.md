# FedMN
This is a Pytorch implementation of paper: Personalized Federated Learning via Heterogeneous Modular Networks.

### Requirements:
Python 3.7.13
Pytorch 1.12.1
Torchvision 0.13.1
cvxpy
networkx

### Data
The data processing is adapted from: https://github.com/omarfoq/FedEM/tree/main/data

### Example to use:
python run_fedmd.py --dataset 'cifar10' --n_class 10 --fedlocal_lr 0.01 --fed_rounds 151 --gpu 0
