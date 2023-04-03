from torch.optim import *
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict
from src.clients.fed_clients import Client
from src.clients.aggregator import Aggregator


def call_optimizer(optimizer_name: str):
    if optimizer_name.lower() == 'sgd':
        opt = SGD
    elif optimizer_name.lower() == 'adam':
        opt = Adam
    elif optimizer_name.lower() == 'rmsprop':
        opt = RMSprop
    else:
        return NotImplementedError("Other optimization is not implemented yet")
    return opt


# NUMBER_OF_CLASSES = {
#         'cifar-10': 10,
#         'cifar-100': 100,
#         'mnist': 10
# }

__all__ = [
    'call_optimizer',
    "SummaryWriter",
    'OrderedDict',
    'Client',
    'Aggregator',
    # 'FedKL_Client'
]
