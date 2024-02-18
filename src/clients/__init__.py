from torch.optim import *
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict, defaultdict
from src.clients.fed_clients import Client, FedBalancerClient
from src.clients.aggregator import Aggregator, AggregationBalancer, AvgAggregator


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


__all__ = [
    'call_optimizer',
    "SummaryWriter",
    'OrderedDict', "defaultdict",
    'Client',
    'Aggregator',
    'FedBalancerClient',
    'AggregationBalancer',
    'AvgAggregator'
    # 'FedKL_Client'
]
