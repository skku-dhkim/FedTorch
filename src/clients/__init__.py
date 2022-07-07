import wandb
import ray
from torch.optim import *

num_of_classes = {
    'cifar-10': 10,
    'cifar-100': 100,
    'mnist': 10
}


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
    'num_of_classes',
    'ray'
]
