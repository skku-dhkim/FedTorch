from torch.optim import *

import torch


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
    'torch',
    'call_optimizer'
]
