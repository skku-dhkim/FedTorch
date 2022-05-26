from torch.utils.data import Dataset, DataLoader
from torch.nn.parameter import Parameter
from typing import Union, Optional, Tuple

num_of_classes = {
    'cifar-10': 10,
    'cifar-100': 100,
    'mnist': 10
}

__all__ = [
    'num_of_classes',
    'Dataset',
    'DataLoader',
    'Parameter',
    'Union', 'Optional', 'Tuple'

]

