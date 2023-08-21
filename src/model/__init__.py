import torch
import copy
import torch.nn.functional as F

from .custom_cnn import CustomCNN, ModelFedCon
from .cnn import SimpleCNN
from .resnet import ResNet50_cifar10
from torchvision.models import *
from torch.nn import Sequential, Linear, ReLU


def model_call(model_name: str, num_of_classes: int, **kwargs):
    if model_name.lower() == 'custom_cnn':
        return CustomCNN(num_of_classes=num_of_classes)
    elif model_name.lower() == 'moon_cnn':
        return ModelFedCon(10, n_classes=num_of_classes)
    elif model_name.lower() == "resnet-50":
        _model = ResNet50_cifar10(num_classes=num_of_classes)
        return _model
    elif model_name.lower() == "resnet-18":
        _model = resnet18()
        fc = Sequential(
            Linear(in_features=512, out_features=256, bias=True),
            ReLU(inplace=True),
            Linear(in_features=256, out_features=num_of_classes, bias=True)
        )
        _model.fc = fc
        return _model
    elif model_name.lower() == 'simple_cnn':
        return SimpleCNN(num_classes=num_of_classes, **kwargs)
    else:
        raise NotImplementedError("Not implemented yet.")


NUMBER_OF_CLASSES = {
        'cifar-10': 10,
        'cifar-100': 100,
        'mnist': 10
}

__all__ = [
    'CustomCNN',
    'SimpleCNN',
    'model_call',
    'F',
    'NUMBER_OF_CLASSES',
    'torch',
    'copy',
]
