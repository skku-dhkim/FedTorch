import torch
import copy
import torch.nn.functional as F

from .custom_cnn import CustomCNN, ModelFedCon
from .cnn import SimpleCNN, ConvNet
from .resnet import ResNet18, ResNet50
from .vgg import vgg11, vgg16
from .mlp import MLP


def model_call(model_name: str, num_of_classes: int, **kwargs):
    if model_name.lower() == 'custom_cnn':
        return CustomCNN(num_of_classes=num_of_classes)
    elif model_name.lower() == 'moon_cnn':
        return ModelFedCon(10, n_classes=num_of_classes)
    elif model_name.lower() == "resnet-50":
        _model = ResNet50(num_classes=num_of_classes, **kwargs)
        return _model
    elif model_name.lower() == "resnet-18":
        _model = ResNet18(num_classes=num_of_classes, **kwargs)
        return _model
    elif model_name.lower() == 'simple_cnn':
        return SimpleCNN(num_classes=num_of_classes, **kwargs)
    elif model_name.lower() == 'convnet':
        return ConvNet(num_classes=num_of_classes, **kwargs)
    elif model_name.lower() == 'mlp':
        return MLP(num_classes=num_of_classes, **kwargs)
    elif model_name.lower() == 'vgg11':
        return vgg11(num_classes=num_of_classes, init_weights=False)
    elif model_name.lower() == 'vgg16':
        return vgg16(num_classes=num_of_classes, init_weights=False)
    else:
        raise NotImplementedError("Not implemented yet.")


NUMBER_OF_CLASSES = {
    'cifar-10': 10,
    'cifar-100': 100,
    'mnist': 10,
    'organamnist': 11,
    'bloodmnist': 8
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
