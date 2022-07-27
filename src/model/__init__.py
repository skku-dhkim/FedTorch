from src.model.skeleton import FederatedModel
from .custom_cnn import CustomCNN, ModelFedCon
from torchvision.models import *
from torch.nn import *

import torch.nn.functional as F


def model_call(model_name: str, num_of_classes: int):
    if model_name.lower() == 'custom_cnn':
        return CustomCNN(num_of_classes=num_of_classes)
    if model_name.lower() == 'moon_cnn':
        return ModelFedCon(10, n_classes=num_of_classes)
    if model_name.lower() == "resnet-50":
        _model = resnet50()
        fc = Sequential(
            Linear(in_features=2048, out_features=1000, bias=True),
            ReLU(inplace=True),
            Linear(in_features=1000, out_features=num_of_classes, bias=True)
        )
        _model.fc = fc
        return _model
    if model_name.lower() == "resnet-18":
        _model = resnet18()
        fc = Sequential(
            Linear(in_features=512, out_features=256, bias=True),
            ReLU(inplace=True),
            Linear(in_features=256, out_features=num_of_classes, bias=True)
        )
        _model.fc = fc
        return _model
    else:
        raise NotImplementedError("Not implemented yet.")


__all__ = [
    'FederatedModel',
    'CustomCNN',
    'model_call',
    'F',
]
