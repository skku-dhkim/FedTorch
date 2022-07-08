from src import *
from src.model import *
import copy
import torch
import torch.nn as nn


def init_weights(layer):
    if isinstance(layer, nn.Conv2d):
        nn.init.xavier_uniform(layer.weight)
        layer.bias.data.fill_(0.01)
    elif isinstance(layer, nn.Linear):
        nn.init.xavier_uniform(layer.weight)
        layer.bias.data.fill_(0.01)


class CustomCNN(FederatedModel):
    def __init__(self, num_classes: int = 10, b_global: bool = False, **kwargs):
        super(CustomCNN, self).__init__()
        if b_global:
            self.features = nn.Sequential(
                nn.Conv2d(3, 6, (5, 5), bias=False),
                nn.ReLU(),
                nn.MaxPool2d((2, 2)),
                nn.Conv2d(6, 16 * kwargs['n_of_clients'], (5, 5), bias=False),
                nn.ReLU(),
                nn.MaxPool2d((2, 2))
            )
            self.fc = nn.Sequential(
                nn.Linear(16 * kwargs['n_of_clients'] * 5 * 5, 120),
                nn.ReLU(),
                nn.Linear(120, 84),
                nn.ReLU(),
                nn.Linear(84, num_classes)
            )
        else:
            self.features = nn.Sequential(
                nn.Conv2d(3, 6, (5, 5)),
                nn.ReLU(),
                nn.MaxPool2d((2, 2)),
                nn.Conv2d(6, 16, (5, 5)),
                nn.ReLU(),
                nn.MaxPool2d((2, 2))
            )
            self.fc = nn.Sequential(
                nn.Linear(16*5*5, 120),
                nn.ReLU(),
                nn.Linear(120, 84),
                nn.ReLU(),
                nn.Linear(84, num_classes)
            )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# class CustomCNN(FederatedModel):
#     def __init__(self, num_classes: int = 10, b_global: bool = False, **kwargs):
#         self.features = FeatureBlock()
#         self.fc = nn.Sequential(
#             nn.Linear(16 * 5 * 5, 120),
#             nn.ReLU(),
#             nn.Linear(120, 84),
#             nn.ReLU(),
#             nn.Linear(84, num_classes)
#         )

class MiniCustomCNN(nn.Module):
    def __init__(self):
        super(MiniCustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, (5, 5)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(6, 8, (5, 5)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )
        self.fc = nn.Sequential(
            nn.Linear(200, 40),
            nn.ReLU(),
            nn.Linear(40, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class FeatureBlock(FederatedModel):
    def __init__(self):
        super(FeatureBlock, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, (5, 5)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(6, 16, (5, 5)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )

    def forward(self, x):
        return self.features(x)


class MixedModel(FederatedModel):
    def __init__(self, models):
        super(MixedModel, self).__init__()
        feature_block = nn.Sequential(
            nn.Conv2d(3, 6, (5, 5)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(6, 16, (5, 5)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )
        self.features = []

        for k, v in models.items():
            _w = OrderedDict({k: val.clone().detach().cpu() for k, val in v.features.state_dict().items()})
            _fb = copy.deepcopy(feature_block)
            _fb.load_state_dict(_w)
            _fb.requires_grad_(False)
            self.features.append(_fb)

        self.fc = nn.Sequential(
            nn.Linear(16*5*5*len(models), 120*len(models)),
            nn.ReLU(),
            nn.Linear(120*len(models), 84*len(models)),
            nn.ReLU(),
            nn.Linear(84*len(models), 10*len(models)),
            nn.ReLU(),
            nn.Linear(10*len(models), 10)
        )

    def freeze_feature_layer(self):
        for feature in self.features:
            feature.requires_grad_(False)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        _out = []
        for model in self.features:
            _y = model(x)
            _y = torch.flatten(_y, start_dim=1)
            _out.append(_y)

        _out = torch.cat(_out, dim=1)
        out = self.fc(_out)
        return out
