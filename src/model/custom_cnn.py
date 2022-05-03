import torch
import torch.nn as nn
import torch.nn.functional as F
from .skeleton import FederatedModel


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
        x = x.permute(0, 3, 1, 2)
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


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
