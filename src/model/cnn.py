from src.model import *
from torch.nn import *


class SimpleCNN(Module):
    def __init__(self, num_classes: int = 10, **kwargs):
        super(SimpleCNN, self).__init__()
        self.layer1 = Conv2d(3, 16, (5, 5))
        self.layer2 = Conv2d(16, 32, (5, 5))
        self.max_pool = MaxPool2d((2, 2))
        if 'data_type' in kwargs.keys() and 'mnist' in kwargs['data_type']:
            # NOTE: If data is mnist type.
            self.fc1 = Linear(32 * 4 * 4, 192)
        else:
            self.fc1 = Linear(32 * 5 * 5, 192)
        self.fc2 = Linear(192, num_classes)

        if 'features' in kwargs:
            self.output_feature_map = kwargs['features']
        else:
            self.output_feature_map = False

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.max_pool(x)
        x = F.relu(self.layer2(x))
        x = self.max_pool(x)
        features = torch.flatten(x, 1)
        out = F.relu(self.fc1(features))
        logit = self.fc2(out)

        if self.output_feature_map:
            return logit, features
        else:
            return logit


class ConvNet(Module):
    def __init__(self, num_classes: int = 10, **kwargs):
        super(ConvNet, self).__init__()

        self.layer1 = Conv2d(3, 64, (3, 3), padding=1)
        self.layer2 = Conv2d(64, 64, (3, 3), padding=1)
        self.layer3 = Conv2d(64, 128, (3, 3), padding=1)
        self.max_pool = MaxPool2d((2, 2))

        if 'data_type' in kwargs.keys() and 'mnist' in kwargs['data_type']:
            # NOTE: If data is mnist type.
            self.fc = Linear(128 * 15 * 15, num_classes)
        else:
            self.fc = Linear(128 * 17 * 17, num_classes)

        if 'features' in kwargs:
            self.output_feature_map = kwargs['features']
        else:
            self.output_feature_map = False

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = self.max_pool(x)
        features = torch.flatten(x, 1)
        logit = self.fc(features)

        if self.output_feature_map:
            return logit, features
        else:
            return logit
