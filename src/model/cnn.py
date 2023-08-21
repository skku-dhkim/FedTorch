from src.model import *
from torch.nn import *


class SimpleCNN(Module):
    def __init__(self, num_classes: int = 10, **kwargs):
        super(SimpleCNN, self).__init__()

        self.features = Sequential(
            Conv2d(3, 6, (5, 5)),
            ReLU(),
            MaxPool2d((2, 2)),
            Conv2d(6, 16, (5, 5)),
            ReLU(),
            MaxPool2d((2, 2))
        )
        self.fc_1 = Linear(16 * 5 * 5, 120)
        self.fc_2 = Linear(120, 84)
        self.logit = Linear(84, num_classes)

        self.fc_list = [self.fc_1, self.fc_2]

        if 'features' in kwargs:
            self.output_feature_map = kwargs['features']
        else:
            self.output_feature_map = False

    def forward(self, x):
        x = self.features(x)
        features = torch.flatten(x, 1)

        for i, layer in enumerate(self.fc_list):
            features = F.relu(layer(features), inplace=True)

        logit = self.logit(features)

        if self.output_feature_map:
            return logit, features
        else:
            return logit
