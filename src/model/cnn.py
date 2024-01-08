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

        if 'data_type' in kwargs.keys() and 'mnist' in kwargs['data_type']:
            # if 'mnist' in kwargs['data_type']:
            # NOTE: If data is mnist type.
            self.classifier = Sequential(
                Linear(16 * 4 * 4, 120),
                ReLU(inplace=True),
                Linear(120, num_classes))
        else:
            self.classifier = Sequential(
                Linear(16 * 5 * 5, 120),
                ReLU(inplace=True),
                Linear(120, num_classes))

        # self.logit = Linear(120, num_classes)
        # self.classifier = Sequential(
        #     Linear(16 * 5 * 5, 120),
        #     ReLU(inplace=True),
        #     Linear(120, num_classes)
        # )

        if 'features' in kwargs:
            self.output_feature_map = kwargs['features']
        else:
            self.output_feature_map = False

    def forward(self, x):
        x = self.features(x)
        features = torch.flatten(x, 1)
        # fc = F.relu(self.classifier(features), inplace=True)
        # logit = self.logit(fc)
        logit = self.classifier(features)

        if self.output_feature_map:
            return logit, features
        else:
            return logit


class ConvNet(Module):
    def __init__(self, num_classes: int = 10, **kwargs):
        super(ConvNet, self).__init__()

        self.features = Sequential(
            Conv2d(3, 64, (3, 3), padding=1),
            ReLU(),
            Conv2d(64, 64, (3, 3), padding=1),
            ReLU(),
            Conv2d(64, 128, (3, 3), padding=1),
            ReLU(),
            MaxPool2d((2, 2), padding=1)
        )

        if 'data_type' in kwargs.keys():
            if 'mnist' in kwargs['data_type']:
                # NOTE: If data is mnist type.
                self.classifier = Linear(128 * 15 * 15, num_classes)
        else:
            self.classifier = Linear(128 * 17 * 17, num_classes)

        # self.classifier = Linear(128 * 17 * 17, num_classes)

        if 'features' in kwargs:
            self.output_feature_map = kwargs['features']
        else:
            self.output_feature_map = False

    def forward(self, x):
        x = self.features(x)
        features = torch.flatten(x, 1)
        logit = self.classifier(features)

        if self.output_feature_map:
            return logit, features
        else:
            return logit
