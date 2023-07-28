from src import *
from src.model import *
from torch.nn import *


def init_weights(layer):
    if isinstance(layer, Conv2d):
        init.xavier_uniform(layer.weight)
        layer.bias.data.fill_(0.01)
    elif isinstance(layer, Linear):
        init.xavier_uniform(layer.weight)
        layer.bias.data.fill_(0.01)


class CustomCNN(Module):
    def __init__(self, num_classes: int = 10, b_global: bool = False, **kwargs):
        super(CustomCNN, self).__init__()
        if b_global:
            self.features = Sequential(
                Conv2d(3, 6, (5, 5), bias=False),
                ReLU(),
                MaxPool2d((2, 2)),
                Conv2d(6, 16 * kwargs['n_of_clients'], (5, 5), bias=False),
                ReLU(),
                MaxPool2d((2, 2))
            )
            self.fc = Sequential(
                Linear(16 * kwargs['n_of_clients'] * 5 * 5, 120),
                ReLU(),
                Linear(120, 84),
                ReLU(),
                Linear(84, num_classes)
            )
        else:
            self.features = Sequential(
                Conv2d(3, 6, (5, 5)),
                ReLU(),
                MaxPool2d((2, 2)),
                Conv2d(6, 16, (5, 5)),
                ReLU(),
                MaxPool2d((2, 2))
            )
            self.fc = Sequential(
                Linear(16 * 5 * 5, 120),
                ReLU(),
                Linear(120, 84),
                ReLU(),
                Linear(84, num_classes)
            )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def feature_maps(self, x):
        with torch.no_grad():
            x = self.features(x)
        return x


class MiniCustomCNN(Module):
    def __init__(self):
        super(MiniCustomCNN, self).__init__()
        self.features = Sequential(
            Conv2d(3, 6, (5, 5)),
            ReLU(),
            MaxPool2d((2, 2)),
            Conv2d(6, 8, (5, 5)),
            ReLU(),
            MaxPool2d((2, 2))
        )
        self.fc = Sequential(
            Linear(200, 40),
            ReLU(),
            Linear(40, 20),
            ReLU(),
            Linear(20, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class FeatureBlock(Module):
    def __init__(self):
        super(FeatureBlock, self).__init__()
        self.features = Sequential(
            Conv2d(3, 6, (5, 5)),
            ReLU(),
            MaxPool2d((2, 2)),
            Conv2d(6, 16, (5, 5)),
            ReLU(),
            MaxPool2d((2, 2))
        )

    def forward(self, x):
        return self.features(x)


class MixedModel(Module):
    def __init__(self, models):
        super(MixedModel, self).__init__()
        feature_block = Sequential(
            Conv2d(3, 6, (5, 5)),
            ReLU(),
            MaxPool2d((2, 2)),
            Conv2d(6, 16, (5, 5)),
            ReLU(),
            MaxPool2d((2, 2))
        )
        self.features = []

        for k, v in models.items():
            _w = OrderedDict({k: val.clone().detach().cpu() for k, val in v.features.state_dict().items()})
            _fb = copy.deepcopy(feature_block)
            _fb.load_state_dict(_w)
            _fb.requires_grad_(False)
            self.features.append(_fb)

        self.fc = Sequential(
            Linear(16 * 5 * 5 * len(models), 120 * len(models)),
            ReLU(),
            Linear(120 * len(models), 84 * len(models)),
            ReLU(),
            Linear(84 * len(models), 10 * len(models)),
            ReLU(),
            Linear(10 * len(models), 10)
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


class ModelFedCon(Module):
    """
        classifier part return, source, label , representation
    """

    def __init__(self, out_dim, n_classes, net_configs=None):
        super(ModelFedCon, self).__init__()

        self.features = SimpleCNN_header(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=n_classes)
        num_ftrs = 84

        self.l1 = Linear(num_ftrs, num_ftrs)
        self.l2 = Linear(num_ftrs, out_dim)

        # last layer
        self.l3 = Linear(out_dim, n_classes)

    def forward(self, x):
        h = self.features(x)
        h = h.squeeze()  # representation tensor
        x = self.l1(h)
        x = F.relu(x)
        x = self.l2(x)

        y = self.l3(x)
        return y


class SimpleCNN_header(Module):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNN_header, self).__init__()
        self.conv1 = Conv2d(3, 6, 5)
        self.relu = ReLU()
        self.pool = MaxPool2d(2, 2)
        self.conv2 = Conv2d(6, 16, 5)

        # for now, we hard coded this network
        # i.e. we fix the number of hidden layers i.e. 2 layers
        self.fc1 = Linear(input_dim, hidden_dims[0])
        self.fc2 = Linear(hidden_dims[0], hidden_dims[1])

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x


# class SimpleCNN(Module):
#     def __init__(self, num_classes: int = 10, **kwargs):
#         super(SimpleCNN, self).__init__()
#
#         self.features = Sequential(
#             Conv2d(3, 6, (5, 5)),
#             ReLU(),
#             MaxPool2d((2, 2)),
#             Conv2d(6, 16, (5, 5)),
#             ReLU(),
#             MaxPool2d((2, 2))
#         )
#         self.fc_1 = Linear(16 * 5 * 5, 120)
#         self.fc_2 = Linear(120, 84)
#         self.logit = Linear(84, num_classes)
#
#         self.fc_list = [self.fc_1, self.fc_2]
#
#         if 'threshold' in kwargs:
#             self.threshold = kwargs['threshold']
#         else:
#             self.threshold = 0
#
#     def forward(self, x):
#         x = self.features(x)
#         features = torch.flatten(x, 1)
#
#         for i, layer in enumerate(self.fc_list):
#             features = F.relu(layer(features))
#
#         logit = self.logit(features)
#         return logit, features
