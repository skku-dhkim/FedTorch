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


#input_dim = 256
#hidden = 120*84
#num_feature = 84
#out_dim=10


##classifier part return, source, label , representation
class ModelFedCon(nn.Module):

    def __init__(self,  out_dim, n_classes, net_configs=None):
        super(ModelFedCon, self).__init__()

        self.features = SimpleCNN_header(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=n_classes)
        num_ftrs = 84

        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, out_dim)

        # last layer
        self.l3 = nn.Linear(out_dim, n_classes)

    def forward(self, x):
        h = self.features(x)
        #print("h before:", h)
        #print("h size:", h.size())
        h = h.squeeze() ####representation tensor
        #print("h after:", h)
        x = self.l1(h)
        x = F.relu(x)
        x = self.l2(x)

        y = self.l3(x)
        return y



###cnn part producing representation
class SimpleCNN_header(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNN_header, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # for now, we hard coded this network
        # i.e. we fix the number of hidden layers i.e. 2 layers
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        #self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):

        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        # x = self.fc3(x)
        return x