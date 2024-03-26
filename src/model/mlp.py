from src.model import *
from torch.nn import *


class MLP(Module):
    def __init__(self, num_classes: int = 10, **kwargs):
        super(MLP, self).__init__()
        self.fc1 = Linear(28*28*3, 512)
        self.fc2 = Linear(512, 512)
        self.fc3 = Linear(512, num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        logit = self.fc3(out)
        return logit
