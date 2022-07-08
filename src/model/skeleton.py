import torch
import torch.nn as nn
from src import *


class FederatedModel(nn.Module):
    def make_empty_weights(self):
        return {k: v.cpu()-v.cpu() for k, v in self.state_dict().items()}

    def set_parameters(self, state_dict: Union[OrderedDict, dict]) -> None:
        self.load_state_dict(state_dict, strict=True)

    def get_parameters(self, ordict: bool = True) -> Union[OrderedDict, list]:
        if ordict:
            return OrderedDict({k: v.clone().detach().cpu() for k, v in self.state_dict().items()})
        else:
            return [val.clone().detach().cpu() for _, val in self.state_dict().items()]

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)
