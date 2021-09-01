import copy
import torch

from model import model_manager
from typing import Optional


class Aggregator:
    def __init__(self, model_name: str, lr: float = 0.01):
        self.global_model = model_manager.get_model(model_name)
        self._global_weights: Optional[torch.nn.Module] = None
        self.lr = lr

    def model_assignment(self, clients: list):
        # Copy the model to client
        for client in clients:
            client.model = copy.deepcopy(self.global_model)

    def __make_empty_model(self):
        self._global_weights = copy.deepcopy(self.global_model)
        for param in self._global_weights.parameters():
            param.data -= param.data

    def fedAvg(self, clients: list):
        # Get the total data size
        total_data_len = 0
        for client in clients:
            total_data_len += len(client.train['x'])

        self.__make_empty_model()

        # TODO FedAvg - version 1
        # # NOTE: FedAvg - weight collection
        # for client in clients:
        #     for name, param in self._global_weights.named_parameters():
        #         param.data += self.lr * ((len(client.train['x']) / total_data_len) * client.model.state_dict()[name])
        #     client.global_iter += 1
        #
        # self.global_model.load_state_dict(self._global_weights.state_dict())

        # TODO FedAvg - version 2
        # NOTE: FedAvg - weight collection
        for client in clients:
            for name, param in self._global_weights.named_parameters():
                param.data += ((len(client.train['x']) / total_data_len) * client.weight_changes[name])
            client.global_iter += 1

        for name, param in self.global_model.named_parameters():
            param.data = param.data + (self.lr * self._global_weights.state_dict()[name])

    def evaluation(self, test_data):
        outputs = self.global_model(test_data['x'])
        labels = test_data['y']
        y_max_scores, y_max_idx = outputs.max(dim=1)
        accuracy = (labels == y_max_idx).sum() / labels.size(0)
        accuracy = accuracy.item() * 100
        # print("Global Acc: {}".format(accuracy))
        return accuracy
