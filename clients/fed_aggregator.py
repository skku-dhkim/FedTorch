import copy
import torch
from model import model_manager
from typing import Optional
from collections import OrderedDict


class Aggregator:
    def __init__(self, model_name: str, lr: float = 0.01):
        self.global_model = model_manager.get_model(model_name)
        self.lr = lr
        self.total_data_len = 0
        self.empty_model = copy.deepcopy(self.global_model.state_dict())

    def __make_empty_model(self):
        for name in self.empty_model:
            self.empty_model[name] -= self.empty_model[name]

    def fedAvg(self, collected_weights):
        # Get the total data size
        total_data_len = 0
        for client in collected_weights:
            total_data_len += client['data_len']

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
        for client in collected_weights:
            for name, param in self.global_model.named_parameters():
                self.empty_model[name] += ((client['data_len'] / total_data_len) * client['weights'][name])

        for name, param in self.global_model.named_parameters():
            param.data = param.data + (self.lr * self.empty_model[name])

    def evaluation(self, test_data):
        outputs = self.global_model(test_data['x'])
        labels = test_data['y']
        y_max_scores, y_max_idx = outputs.max(dim=1)
        accuracy = (labels == y_max_idx).sum() / labels.size(0)
        accuracy = accuracy.item() * 100
        return accuracy

    def get_weights(self, deep_copy=False):
        if deep_copy:
            return copy.deepcopy(self.global_model.state_dict())
        return self.global_model.state_dict()

    def set_weights(self, weights):
        self.global_model.load_state_dict(weights)
