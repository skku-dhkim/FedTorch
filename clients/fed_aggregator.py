import torch
import copy
from model import model_manager


class Aggregator:
    def __init__(self, model_name: str):
        self.model = model_manager.get_model(model_name)
        self._global_weights = None

    def model_assignment(self, clients: list):
        # Copy the model to client
        for client in clients:
            client.model = copy.deepcopy(self.model)

        # Make empty weights for accumulating the global weight
        self.__make_empty_model()

    def __make_empty_model(self):
        self._global_weights = self.model.state_dict()
        for param in self._global_weights:
            self._global_weights[param] -= self._global_weights[param]

    def fedAvg(self, clients: list):
        # Get the total data size
        total_data_len = 0
        for client in clients:
            total_data_len += len(client.train['x'])

        # FedAvg
        for client in clients:
            for param in client.model.state_dict():
                self._global_weights[param] += \
                    ((len(client.train['x']) / total_data_len) * client.model.state_dict()[param])

            client.global_iter += 1

    def evaluation(self, test_data):
        outputs = self.model(test_data['x'])
        labels = test_data['y']
        y_max_scores, y_max_idx = outputs.max(dim=1)
        accuracy = (labels == y_max_idx).sum() / labels.size(0)
        accuracy = accuracy.item() * 100
        # print("Global Acc: {}".format(accuracy))
        return accuracy
