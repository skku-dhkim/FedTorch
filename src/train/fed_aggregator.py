import copy
import os

import numpy as np
import warnings
import ray
import torch
import re

from src.model import model_manager, custom_cnn
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix

# warnings.filterwarnings("ignore")


# @ray.remote(num_cpus=1)
class Aggregator:
    def __init__(self, test_data, model_name: str, num_of_classes: int, global_lr: float, log_path: str):
        self.global_model = model_manager.get_model(model_name, num_of_classes=num_of_classes)
        self.lr = global_lr
        self.test_loader = test_data
        self.summary_path = os.path.join(log_path, "tensorboard/global")
        self.global_iter = 0

    def fedAvg(self, collected_weights):
        empty_model = {}
        total_len = 0

        for _, data_size in collected_weights:
            total_len += data_size

        for k, v in self.global_model.state_dict().items():
            for weights, data_size in collected_weights:
                if k not in empty_model.keys():
                    empty_model[k] = self.lr * weights[k] * (data_size / total_len)
                else:
                    empty_model[k] += self.lr * weights[k] * (data_size / total_len)

        # Global model updates
        self.global_model.load_state_dict(empty_model)
        self.global_iter += 1

    def fedConCat(self, collected_weights, feature_map: bool):
        from src.model.custom_cnn import CustomCNN

        def _findWord(word, txt):
            return re.search("\A{}".format(word), txt)

        if feature_map:
            concat_model = CustomCNN(num_of_clients=len(collected_weights), b_global=True,
                                     n_of_clients=len(collected_weights))
            empty_model = concat_model.make_empty_weights()
            last_layer_name = list(concat_model.features.state_dict().keys())[-1]
            for k, v in concat_model.state_dict().items():
                if _findWord("fc", k) is not None:
                    empty_model[k] = v
                elif k != "features." + last_layer_name:
                    for client in collected_weights:
                        empty_model[k] += self.lr * (client[k] / len(collected_weights))
                else:
                    _w = []
                    for client in collected_weights:
                        _w.append(client[k])
                    empty_model[k] = torch.cat(_w)
            self.global_model = concat_model
            self.global_model.set_weights(empty_model)
            self.global_model.features.requires_grad_(False)
            self.global_model.fc.requires_grad_(True)

        else:
            empty_model = self.global_model.make_empty_weights()

            for k, v in self.global_model.state_dict().items():
                if _findWord("fc", k) is not None:
                    for client in collected_weights:
                        empty_model[k] += self.lr * (client[k] / len(collected_weights))
                else:
                    empty_model[k] = collected_weights[0][k]
            # print(empty_model)
            # print(self.global_model.state_dict())
            self.global_model.set_weights(empty_model)
            self.global_model.features.requires_grad_(False)
            self.global_model.fc.requires_grad_(True)

        self.global_iter += 1



        # for k, v in concat_model.features.state_dict().items():
        #     _w = []
        #     key = "features."+k
        #     for client in collected_weights:
        #         print(client[key].shape)
        #         _w.append(client[key])
        #     empty_model[key] = torch.cat(_w)
        #
        #     print(empty_model[key].shape)

        # concat_model.set_weights(empty_model)
        # print(empty_model[k])
        # empty_model[k] += self.lr * (client[k] / len(collected_weights))

        # self.global_model.features.training = False
        # print(self.global_model.features)
        # print(self.global_model.fc)
        self.global_iter += 1
        # self.global_model = concat_model

    def evaluation(self):
        with torch.no_grad():
            self.global_model.to('cpu')
            accuracy = []
            total = []
            for data in self.test_loader:
                x = data['x'].to('cpu')
                y = data['y'].to('cpu')
                outputs = self.global_model(x)
                y_max_scores, y_max_idx = outputs.max(dim=1)
                accuracy.append((y == y_max_idx).sum().item())
                total.append(len(y))
            accuracy = sum(accuracy) / sum(total)
            with SummaryWriter(self.summary_path) as writer:
                writer.add_scalar('global_acc', accuracy, self.global_iter)
        return accuracy
