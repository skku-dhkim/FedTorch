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

    def evaluation(self, device):
        outputs = self.global_model(torch.Tensor(self.test_data['x']).to(device))
        labels = torch.Tensor(self.test_data['y']).to(device)
        y_max_scores, y_max_idx = outputs.max(dim=1)
        accuracy = (labels == y_max_idx).sum() / labels.size(0)
        # print(accuracy)
        # accuracy = accuracy.item()
        with SummaryWriter(self.summary_path) as writer:
            writer.add_scalar('global_acc', accuracy.item(), self.global_iter)
        return accuracy
