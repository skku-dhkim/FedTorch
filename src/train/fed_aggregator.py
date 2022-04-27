import copy
import os

import numpy as np
import warnings
import ray
import torch

from src.model import model_manager
from collections import OrderedDict
from src.train.similarity import cosin_clustering, uclidean_clustering
from torch.utils.tensorboard import SummaryWriter

# warnings.filterwarnings("ignore")


# @ray.remote(num_cpus=1)
class Aggregator:
    def __init__(self, test_data, model_name: str, global_lr: float, log_path: str):
        self.global_model = model_manager.get_model(model_name)
        self.lr = global_lr
        # self.total_data_len = 0
        self.empty_model = copy.deepcopy(self.global_model.state_dict())
        self.test_data = test_data
        self.summary_path = os.path.join(log_path, "tensorboard/global")
        self.global_iter = 0

    def fedAvg(self, collected_weights):
        empty_model = self.global_model.make_empty_weights()

        for k, v in empty_model.items():
            for client in collected_weights:
                empty_model[k] += self.lr * (client[k] / len(collected_weights))

        self.global_model.set_weights(empty_model)
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

    # def get_weights(self, deep_copy=False):
    #     if deep_copy:
    #         return copy.deepcopy(self.global_model.state_dict())
    #     return self.global_model.state_dict()
    #
    # def set_weights(self, weights):
    #     self.global_model.load_state_dict(weights)

    # def cosin_clustering(self, weights, original_shape):
    #     kmeans = KMeans(n_clusters=original_shape[1])
    #     lables = kmeans.fit_predict(weights)
    #     # lables = kmeans.labels_
    #     centers = kmeans.cluster_centers_
    #     print(lables)
    #     print(centers)
    #     # print(centers)
    #     # _csr_matrix = csr_matrix(weights)
    #     # spherical_kmeans = SphericalKMeans(
    #     #     n_clusters=original_shape[1],
    #     #     max_iter=10,
    #     #     verbose=1,
    #     #     init='similar_cut',
    #     #     sparsity='minimum_df',
    #     #     minimum_df_factor=0.05
    #     # )
    #     # labels = spherical_kmeans.fit_predict(_csr_matrix)
    #     # print(labels)
    #     print("--"*10+"Next"+"--"*10)
