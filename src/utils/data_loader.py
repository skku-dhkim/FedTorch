import os
from abc import ABC
from typing import Dict, Optional, Any

import pandas as pd
from torchvision.datasets import *
from torchvision.transforms import *
from torch.utils.data import Dataset, DataLoader

import seaborn as sns
import numpy as np
import torch
import random


class CustomDataLoader:
    def __init__(self, train_data, test_data, log_path):
        self.train_X = train_data.data
        self.train_Y = np.array(train_data.targets)

        self.test_X = test_data.data
        self.test_Y = np.array(test_data.targets)

        self.num_of_categories = len(train_data.classes)
        self.categories_train_X, self.categories_train_Y = None, None

        self.log_path = os.path.join(log_path, "client_meta")
        os.makedirs(self.log_path, exist_ok=True)

    def _data_sampling(self, dirichlet_alpha: float, num_of_clients: int, num_of_classes: int) -> pd.DataFrame:
        """
        :param dirichlet_alpha: (float) adjusting iid-ness. higher the alpha, more iid-ness distribution.
        :param num_of_clients: (int) number of clients
        :param num_of_classes: (int) number of classes
        :return:
            DataFrame: Client data distribution for iid-ness.
        """
        # Get dirichlet distribution
        s = np.random.dirichlet(np.repeat(dirichlet_alpha, num_of_clients), num_of_classes)
        c_dist = pd.DataFrame(s)

        # Round for data division convenience.
        c_dist = c_dist.round(3)

        # To plot
        sns.set(rc={'figure.figsize': (10, 10)})
        ax = sns.heatmap(c_dist, cmap='YlGnBu', annot=True)
        ax.set(xlabel='Clients', ylabel='Classes')
        figure = ax.get_figure()

        # Save to Image
        figure.savefig(os.path.join(self.log_path, 'client_meta.png'), format='png')

        return c_dist.transpose()

    def _data_proportion_allocate(self, clients: list, proportion: pd.DataFrame) -> list:
        """
        :param clients: (list) Client lists
        :param proportion: (DataFrame) Data proportion for every client on every labels.
        :return:
            list: Train dataset for every client.
        """
        # Initialize index manager. This is for start and last index managing.
        idx_manage = {}
        for i in range(proportion.shape[1]):
            idx_manage[i] = 0

        # Start allocating data
        for idx, client in enumerate(clients):
            distribution = proportion.iloc[idx]
            for k, dist in enumerate(distribution):
                num_of_data = int(len(self.categories_train_X[k]) * dist)
                client['train']['x'].append(self.categories_train_X[k][idx_manage[k]:idx_manage[k] + num_of_data])
                client['train']['y'].append(self.categories_train_Y[k][idx_manage[k]:idx_manage[k] + num_of_data])
                # Update Last index number. It will be first index at next iteration.
                idx_manage[k] = idx_manage[k] + num_of_data

            # Make an integrated array.
            client['train']['x'] = np.concatenate(client['train']['x'])
            client['train']['y'] = np.concatenate(client['train']['y'])

            # Make random index list
            index = [j for j in range(len(client['train']['x']))]
            random.shuffle(index)

            client['train']['x'] = client['train']['x'][index]
            client['train']['y'] = client['train']['y'][index]
            client['test']['x'] = self.test_X
            client['test']['y'] = self.test_Y

        return clients

    def _categorize(self, x: np.ndarray, y: np.ndarray) -> tuple:
        """
        :param x: (numpy.ndarray) input x
        :param y: (numpy.ndarray) label y
        :return:
            tuple: (categories_X: dict, categories_Y: dict)
        """
        categories_X: Dict[int, Optional[Any]] = {}
        categories_Y: Dict[int, Optional[Any]] = {}

        for i in range(self.num_of_categories):
            # Get category index
            category_index = np.where(y == i)[0]
            categories_X[i] = x[category_index]
            categories_Y[i] = y[category_index]

            # Make random index list
            index = [j for j in range(len(categories_X[i]))]
            random.shuffle(index)

            # Apply random shuffling with same index number
            categories_X[i] = categories_X[i][index]
            categories_Y[i] = categories_Y[i][index]

        return categories_X, categories_Y

    def load(self, number_of_clients: int, dirichlet_alpha: float) -> tuple:
        """
        :param number_of_clients: (int) Number of client to join federated learning.
        :param dirichlet_alpha: (float) Dirichlet distribution alpha. Greater the value more iid-ness data distribution.
        :return:
            tuple: (list: Client data set with non-iid setting, DataLoader: Test set loader)
        """
        # 1. Client definition and matching classes
        clients = [{'train': {'x': [], 'y': []}, 'test': {'x': [], 'y': []}} for _ in range(number_of_clients)]

        # 2. Categorization of dataset
        self.categories_train_X, self.categories_train_Y = self._categorize(self.train_X, self.train_Y)

        # 3. Get data separation distribution
        client_distribution = self._data_sampling(dirichlet_alpha=dirichlet_alpha,
                                                  num_of_clients=number_of_clients,
                                                  num_of_classes=self.num_of_categories)

        # 4. Data allocation
        federated_dataset = self._data_proportion_allocate(clients, proportion=client_distribution)

        test_loader = DataLoader(DatasetWrapper({'x': self.test_X, 'y': self.test_Y}))

        return federated_dataset, test_loader


class FedMNIST(CustomDataLoader):
    def __init__(self, log_path):
        train_data = MNIST(
            root="./data",
            train=True,
            download=True,
        )
        test_data = MNIST(
            root="./data",
            train=False,
            download=True,
        )

        CustomDataLoader.__init__(self, train_data, test_data, log_path)


class FedCifar(CustomDataLoader):
    def __init__(self, mode, log_path):
        if mode == 'cifar-10':
            normalize = Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                  std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

            train_data = CIFAR10(
                root="./data",
                train=True,
                download=True,
                transform=Compose([
                    ToTensor(),
                    normalize
                ])
            )
            test_data = CIFAR10(
                root="./data",
                train=False,
                download=True,
                transform=Compose([
                    ToTensor(),
                    normalize
                ])
            )
        elif mode == 'cifar-100':
            normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                             std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
            train_data = CIFAR100(
                root="./data",
                train=True,
                download=True,
                transform=Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15),
                    transforms.ToTensor(),
                    normalize
                ])
            )
            test_data = CIFAR100(
                root="./data",
                train=False,
                download=True,
                transform=Compose([
                    transforms.ToTensor(),
                    normalize
                ])
            )
        else:
            raise ValueError("Invalid Parameter \'{}\'".format(mode))

        CustomDataLoader.__init__(self, train_data, test_data, log_path)


class DatasetWrapper(Dataset, ABC):
    def __init__(self, data):
        self.data_x = torch.Tensor(data['x'])
        self.data_y = data['y']

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, item):
        return {'x': self.data_x[item], 'y': self.data_y[item]}
