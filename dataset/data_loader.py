from abc import ABC

from torchvision.datasets import *
from torchvision import transforms
from multipledispatch import dispatch
from torch.utils.data import Dataset

import numpy as np
import torch
import random
import re

torch.manual_seed(77)


@dispatch(int, int, int)
def f_classes_per_client(_number_of_clients: int, _classes_per_client: int, _num_of_categories: int):
    clients = {}

    def __get_shuffled_index(__num_of_categories):
        __class_idx = [x for x in range(__num_of_categories)]
        random.shuffle(__class_idx)
        return __class_idx

    class_idx = __get_shuffled_index(_num_of_categories)

    for idx in range(_number_of_clients):
        client_id = "client_{}".format(idx)
        for count in range(_classes_per_client):
            if len(class_idx) == 0:
                # Init class_idx when list is empty.
                class_idx = __get_shuffled_index(_num_of_categories)
            # Pop one
            target_idx = class_idx.pop()
            # Insert class settings
            if client_id not in clients.keys():
                clients[client_id] = {
                    target_idx: 0
                }
            else:
                clients[client_id][target_idx] = 0
    return clients


@dispatch(int, str, int)
def f_classes_per_client(_number_of_clients: int, _classes_per_client: str, _num_of_categories: int):
    if _classes_per_client.lower() != "random":
        raise ValueError("Invalid \'classes_per_client\' value: {}".format(_classes_per_client))

    clients = {}

    def __random_list(__num_of_categories):
        __random_idx = random.sample(range(0, __num_of_categories), random.randint(1, __num_of_categories))
        return __random_idx

    for idx in range(_number_of_clients):
        random_idx = __random_list(_num_of_categories)
        client_id = "client_{}".format(idx)
        for r_idx in random_idx:
            if client_id not in clients.keys():
                clients[client_id] = {
                    r_idx: 0
                }
            else:
                clients[client_id][r_idx] = 0
    return clients


class DataLoader:
    def __init__(self, train_data, test_data):
        self.train_X = train_data.data.numpy() / 255.0
        self.train_Y = train_data.targets.numpy()
        self.test_X = test_data.data.numpy() / 255.0
        self.test_Y = test_data.targets.numpy()
        self.num_of_categories = len(train_data.classes)

        self.categories_train_X, self.categories_train_Y = None, None

    def __data_proportion_allocate(self, clients: dict, proportion: bool):
        def sampling(__clients, __total_target_len, __total_client_per_target):
            __class_len = len(__total_target_len.keys())
            for __idx in range(__class_len):
                if proportion is not True:
                    # Equal Divide
                    __num_of_classes_per_client = int(__total_target_len[__idx] / __total_client_per_target[__idx])

                    for __client in __clients:
                        if __idx in __clients[__client].keys():
                            __clients[__client][__idx] = __num_of_classes_per_client

                else:
                    # Random divide
                    rnd_dist = np.random.dirichlet(np.ones(__total_client_per_target[__idx]), size=1)
                    dist_index = 0
                    for __client in __clients:
                        if __idx in __clients[__client].keys():
                            __clients[__client][__idx] = int(__total_target_len[__idx] * rnd_dist[0][dist_index])
                            dist_index += 1

            return __clients

        # 1. Count total number of samples per class
        total_target_len = {}
        for i in range(self.num_of_categories):
            total_target_len[i] = 0
        for idx in self.categories_train_Y:
            total_target_len[idx] = len(self.categories_train_Y[idx])

        # 2. Count the number of clients per class
        total_client_per_target = {}
        for i in range(self.num_of_categories):
            total_client_per_target[i] = 0
        for client in clients:
            for idx in clients[client].keys():
                total_client_per_target[idx] += 1

        # 3. Calculate samples per class of each clients.
        clients = sampling(clients, total_target_len, total_client_per_target)

        # 4. Allocate the sample
        federated_dataset = {}
        for client in clients:
            federated_dataset[client] = {'x':  None, 'y': None}

        for i in range(self.num_of_categories):
            start_idx = 0
            temp_repo_x = self.categories_train_X[i].copy()
            temp_repo_y = self.categories_train_Y[i].copy()

            for client in clients:
                # If index 'i' is exist.
                if i in clients[client].keys():
                    last_index = start_idx + clients[client][i]

                    if federated_dataset[client]['x'] is None:
                        federated_dataset[client]['x'] = temp_repo_x[start_idx:last_index]
                        federated_dataset[client]['y'] = temp_repo_y[start_idx:last_index]
                    else:
                        federated_dataset[client]['x'] = np.append(federated_dataset[client]['x'],
                                                                   temp_repo_x[start_idx:last_index], axis=0)
                        federated_dataset[client]['y'] = np.append(federated_dataset[client]['y'],
                                                                   temp_repo_y[start_idx:last_index], axis=0)

                    start_idx = last_index

        for client in federated_dataset:
            federated_dataset[client]['x'] = torch.tensor(federated_dataset[client]['x'], dtype=torch.float)
            federated_dataset[client]['x'] = federated_dataset[client]['x'].permute(0, 3, 1, 2)
            federated_dataset[client]['y'] = torch.tensor(federated_dataset[client]['y'])

        return federated_dataset

    def load(self, number_of_clients, classes_per_client, random_dist=False):
        def categorize(__x, __y, __num_of_categories):
            """
            Get Categorized data value by index number.
            :param __x: (numpy) data X
            :param __y: (numpy) target Y
            :param __num_of_categories: (int) number of classes
            :return: (dict, dict) categorized data X, categorized target Y
            """
            __categories_X = {}
            __categories_Y = {}
            for __i in range(__num_of_categories):
                __category_index = np.where(__y == __i)[0]
                __categories_X[__i] = __x[__category_index]
                __categories_Y[__i] = __y[__category_index]
            return __categories_X, __categories_Y

        # 1. Client definition and matching classes
        clients = f_classes_per_client(number_of_clients, classes_per_client, self.num_of_categories)

        # 2. Categorization of dataset
        self.categories_train_X, self.categories_train_Y = categorize(self.train_X, self.train_Y,
                                                                      self.num_of_categories)
        # 3. Training set separation
        federated_dataset = self.__data_proportion_allocate(clients, proportion=random_dist)

        # 4. Make test data set to Torch tensor
        self.test_X = torch.tensor(self.test_X, dtype=torch.float)
        self.test_X = self.test_X.permute(0, 3, 1, 2)
        self.test_Y = torch.tensor(self.test_Y)

        return federated_dataset, {'x': self.test_X, 'y': self.test_Y}, clients

    def load_original(self):
        x = self.train_X
        x = torch.Tensor(x)
        x = x.permute(0, 3, 1, 2)

        tx = self.test_X
        tx = torch.Tensor(tx)
        tx = tx.permute(0, 3, 1, 2)

        return {'x': x, 'y': self.train_Y}, {'x': tx, 'y': self.test_Y}


class FedMNIST(DataLoader):
    def __init__(self):
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

        DataLoader.__init__(self, train_data, test_data)


class FedCifar(DataLoader):
    def __init__(self):
        train_data = CIFAR10(
            root="./data",
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            # target_transform=ToTensor()
        )
        test_data = CIFAR10(
            root="./data",
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]),
            # target_transform=ToTensor()
        )

        train_data.data = torch.tensor(train_data.data)
        train_data.targets = torch.tensor(train_data.targets)

        test_data.data = torch.tensor(test_data.data)
        test_data.targets = torch.tensor(test_data.targets)

        DataLoader.__init__(self, train_data, test_data)


class DatasetWrapper(Dataset, ABC):
    def __init__(self, data):
        self.data_x = data['x']
        self.data_y = data['y']

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, item):
        return {'x': self.data_x[item], 'y': self.data_y[item]}
