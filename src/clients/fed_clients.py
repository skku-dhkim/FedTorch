import collections
import copy
import os
import torch

from collections import OrderedDict
from typing import Optional

from torch.nn import Module
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src import num_of_classes
from src.utils.data_loader import DatasetWrapper
from src.model import model_call, FederatedModel
from typing import Optional


class Client:
    def __init__(self,
                 client_name: str,
                 data: dict,
                 train_settings: dict,
                 log_path: str):
        # Client Meta setting
        self.name = client_name

        # Data settings
        self.train = data['data']['train']
        self.train_loader = DataLoader(DatasetWrapper(data=self.train),
                                       batch_size=train_settings['batch_size'], shuffle=True)

        self.test = data['data']['test']
        self.test_loader = DataLoader(DatasetWrapper(data=self.test),
                                      batch_size=train_settings['batch_size'], shuffle=True)

        # Training settings
        self.training_settings = train_settings
        self.global_iter = 0

        # Model
        self.model: Optional[FederatedModel] = model_call(train_settings['model'],
                                                           num_of_classes[data['dataset']])

        # Log path
        self.log_path = log_path

    # @property
    # def model(self) -> FederatedModel:
    #     return self._model
    #
    # @model.setter
    # def model(self, v) -> None:
    #     self._model = copy.deepcopy(v)

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: v.clone().detach() for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, ordict=True):
        if ordict:
            return OrderedDict({k: val.clone().detach().cpu() for k, val in self.model.state_dict().items()})
        else:
            return [val.clone().detach().cpu() for _, val in self.model.state_dict().items()]

    def save_model(self):
        save_path = os.path.join(self.log_path, "client_model/{}/{}.pt".format(self.name, self.global_iter))
        torch.save(self.model.state_dict(), save_path)

    def save_data(self):
        save_path = os.path.join(self.log_path, "client_data/{}/{}.pt".format(self.name, self.global_iter))
        torch.save(self.train_loader, save_path)

    def data_len(self):
        return len(self.train['x'])


class FedClient:
    def __init__(self, client_name: str,
                 data: dict,
                 batch_size: int,
                 train_settings: dict):
        # Client Meta setting
        self.name = client_name

        # Data settings
        self.train = data['train']
        self.train_loader = DataLoader(DatasetWrapper(data=self.train), batch_size=batch_size, shuffle=True)

        # Training settings
        self.training_settings = train_settings
        self.global_iter = 0

        # Evaluation settings
        self.training_loss = 0.0

        # Model
        self._model = None

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, v):
        self._model = copy.deepcopy(v)

    def data_len(self):
        return len(self.train['x'])
