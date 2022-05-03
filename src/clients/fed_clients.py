import collections
import copy
from collections import OrderedDict
from typing import Optional

from torch.nn import Module
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.utils.data_loader import DatasetWrapper


class Client:
    def __init__(self, client_name: str, data: dict):
        # NOTE: Client Meta setting
        self.name = client_name

        # NOTE: Data settings
        self.train = data['train']
        self.dataset = DatasetWrapper(data=self.train)
        self.train_loader = None

        # NOTE: Training settings
        self.model: Optional[Module] = None
        self.global_iter = 0
        self.weight_changes = None

        # NOTE: Evaluation settings
        self.training_loss = 0.0

        if test_data:
            self.test_dataset = test_data

    def backup_original_weights(self):
        # NOTE: Copy originals
        self.weight_changes = copy.deepcopy(self.model.state_dict())

    def get_change_weights(self):
        for param in self.model.state_dict():
            self.weight_changes[param] = self.model.state_dict()[param] - self.weight_changes[param]

    def train_steps(self, loss_fn, optimizer, epochs, experiment_name):
        writer = SummaryWriter("{}/{}/{}".format(summary_log_path, experiment_name, self.name))

        # NOTE: Get original weights from model
        self.backup_original_weights()

        # NOTE: Train steps
        for epoch in range(epochs):
            for i, data in enumerate(self.train_loader, 0):
                inputs = data['x']
                labels = data['y']

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

                if i+1 == len(self.train_loader):
                    # NOTE: Summary the losses
                    self.training_loss += loss.item()
                    global_count = self.global_iter * epochs + epoch
                    writer.add_scalar('training_loss',
                                      self.training_loss / (global_count + 1), global_count)

                    # NOTE: Summary Accuracy
                    y_max_scores, y_max_idx = outputs.max(dim=1)
                    accuracy = (labels == y_max_idx).sum() / labels.size(0)
                    accuracy = accuracy.item() * 100
                    writer.add_scalar('training_acc', accuracy, global_count)

        # NOTE: Calculate weight changes (Gradient)
        self.get_change_weights()


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
