import torch
import copy
import os
import ray

from torch.utils.tensorboard import SummaryWriter
from src.clients.fed_clients import FedClient
from src.model.skeleton import FederatedModel
from torch.utils.data import DataLoader
from collections import OrderedDict
from src.model import model_manager
from tqdm import tqdm
from typing import Optional


@ray.remote
class Trainer:
    def __init__(self, log_path: str, model: FederatedModel, test_loader: Optional[DataLoader] = None):
        self.summary_path = os.path.join(log_path, "tensorboard")
        self.model = copy.deepcopy(model)
        self.test_loader = test_loader

    def train(self, client: FedClient, device: torch.device) -> FedClient:

        # 1. Set global model
        self.model.load_state_dict(client.model)
        self.model.to(device)

        # Tensorboard Summary writer
        writer = SummaryWriter(os.path.join(self.summary_path, "client{}".format(client.name)))

        # TODO: Various optimization function should be implemented future.
        if client.training_settings['optim'].lower() == 'sgd':
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                                        lr=client.training_settings['local_lr'],
                                        momentum=client.training_settings['momentum'],
                                        weight_decay=1e-5)
        else:
            raise NotImplementedError("Other optimization is not implemented yet")

        # TODO: Various loss function should be implemented future.
        loss_fn = torch.nn.CrossEntropyLoss().to(device)

        # Log train and test set accuracy before training.
        # NOTE: You can make it to comments if you don't need logs
        # train_acc = self.compute_accuracy(data_loader=client.train_loader)
        # print("Train ACC Before training - {}: {:.2f}".format(client.name, train_acc))
        # test_acc = self.compute_accuracy(data_loader=self.test_loader)
        # print("Test ACC Before training - {}: {:.2f}".format(client.name, test_acc))

        # 3. Training logic
        for epoch in range(client.training_settings['local_epochs']):
            training_loss = 0
            training_acc = 0
            counter = 0
            # Training steps
            for data in client.train_loader:
                inputs = data['x'].to(device)
                labels = data['y'].to(device)

                optimizer.zero_grad()
                inputs.requires_grad = False
                labels.requires_grad = False

                outputs = self.model(inputs)
                loss = loss_fn(outputs, labels)

                loss.backward()
                optimizer.step()

                # Summary Loss
                training_loss += loss.item()
                training_acc += self.compute_accuracy(x=inputs, y=labels)

                counter += 1

            writer.add_scalar('training_loss',
                              training_loss / counter,
                              client.training_settings['local_epochs'] * client.global_iter + epoch)
            writer.add_scalar('training_acc',
                              training_acc / counter,
                              client.training_settings['local_epochs'] * client.global_iter + epoch)

        writer.close()

        # Log train and test set accuracy before training.
        # NOTE: You can make it to comments if you don't need logs
        # train_acc = self.compute_accuracy(data_loader=client.train_loader)
        # print("Train ACC After training - {}: {:.2f}".format(client.name, train_acc))
        # test_acc = self.compute_accuracy(data_loader=self.test_loader)
        # print("Test ACC After training - {}: {:.2f}".format(client.name, test_acc))

        client.global_iter += 1
        client.model = self.model.get_weights()

        return client

    def compute_accuracy(self, x: Optional[torch.Tensor] = None,
                         y: Optional[torch.Tensor] = None,
                         data_loader: Optional[DataLoader] = None) -> float:
        """
        :param x: (torch.Tensor) input x
        :param y: (torch.Tensor) label y
        :param data_loader: (torch.data.DataLoader) data loader class
        :return:
            float: accuracy of input data.
        """
        model = copy.deepcopy(self.model)
        model.to('cpu')
        with torch.no_grad():
            if data_loader is not None:
                correct = []
                total = []
                for data in data_loader:
                    x = data['x'].to('cpu')
                    y = data['y'].to('cpu')
                    outputs = model(x).to('cpu')
                    y_max_scores, y_max_idx = outputs.max(dim=1)
                    correct.append((y == y_max_idx).sum().item())
                    total.append(len(x))
                training_acc = sum(correct) / sum(total)
            else:
                outputs = model(x.to('cpu'))
                labels = y.to('cpu')
                y_max_scores, y_max_idx = outputs.max(dim=1)
                training_acc = (labels == y_max_idx).sum().item()
                total = len(y)
                training_acc = training_acc / total
        return training_acc

