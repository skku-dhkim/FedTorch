import torch
import copy
import os
import ray

from torch.utils.tensorboard import SummaryWriter
from src.clients.fed_clients import FedClient
from collections import OrderedDict
from src.model import model_manager
from tqdm import tqdm


# @ray.remote(num_cpus=1, num_gpus=1)
@ray.remote
class Trainer:
    def __init__(self, log_path: str, model_name: str):
        self.summary_path = os.path.join(log_path, "tensorboard")
        self.model = model_manager.get_model(model_name)

    def train(self, client: FedClient, device) -> FedClient:

        # 1. Set global model
        self.model.load_state_dict(client.model)

        # Tensorboard Summary writer
        writer = SummaryWriter(os.path.join(self.summary_path, "client{}".format(client.name)))

        # TODO: Various optimization function should be implemented future.
        if client.training_settings['optim'].lower() == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(),
                                        lr=client.training_settings['local_lr'],
                                        momentum=client.training_settings['momentum'])
        else:
            raise NotImplementedError("Other optimization is not implemented yet")

        # TODO: Various loss function should be implemented future.
        loss_fn = torch.nn.CrossEntropyLoss()

        # 2. Get original weights from model
        self.model.to(device)

        # 3. Training logic
        for epoch in range(client.training_settings['local_epochs']):
            training_loss = 0
            training_acc = 0
            counter = 0
            for i, data in enumerate(client.train_loader):
                inputs = data['x'].to(device)
                labels = data['y'].to(device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

                # Summary Loss
                training_loss += loss.item()
                # Summary Accuracy
                y_max_scores, y_max_idx = outputs.max(dim=1)
                training_acc += ((labels == y_max_idx).sum() / labels.size(0)).item()

                counter += 1

            writer.add_scalar('training_loss',
                              training_loss / counter,
                              client.training_settings['local_epochs'] * client.global_iter + epoch)
            writer.add_scalar('training_acc',
                              training_acc / counter,
                              client.training_settings['local_epochs'] * client.global_iter + epoch)

        client.global_iter += 1
        client.model = self.model.get_weights()
        writer.close()

        return client
