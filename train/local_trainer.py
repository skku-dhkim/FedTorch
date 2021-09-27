import torch
import copy
from torch.utils.tensorboard import SummaryWriter
from conf.logger_config import summary_log_path
from collections import OrderedDict
from model import model_manager


class Trainer:
    def __init__(self, experiment_name, model_name):
        self.experiment_name = experiment_name
        self.model = model_manager.get_model(model_name)

    def train(self, client_name, global_iter, training_loss,
              model, train_loader,
              lr, momentum, epochs, device):

        # 1. Set global model
        self.model.load_state_dict(model)

        # NOTE: Tensorboard Summary writer
        writer = SummaryWriter("{}/{}/{}".format(summary_log_path, self.experiment_name, client_name))

        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        loss_fn = torch.nn.CrossEntropyLoss()

        # 2. Get original weights from model
        original_weights = copy.deepcopy(self.model.state_dict())
        self.model.to(device)

        # 3. Training logic
        for epoch in range(epochs):
            for i, data in enumerate(train_loader, 0):
                inputs = data['x'].to(device)
                labels = data['y'].to(device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

                if i + 1 == len(train_loader):
                    # NOTE: Summary the losses
                    training_loss += loss.item()
                    global_count = global_iter * epochs + epoch
                    writer.add_scalar('training_loss',
                                      training_loss / (global_count + 1), global_count)

                    # NOTE: Summary Accuracy
                    y_max_scores, y_max_idx = outputs.max(dim=1)
                    accuracy = (labels == y_max_idx).sum() / labels.size(0)
                    accuracy = accuracy.item() * 100
                    writer.add_scalar('training_acc', accuracy, global_count)

        # 4. Calculate weight changes (Gradient)
        weight_changes = OrderedDict()
        for param in self.model.state_dict():
            weight_changes[param] = self.model.state_dict()[param] - original_weights[param]

        return {'name': client_name, 'weights': weight_changes, 'data_len': len(train_loader)}

