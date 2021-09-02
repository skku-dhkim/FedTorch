from clients.fed_clients import Client
from torch.utils.tensorboard import SummaryWriter
from conf.logger_config import summary_log_path
import torch


class Trainer:
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name

    def train_steps(self, queue, client: Client, loss_fn, optimizer, epochs, gpu_flag, pid):

        if gpu_flag:
            device = torch.device(f'cuda:{pid}')
        else:
            device = torch.device('cpu')

        torch.cuda.set_device(device)

        writer = SummaryWriter("{}/{}/{}".format(summary_log_path, self.experiment_name, client.name))

        # NOTE: Get original weights from model
        client.backup_original_weights()

        for epoch in range(epochs):
            for i, data in enumerate(client.train_loader, 0):
                inputs = data['x']
                labels = data['y']

                optimizer.zero_grad()
                outputs = client.model(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

                if i+1 == len(client.train_loader):
                    # NOTE: Summary the losses
                    client.training_loss += loss.item()
                    global_count = client.global_iter * epochs + epoch
                    writer.add_scalar('training_loss',
                                      client.training_loss / (global_count + 1), global_count)

                    # NOTE: Summary Accuracy
                    y_max_scores, y_max_idx = outputs.max(dim=1)
                    accuracy = (labels == y_max_idx).sum() / labels.size(0)
                    accuracy = accuracy.item() * 100
                    writer.add_scalar('training_acc', accuracy, global_count)

        # NOTE: Calculate weight changes (Gradient)
        client.get_change_weights()

        # NOTE: Note put result into multiprocessing queue
        queue.put(client)
