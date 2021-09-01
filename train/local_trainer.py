from clients.fed_clients import Client
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name

    def train_steps(self, client: Client, loss_fn, optimizer, epochs):
        writer = SummaryWriter("./logs/{}/{}".format(self.experiment_name, client.name))

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

