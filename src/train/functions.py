from . import call_optimizer
from src import *
from src.model import model_call
from src.clients import Client, Aggregator
from torch.nn import Module
from torch import nn
from torch.utils.tensorboard import SummaryWriter

import random
import ray
import torch


def model_download(aggregator: Aggregator, clients: dict) -> None:
    """

    Args:
        aggregator: (Aggregator class)
        clients: (dict) client list of Ray Client Actor.

    Returns: None

    """
    model_weights = aggregator.get_parameters()
    for k, client in clients.items():
        client.model = model_weights
    # TODO: Deprecate on next version.
    # ray.get([client.set_parameters.remote(model_weights) for _, client in clients.items()])


def model_collection(clients: dict, aggregator: Aggregator, with_data_len: bool = False) -> None:
    """

    Args:
        clients: (dict) {str: ray Client Actor}
        aggregator: (Aggregator)
        with_data_len: (bool) Data length return if True.

    Returns: None

    """
    collected_weights = {}
    for k, client in clients.items():
        if with_data_len:
            collected_weights[k] = {'weights': client.get_parameters(),
                                    'data_len': client.data_len()}
        else:
            collected_weights[k] = client.get_parameters()
    aggregator.collected_weights = collected_weights


def compute_accuracy(model: Module, loss_fn: nn, data_loader: DataLoader):
    """
    Compute the accuracy using its test dataloader.
    Returns:
        (float) training_acc: Training accuracy of client's test data.
    """
    # TODO: Need to check is_available() is allowed.
    device = "cuda" if torch.cuda.is_available() is True else "cpu"

    model.to(device)
    model.eval()

    correct = []
    loss_list = []
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            loss = loss_fn(outputs, y)
            loss_list.append(loss.item())
            y_max_scores, y_max_idx = outputs.max(dim=1)
            correct.append((y == y_max_idx).sum().item())
        acc = np.average(correct)
        loss = np.average(loss_list)
    return acc, loss


@ray.remote
def train(
        client: Client,
        training_settings: dict,
        num_of_classes: int,
        early_stopping: bool = False):

    # TODO: Need to check is_available() is allowed.
    device = "cuda" if torch.cuda.is_available() is True else "cpu"

    # INFO: Unblock the code if you use M1 GPU
    # device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')

    summary_writer = SummaryWriter(os.path.join(client.summary_path, "summaries"))

    # INFO - Call the model architecture and set parameters.
    model = model_call(training_settings['model'], num_of_classes)
    model.load_state_dict(client.model)
    model = model.to(device)

    # INFO - Optimizer
    optimizer = call_optimizer(training_settings['optim'])

    # INFO - Optimizations
    if training_settings['optim'].lower() == 'sgd':
        optim = optimizer(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=training_settings['local_lr'],
                          momentum=training_settings['momentum'],
                          weight_decay=1e-5)
    else:
        optim = optimizer(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=training_settings['local_lr'])

    # if early_stopping:
    #     # TODO: patience value may be the hyperparameter.
    #     early_stop = EarlyStopping(patience=5, summary_path=client_info['summary_path'], delta=0)
    # else:
    #     early_stop = None

    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    # INFO: Local training logic
    for _ in range(training_settings['local_epochs']):
        training_loss = 0
        summary_counter = 0

        # INFO: Training steps
        for x, y in client.train_loader:
            inputs = x.to(device)
            labels = y.to(device)

            model.train()
            model.to(device)

            optim.zero_grad()

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            loss.backward()
            optim.step()

            # INFO - Step summary
            training_loss += loss.item()

            client.step_counter += 1
            summary_counter += 1

            if summary_counter % training_settings["summary_count"] == 0:
                training_acc, _ = compute_accuracy(model, loss_fn, client.train_loader)
                summary_writer.add_scalar('step_loss', training_loss / summary_counter, client.step_counter)
                summary_writer.add_scalar('step_acc', training_acc, client.step_counter)
                summary_counter = 0
                training_loss = 0

        # INFO - Epoch summary
        test_acc, test_loss = compute_accuracy(model, loss_fn, client.test_loader)
        train_acc, train_loss = compute_accuracy(model, loss_fn, client.train_loader)

        summary_writer.add_scalar('epoch_loss/train', train_loss, client.epoch_counter)
        summary_writer.add_scalar('epoch_loss/test', test_loss, client.epoch_counter)

        summary_writer.add_scalar('epoch_acc/local_train', train_acc, client.epoch_counter)
        summary_writer.add_scalar('epoch_acc/local_test', train_acc, client.epoch_counter)

        client.epoch_counter += 1

    # INFO - Local model update
    client.model = OrderedDict({k: v.clone().detach().cpu() for k, v in model.state_dict().items()})
    return client


def local_training(clients: list,
                   training_settings: dict,
                   num_of_class: int) -> list:
    """
    Args:
        clients: (dict) client ID and Object pair
        training_settings: (dict) Training setting dictionary
        num_of_class: (int) Number of classes
    Returns: (List) Client Object result

    """
    # sampled_clients = random.sample(list(clients.values()), k=int(len(clients.keys()) * sample_ratio))
    ray_jobs = []
    for client in clients:
        if training_settings['use_gpu']:
            ray_jobs.append(train.options(num_gpus=training_settings['gpu_frac']).remote(client,
                                                                                         training_settings,
                                                                                         num_of_class))
        else:
            ray_jobs.append(train.options().remote(client,
                                                   training_settings,
                                                   num_of_class))
    trained_result = []
    while len(ray_jobs):
        done_id, ray_jobs = ray.wait(ray_jobs)
        trained_result.append(ray.get(done_id[0]))

    return trained_result


def client_sampling(clients: dict, sample_ratio: float, global_round: int):
    sampled_clients = random.sample(list(clients.values()), k=int(len(clients.keys()) * sample_ratio))
    for client in sampled_clients:
        client.global_iter.append(global_round)
    return sampled_clients


def update_client_dict(clients: dict, trained_client: list):
    for client in trained_client:
        clients[client.name] = client
    return clients


# TODO: Need to be fixed.
# New Modification 22.07.20
def local_training_moon(clients: dict) -> None:
    """

    Args:
        clients: (dict) {str: ray Client Actor}

    Returns: None

    """
    ray.get([client.train_moon.remote() for _, client in clients.items()])

