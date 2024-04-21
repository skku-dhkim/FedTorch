import torch.nn

from src.methods import *
from .utils import *
from src.methods.federated import run
from src.clients import Client
from src.train.train_utils import client_gradient, compute_layer_norms


@ray.remote(max_calls=1)
def train(
        client: Client,
        training_settings: dict,
        num_of_classes: int):
    # TODO: Need to check is_available() is allowed.
    device = "cuda" if torch.cuda.is_available() is True else "cpu"

    # INFO: Unblock the code if you use M1 GPU
    # device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    summary_writer = SummaryWriter(os.path.join(client.summary_path, "summaries"))

    # INFO - Call the model architecture and set parameters.
    model = model_call(training_settings['model'], num_of_classes, data_type=client.data_type)
    model.load_state_dict(client.model)
    model = model.to(device)

    # INFO - Optimizer
    optimizer = call_optimizer(training_settings['optim'])

    # INFO - Optimizations
    if training_settings['optim'].lower() == 'sgd':
        optim = optimizer(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=training_settings['local_lr'],
                          momentum=training_settings['momentum'],
                          weight_decay=training_settings['weight_decay'])
    else:
        optim = optimizer(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=training_settings['local_lr'])

    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    # INFO: Local training logic
    for _ in range(training_settings['local_epochs']):
        training_loss = 0
        summary_counter = 0

        # INFO: Training steps
        for x, y in client.train_loader:
            inputs = x.to(device)
            labels = y.to(device).to(torch.long)

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

        # INFO - Epoch summary
        test_acc, test_loss = F.compute_accuracy(model, client.test_loader, loss_fn)
        train_acc, train_loss = F.compute_accuracy(model, client.train_loader, loss_fn)

        summary_writer.add_scalar('acc/local_train', train_acc, client.epoch_counter)
        summary_writer.add_scalar('acc/local_test', test_acc, client.epoch_counter)

        summary_writer.add_scalar('loss/local_train', train_loss, client.epoch_counter)
        summary_writer.add_scalar('loss/local_test', test_loss, client.epoch_counter)

        client.epoch_counter += 1

    # if training_settings['aggregator'].lower() == 'balancer':
    #     client.model_norm = compute_layer_norms(model)

    # INFO - Local model update
    client.grad = client_gradient(previous=client.model, current=model.state_dict())
    client.model = OrderedDict({k: v.clone().detach().cpu() for k, v in model.state_dict().items()})

    # INFO - For the Aggregation Balancer
    if training_settings['aggregator'].lower() == 'balancer':
        client.grad_norm = compute_layer_norms(client.grad)

    return client


def run_fedavg(client_setting: dict, training_setting: dict):
    run(client_setting, training_setting, train_fnc=train)
