import torch.nn

from src.methods import *
from .utils import *
from src.losses.loss import FeatureBalanceLoss
from src.clients import FedBalancerClient
from src.methods.federated import run
from src.train.train_utils import compute_layer_norms


@ray.remote(max_calls=1)
def train(client: FedBalancerClient, training_settings: dict, num_of_classes: int):
    device = "cuda" if torch.cuda.is_available() is True else "cpu"

    # INFO: Unblock the code if you use M1 GPU
    # device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')

    summary_writer = SummaryWriter(os.path.join(client.summary_path, "summaries"))

    # INFO - Call the model architecture and set parameters.
    model = model_call(training_settings['model'], num_of_classes, features_map=True, data_type=client.data_type)
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

    # INFO - Loss function
    loss_fn = FeatureBalanceLoss(training_settings['global_epochs'], client.num_per_class)

    # INFO: Local training logic
    for i in range(training_settings['local_epochs']):

        # INFO: Training steps
        for x, y in client.train_loader:
            inputs = x.to(device)
            labels = y.to(device)

            model.train()
            model.to(device)

            optim.zero_grad()

            outputs, feature_map = model(inputs)

            loss = loss_fn(outputs, labels, feature_map, i, device)

            loss.backward()
            optim.step()

        training_acc, training_losses = F.compute_accuracy(model, client.train_loader, loss_fn)
        test_acc, test_losses = F.compute_accuracy(model, client.test_loader, loss_fn)

        # INFO - Epoch summary
        summary_writer.add_scalar('acc/local_train', training_acc, client.epoch_counter)
        summary_writer.add_scalar('acc/local_test', test_acc, client.epoch_counter)

        summary_writer.add_scalar('loss/local_train', training_losses, client.epoch_counter)
        summary_writer.add_scalar('loss/local_test', test_losses, client.epoch_counter)

        client.epoch_counter += 1

    # INFO - Local model update
    client.model = OrderedDict({k: v.clone().detach().cpu() for k, v in model.state_dict().items()})

    # INFO - For the Aggregation Balancer
    if training_settings['aggregator'].lower() == 'balancer':
        client.model_norm = compute_layer_norms(model)

    return client


def run_fedbal(client_setting: dict, training_setting: dict):
    run(client_setting, training_setting, train_fnc=train)
