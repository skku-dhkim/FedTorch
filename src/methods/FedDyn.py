from . import *
from .utils import *
from src.methods.federated import run


@ray.remote(max_calls=1)
def train(client: Client, training_settings: dict, num_of_classes: int):
    # TODO: Need to check is_available() is allowed.
    device = "cuda" if torch.cuda.is_available() is True else "cpu"

    # INFO: Unblock the code if you use M1 GPU
    # device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    summary_writer = SummaryWriter(os.path.join(client.summary_path, "summaries"))

    # INFO - Call the model architecture and set parameters.
    model = model_call(training_settings['model'], num_of_classes)
    model.load_state_dict(client.model)
    model = model.to(device)

    original_state = F.get_parameters(model)

    if not hasattr(client, 'gradL'):
        client.gradL = {}
        for k, v in original_state.items():
            client.gradL[k] = torch.zeros_like(v.data)

    # INFO - Optimizer
    optimizer = call_optimizer(training_settings['optim'])

    # INFO - Optimizations
    if training_settings['optim'].lower() == 'sgd':
        optim = optimizer(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=training_settings['local_lr'],
                          momentum=training_settings['momentum'], weight_decay=training_settings['weight_decay'])
    else:
        optim = optimizer(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=training_settings['local_lr'])

    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    # INFO: Local training logic
    for _ in range(training_settings['local_epochs']):
        # INFO: Training steps
        for x, y in client.train_loader:
            inputs = x.to(device)
            labels = y.to(device).to(torch.long)

            model.train()
            model.to(device)

            optim.zero_grad()

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss2, loss3 = 0, 0

            current_state = F.get_parameters(model)

            for k in current_state.keys():
                current_state[k] = current_state[k].to(device)
                original_state[k] = original_state[k].to(device)
                gL = client.gradL[k].clone().detach().to(device)

                loss2 += torch.dot(gL.flatten(), current_state[k].flatten())
                loss3 += torch.sum(torch.pow(current_state[k]-original_state[k], 2))
            loss = loss - loss2 + (training_settings['dyn_alpha']/2) * loss3

            loss.backward()
            optim.step()

            current_state = F.get_parameters(model)

            for k in current_state.keys():
                current_state[k] = current_state[k].to(device)
                original_state[k] = original_state[k].to(device)
                gL = client.gradL[k].clone().detach().to(device)
                client.gradL[k] = gL - training_settings['dyn_alpha'] * (current_state[k] - original_state[k])

        # INFO - Epoch summary
        test_acc, test_loss = F.compute_accuracy(model, client.test_loader, loss_fn)
        train_acc, train_loss = F.compute_accuracy(model, client.train_loader, loss_fn)

        summary_writer.add_scalar('epoch_acc/local_train', train_acc, client.epoch_counter)
        summary_writer.add_scalar('epoch_acc/local_test', test_acc, client.epoch_counter)

        summary_writer.add_scalar('loss/train', train_loss, client.epoch_counter)
        summary_writer.add_scalar('loss/test', test_loss, client.epoch_counter)

        client.epoch_counter += 1

    # INFO - Local model update
    client.model = OrderedDict({k: v.clone().detach().cpu() for k, v in model.state_dict().items()})
    return client


def run_feddyn(client_setting: dict, training_setting: dict):
    run(client_setting, training_setting, train_fnc=train)
