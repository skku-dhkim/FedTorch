from . import *
from .utils import *
from src.methods.federated import run


# TODO: Change all formats
@ray.remote(max_calls=1)
def train(
        client: Client,
        training_settings: dict,
        num_of_classes: int,
        early_stopping: bool = False):
    device = "cuda" if torch.cuda.is_available() is True else "cpu"

    # INFO: Unblock the code if you use M1 GPU
    # device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')

    summary_writer = SummaryWriter(os.path.join(client.summary_path, "summaries"))

    # INFO - Call the model architecture and set parameters.
    model = model_call(training_settings['model'], num_of_classes, data_type=client.data_type)
    model.load_state_dict(client.model)
    model = model.to(device)

    model_g = model_call(training_settings['model'], num_of_classes, data_type=client.data_type)
    model_g.load_state_dict(client.model)
    model_g = model_g.to(device)

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
        _feature_kl_loss = 0
        _outputs_kl_loss = 0

        summary_counter = 0

        # INFO: Training steps
        for x, y in client.train_loader:
            inputs = x.to(device)
            labels = y.to(device)

            model.train()
            model.to(device)

            optim.zero_grad()

            local_outputs = model(inputs)

            model_g.train()
            model_g.to(device)
            global_outputs = model_g(inputs)

            one_hot = F.one_hot_encode(labels.detach(), num_classes=num_of_classes, device=device)

            # TODO: torch.mean() should be included into norm_gap_fn.
            Pc = F.calculate_norm_gap(local_outputs.detach(), one_hot, logit=True, normalize=False, prob=True)
            Pg = F.calculate_norm_gap(global_outputs.detach(), one_hot, logit=True, normalize=False, prob=True)
            entropy = torch.mean(F.entropy(local_outputs.detach(), prob=False, normalize=True, base='exp'))
            _r = torch.mean(Pc) / (torch.mean(Pg) + torch.mean(Pc))
            r = torch.exp(-_r)

            T = training_settings['indicator_temp']
            indicator = torch.sqrt(r / (1 + T * entropy))

            cross_entropy_loss = loss_fn(local_outputs, labels)
            outputs_kl_loss = F.loss_fn_kd(local_outputs, global_outputs.detach(),
                                           alpha=indicator,
                                           temperature=training_settings['kl_temp'])
            loss = (1 - indicator) * cross_entropy_loss + outputs_kl_loss

            loss.backward()
            optim.step()

            # INFO - Step summary
            training_loss += loss.item()

            client.step_counter += 1
            summary_counter += 1

            summary_writer.add_scalar('experiment/r', r, client.step_counter)
            summary_writer.add_scalar('experiment/indicator', indicator, client.step_counter)
            summary_writer.add_scalar('experiment/entropy', entropy, client.step_counter)

        client.epoch_counter += 1

        # INFO - Epoch summary
        F.mark_accuracy(model_l=model, model_g=model_g, dataloader=client.train_loader, summary_writer=summary_writer,
                        tag='epoch_metric/train_data', epoch=client.epoch_counter)

        F.mark_accuracy(model_l=model, model_g=model_g, dataloader=client.test_loader, summary_writer=summary_writer,
                        tag='epoch_metric/test_data', epoch=client.epoch_counter)

        F.mark_entropy(model_l=model, model_g=model_g, dataloader=client.train_loader,
                       summary_writer=summary_writer, epoch=client.epoch_counter)
        F.mark_norm_gap(model_l=model, model_g=model_g, dataloader=client.train_loader,
                        summary_writer=summary_writer, epoch=client.epoch_counter, prob=True)

    # INFO - Local model update
    client.model = OrderedDict({k: v.clone().detach().cpu() for k, v in model.state_dict().items()})
    return client


def run_fedkl(client_setting: dict, training_setting: dict):
    run(client_setting, training_setting, train_fnc=train)
