from . import *
from .utils import *
from src.methods.federated import run


@ray.remote(max_calls=1)
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

    original_state = F.get_parameters(model)

    # INFO - Optimizer
    optimizer = call_optimizer(training_settings['optim'])

    # INFO - Optimizations
    if training_settings['optim'].lower() == 'sgd':
        optim = optimizer(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=training_settings['local_lr'],
                          momentum=training_settings['momentum'])
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
            labels = y.to(device).to(torch.long)

            model.train()
            model.to(device)

            optim.zero_grad()

            outputs = model(inputs)
            loss = loss_fn(outputs, labels.to(torch.long))

            loss.backward()
            optim.step()

            current_state = F.get_parameters(model)

            new_state = F.Constrainting(original_state,current_state)

            model.load_state_dict(new_state, strict=True)

            # INFO - Step summary
            training_loss += loss.item()

            client.step_counter += 1
            summary_counter += 1

            if summary_counter % training_settings["summary_count"] == 0:
                training_acc, _ = F.compute_accuracy(model, client.train_loader, loss_fn)
                summary_writer.add_scalar('step_loss', training_loss / summary_counter, client.step_counter)
                summary_writer.add_scalar('step_acc', training_acc, client.step_counter)
                summary_counter = 0
                training_loss = 0

        # INFO - Epoch summary
        test_acc, test_loss = F.compute_accuracy(model, client.test_loader, loss_fn)
        train_acc, train_loss = F.compute_accuracy(model, client.train_loader, loss_fn)



        summary_writer.add_scalar('epoch_loss/train', train_loss, client.epoch_counter)
        summary_writer.add_scalar('epoch_loss/test', test_loss, client.epoch_counter)


        ## Hessian info
        # F.mark_hessian(model, client.test_loader, summary_writer, client.epoch_counter)


        summary_writer.add_scalar('epoch_acc/local_train', train_acc, client.epoch_counter)
        summary_writer.add_scalar('epoch_acc/local_test', test_acc, client.epoch_counter)

        # F.mark_accuracy(client, model, summary_writer)
        # F.mark_entropy(client, model, summary_writer)
        F.mark_cosine_similarity(current_state,original_state,summary_writer,client.epoch_counter)
        F.mark_norm_size(current_state,summary_writer,client.epoch_counter)

        client.epoch_counter += 1

    # INFO - Local model update
    client.model = OrderedDict({k: v.clone().detach().cpu() for k, v in model.state_dict().items()})
    return client


def run_fedconst(client_setting: dict, training_setting: dict):
    run(client_setting, training_setting, train_fnc=train)
