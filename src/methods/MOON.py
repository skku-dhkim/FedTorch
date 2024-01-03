from . import *
from .utils import *
from copy import deepcopy
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


    if not hasattr(client,'prev_model'):
        client.prev_model = deepcopy(model)

    global_model = deepcopy(model)


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

        ###############################################################################
        representations = {}

        # INFO: HELPER FUNCTION FOR FEATURE EXTRACTION
        def get_representations(name):
            def hook(model, input, output):
                representations[name] = output.detach()

            return hook

        # INFO: REGISTER HOOK
        # TODO: Model specific, need to make general for the future.
        model.features.register_forward_hook(get_representations('rep'))
        ###############################################################################


        # INFO: Training steps
        for x, y in client.train_loader:
            inputs = x.to(device)
            labels = y.to(device).to(torch.long)

            inputs.requires_grad = False
            labels.requires_grad = False

            model.train()
            model.to(device)

            optim.zero_grad()

            outputs = model(inputs)

            cos_sim = torch.nn.CosineSimilarity(dim=-1)

            # MOON's reperesentation extraction mechanism


            l_rep = representations['rep'].reshape((inputs.shape[0], -1))

            _ = global_model(inputs)
            g_rep = representations['rep'].reshape((inputs.shape[0], -1))
            posi = cos_sim(l_rep, g_rep).reshape(-1, 1)


            _ = client.prev_model(inputs)
            p_rep = representations['rep'].reshape((inputs.shape[0], -1))
            nega = cos_sim(l_rep, p_rep).reshape(-1, 1)


            logits = torch.cat((posi, nega), dim=1)

            # Hyperparameter
            #temperature = 0.5
            temperature = training_settings['T']
            mu = training_settings['mu']

            logits /= temperature
            targets = torch.zeros(inputs.size(0)).to(device).long()


            loss2 = mu * loss_fn(logits, targets)
            loss1 = loss_fn(outputs, labels)
            loss = loss1 + loss2


            ############################################################

            loss.backward()
            optim.step()

            current_state = F.get_parameters(model)

            ############## for constraint  ############################
            #new_state = F.Constrainting(original_state, current_state)
            #
            #model.load_state_dict(new_state, strict=True)
            ###########################################################


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

        summary_writer.add_scalar('epoch_acc/local_train', train_acc, client.epoch_counter)
        summary_writer.add_scalar('epoch_acc/local_test', test_acc, client.epoch_counter)

        # F.mark_accuracy(client, model, summary_writer)
        # F.mark_entropy(client, model, summary_writer)

        F.mark_cosine_similarity(current_state, original_state, summary_writer, client.epoch_counter)
        F.mark_norm_size(current_state, summary_writer, client.epoch_counter)

        client.epoch_counter += 1

    # INFO - Local model update
    client.model = OrderedDict({k: v.clone().detach().cpu() for k, v in model.state_dict().items()})
    client.prev_model = deepcopy(model)
    return client


def run_moon(client_setting: dict, training_setting: dict):
    run(client_setting, training_setting, train_fnc=train)
