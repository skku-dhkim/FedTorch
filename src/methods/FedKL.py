from src.methods import *
from .utils import *


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
    model = model_call(training_settings['model'], num_of_classes)
    model.load_state_dict(client.model)
    model = model.to(device)

    model_g = model_call(training_settings['model'], num_of_classes)
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


def local_training(clients: list,
                   training_settings: dict,
                   num_of_class: int) -> list:
    """
    Args:
        clients: (list) client list to join the training.
        training_settings: (dict) Training setting dictionary
        num_of_class: (int) Number of classes
    Returns: (List) Client Object result

    """
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


def fed_kl(clients: List[Client], aggregator: Aggregator, global_lr: float, model_save: bool = False):
    total_len = 0
    empty_model = OrderedDict()

    # INFO: Original FedAvg
    for client in clients:
        total_len += client.data_len()

    for k, v in aggregator.model.state_dict().items():
        for client in clients:
            if k not in empty_model.keys():
                empty_model[k] = client.model[k] * (client.data_len() / total_len) * global_lr
            else:
                empty_model[k] += client.model[k] * (client.data_len() / total_len) * global_lr

    # Global model updates
    aggregator.set_parameters(empty_model, strict=False)
    aggregator.global_iter += 1

    aggregator.test_accuracy = aggregator.compute_accuracy()

    # TODO: Adapt in a future.
    # Calculate cos_similarity with previous representations
    # aggregator.calc_rep_similarity()
    #
    # # Calculate cos_similarity of weights
    # current_model = self.get_parameters()
    # self.calc_cos_similarity(original_model, current_model)
    aggregator.summary_writer.add_scalar('global_test_acc', aggregator.test_accuracy, aggregator.global_iter)

    if model_save:
        aggregator.save_model()


def run(client_setting: dict, training_setting: dict, b_save_model: bool = False, b_save_data: bool = False):
    stream_logger, _ = get_logger(LOGGER_DICT['stream'])
    summary_logger, _ = get_logger(LOGGER_DICT['summary'])

    # INFO - Dataset creation
    fed_dataset, valid_loader, test_loader = data_preprocessing(client_setting)

    # INFO - Client initialization
    client = Client
    clients, aggregator = client_initialize(client, fed_dataset, test_loader, valid_loader,
                                            client_setting, training_setting)
    start_runtime = time.time()
    # INFO - Training Global Steps
    try:
        stream_logger.info("[3] Global step starts...")

        pbar = tqdm(range(training_setting['global_iter']), desc="Global steps #",
                    postfix={'global_acc': aggregator.test_accuracy})

        for gr in pbar:
            start_time_global_iter = time.time()

            # INFO - Save the global model
            aggregator.save_model()

            # INFO - Download the model from aggregator
            stream_logger.debug("[*] Client downloads the model from aggregator...")
            F.model_download(aggregator=aggregator, clients=clients)

            stream_logger.debug("[*] Local training process...")
            # INFO - Normal Local Training
            sampled_clients = F.client_sampling(clients, sample_ratio=training_setting['sample_ratio'], global_round=gr)
            trained_clients = local_training(clients=sampled_clients,
                                             training_settings=training_setting,
                                             num_of_class=NUMBER_OF_CLASSES[client_setting['dataset'].lower()])
            stream_logger.debug("[*] Federated aggregation scheme...")
            fed_kl(trained_clients, aggregator, training_setting['global_lr'])
            clients = F.update_client_dict(clients, trained_clients)

            # INFO - Save client models
            if b_save_model:
                save_model(clients)

            end_time_global_iter = time.time()
            pbar.set_postfix({'global_acc': aggregator.test_accuracy})
            summary_logger.info("Global Running time: {}::{:.2f}".format(gr,
                                                                         end_time_global_iter - start_time_global_iter))
            summary_logger.info("Test Accuracy: {}".format(aggregator.test_accuracy))
        summary_logger.info("Global iteration finished successfully.")
    except Exception as e:
        system_logger, _ = get_logger(LOGGER_DICT['system'])
        system_logger.error(traceback.format_exc())
        raise Exception(traceback.format_exc())

    end_run_time = time.time()
    summary_logger.info("Global Running time: {:.2f}".format(end_run_time - start_runtime))

    # INFO - Save client's data
    if b_save_data:
        save_data(clients)
        aggregator.save_data()

    summary_logger.info("Experiment finished.")
    stream_logger.info("Experiment finished.")
