import torch.nn

from src.methods import *
from .utils import *
from src.losses.loss import FeatureBalanceLoss


@ray.remote(max_calls=1)
def train(client: FedBalancerClient, training_settings: dict, num_of_classes: int):
    device = "cuda" if torch.cuda.is_available() is True else "cpu"

    # INFO: Unblock the code if you use M1 GPU
    # device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')

    summary_writer = SummaryWriter(os.path.join(client.summary_path, "summaries"))

    # INFO - Call the model architecture and set parameters.
    model = model_call(training_settings['model'], num_of_classes, threshold=0)
    model.load_state_dict(client.model)
    model = model.to(device)

    # model_g = model_call(training_settings['model'], num_of_classes, threshold=0)
    # model_g.load_state_dict(client.model)
    # model_g = model_g.to(device)
    # model_g.eval()

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

    # loss_fn = torch.nn.CrossEntropyLoss().to(device)
    loss_fn = FeatureBalanceLoss(client.num_per_class,
                                 training_settings['global_iter'], len(client.global_iter)).to(device)

    # INFO: Local training logic
    for _ in range(training_settings['local_epochs']):

        # INFO: Training steps
        for x, y in client.train_loader:
            inputs = x.to(device)
            labels = y.to(device)

            model.train()
            model.to(device)

            optim.zero_grad()

            outputs, feature_map = model(inputs)
            # model_g.train()
            # model_g.to(device)
            # global_outputs, _ = model_g(inputs)

            loss = loss_fn(outputs, labels, feature_map)
            loss.backward()
            optim.step()
            # client.step_counter += 1

        # client.activation_counts = model.activation_count

        client.epoch_counter += 1
        training_acc, training_losses = F.compute_accuracy(model, client.train_loader, loss_fn)
        test_acc, test_losses = F.compute_accuracy(model, client.test_loader, loss_fn)

        # INFO - Epoch summary
        summary_writer.add_scalar('acc/train', training_acc, client.epoch_counter)
        summary_writer.add_scalar('loss/train', training_losses, client.epoch_counter)
        summary_writer.add_scalar('acc/test', test_acc, client.epoch_counter)
        summary_writer.add_scalar('loss/test', test_losses, client.epoch_counter)

        # F.mark_accuracy(model_l=model, model_g=model_g, dataloader=client.train_loader, summary_writer=summary_writer,
        #                 tag='epoch_metric/train_data', epoch=client.epoch_counter)
        #
        # F.mark_accuracy(model_l=model, model_g=model_g, dataloader=client.test_loader, summary_writer=summary_writer,
        #                 tag='epoch_metric/test_data', epoch=client.epoch_counter)

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


def aggregation_balancer(clients: List[FedBalancerClient], aggregator: Aggregator, model_save: bool = False):
    # for i, client in enumerate(clients):
    #     print("Index-{}: Client ID-{}".format(i, client.name))

    # previous_g_model = aggregator.model.state_dict()
    empty_model = OrderedDict((key, []) for key in aggregator.model.state_dict().keys())
    # empty_global_model = OrderedDict((key, None) for key in aggregator.model.state_dict().keys())
    # empty_activation_counter = OrderedDict((key, []) for key in aggregator.model.state_dict().keys())

    # INFO: Collect the weights from all client in a same layer.
    for k, v in aggregator.model.state_dict().items():
        for client in clients:
            empty_model[k].append(client.model[k])
        empty_model[k] = torch.stack(empty_model[k])

    # INFO: Collect the activation counter
    # for k, v in aggregator.model.state_dict().items():
    #     if 'fc' in k and 'weight' in k:
    #         for client in clients:
    #             # aggregator.summary_writer.add_histogram('Client-{}-ActCounter/{}'.format(client.name, k),
    #             #                                         client.activation_counts[k], aggregator.global_iter)
    #             normalized = F.min_max_normalization(client.activation_counts[k])
    #             empty_activation_counter[k].append(normalized)
    #             # empty_activation_counter[k].append(client.activation_counts[k])
    #         empty_activation_counter[k] = torch.stack(empty_activation_counter[k])
    # for k, v in empty_activation_counter.items():
    #     v = torch.Tensor(v)
    #     sorted_v, idx = v.sort(axis=0)
    #
    # TODO: Compare the weight differences between global and local classifier
    fc_importance_score = None

    for name, v in empty_model.items():
        if 'fc' in name:
            if fc_importance_score is None:
                fc_importance_score = calculate_filter_importance(v, type='prob', layer=name)
                empty_model[name] = torch.sum(fc_importance_score * v, dim=0)
            else:
                # INFO: Same for bias
                fc_importance_score = fc_importance_score.squeeze(1)
                empty_model[name] = torch.sum(fc_importance_score * v, dim=0)
                fc_importance_score = None
        else:
            # NOTE: Averaging the Feature extractor and the logit.
            empty_model[name] = torch.mean(v, 0)

    # empty_model = OrderedDict()
    # total_len = 0
    # # INFO: Original FedAvg
    # for client in clients:
    #     total_len += client.data_len()
    #
    # for k, v in aggregator.model.state_dict().items():
    #     for client in clients:
    #         if k not in empty_model.keys():
    #             # empty_model[k] = client.model[k] * (client.data_len() / total_len) * global_lr
    #             empty_model[k] = client.model[k] * (1 / 10) * global_lr
    #
    #         else:
    #             empty_model[k] += client.model[k] * (1 / 10) * global_lr
    #             # empty_model[k] = client.model[k] * (client.data_len() / total_len) * global_lr

    # NOTE: Global model updates
    aggregator.set_parameters(empty_model, strict=True)
    aggregator.global_iter += 1

    aggregator.test_accuracy = aggregator.compute_accuracy()

    # TODO: Adapt in a future.
    aggregator.summary_writer.add_scalar('global_test_acc', aggregator.test_accuracy, aggregator.global_iter)

    if model_save:
        aggregator.save_model()


def calculate_filter_importance(vector, **kwargs):
    weight_vec = vector.view(vector.size()[0], -1)
    norm_vector = weight_vec.norm(p=2, dim=-1)
    norm_vector = 1 / norm_vector

    if 'prob' in kwargs['type']:
        base = torch.sum(norm_vector, dim=0)
        norm_vector = norm_vector/base
    elif 'softmax' in kwargs['type']:
        norm_vector = torch.softmax(norm_vector, dim=0)
    else:
        pass

    if 'features' in kwargs['layer']:
        return norm_vector.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    else:
        return norm_vector.unsqueeze(-1).unsqueeze(-1)


def run(client_setting: dict, training_setting: dict, experiment_name: str,
        b_save_model: bool = False, b_save_data: bool = False):
    stream_logger, _ = get_logger(LOGGER_DICT['stream'])
    summary_logger, _ = get_logger(LOGGER_DICT['summary'])

    # INFO - Dataset creation
    fed_dataset, valid_loader, test_loader = data_preprocessing(client_setting)

    # INFO - Client initialization
    client = FedBalancerClient
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
            aggregation_balancer(trained_clients, aggregator, training_setting['global_lr'])
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
