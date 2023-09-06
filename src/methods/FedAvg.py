from . import *
from src.model import NUMBER_OF_CLASSES
from .utils import *
from .FedBalancer import aggregation_balancer
from src.clients import FedBalancerClient, AggregationBalancer


@ray.remote(max_calls=1)
def train(
        client: Client,
        training_settings: dict,
        num_of_classes: int):
    # TODO: Need to check is_available() is allowed.
    device = "cuda" if torch.cuda.is_available() is True else "cpu"

    # INFO: Unblock the code if you use M1 GPU
    # device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')

    summary_writer = SummaryWriter(os.path.join(client.summary_path, "summaries"))

    # INFO - Call the model architecture and set parameters.
    model = model_call(training_settings['model'], num_of_classes, features=False)
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
                          momentum=training_settings['momentum'])
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

        # INFO - CoS similarity of evaluation.
        cos_similarity = torch.nn.CosineSimilarity(dim=-1)

        model_state = model.state_dict()
        model_g_state = model_g.state_dict()

        for key, value in model_state.items():
            flatten_model = value.view(-1)
            flatten_g_model = model_g_state[key].view(-1)
            similarity = cos_similarity(flatten_model.cpu(), flatten_g_model.cpu())
            summary_writer.add_histogram("{}/cos_sim".format(key), similarity, len(client.global_iter))

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


def fed_avg(clients: List[Client], aggregator: Aggregator, global_lr: float, model_save: bool = False):
    total_len = 0
    empty_model = OrderedDict()

    for client in clients:
        total_len += client.data_len()

    for k, v in aggregator.model.state_dict().items():
        for client in clients:
            if k not in empty_model.keys():
                empty_model[k] = client.model[k] * (client.data_len() / total_len) * global_lr
            else:
                empty_model[k] += client.model[k] * (client.data_len() / total_len) * global_lr

    # Global model updates
    aggregator.set_parameters(empty_model)
    aggregator.global_iter += 1

    aggregator.test_accuracy = aggregator.compute_accuracy()
    aggregator.summary_writer.add_scalar('global_test_acc', aggregator.test_accuracy, aggregator.global_iter)

    if model_save:
        aggregator.save_model()


def run(client_setting: dict, training_setting: dict, b_save_model: bool = False, b_save_data: bool = False):
    stream_logger, _ = get_logger(LOGGER_DICT['stream'])
    summary_logger, _ = get_logger(LOGGER_DICT['summary'])

    # INFO - Dataset creation
    fed_dataset, valid_loader, test_loader = data_preprocessing(client_setting)

    # INFO - Client initialization
    # if 'client' in client_setting.keys() and client_setting['client'] is True:
    #     client = FedBalancerClient
    # else:
    client = Client

    if training_setting['balancer'] is True:
        aggregator: type(AggregationBalancer) = AggregationBalancer
    else:
        aggregator = Aggregator

    clients, aggregator = client_initialize(client, aggregator,
                                            fed_dataset, test_loader, valid_loader,
                                            client_setting,
                                            training_setting)

    start_runtime = time.time()
    # INFO - Training Global Steps
    try:
        stream_logger.info("[3] Global step starts...")

        pbar = tqdm(range(training_setting['global_epochs']), desc="Global steps #",
                    postfix={'global_acc': aggregator.test_accuracy})

        lr_decay = False
        if 'lr_decay' in training_setting.keys():
            initial_lr = training_setting['local_lr']
            total_g_epochs = training_setting['global_epochs']
            lr_decay = True

        for gr in pbar:
            start_time_global_iter = time.time()

            # INFO - Save the global model
            aggregator.save_model()

            # INFO - Download the model from aggregator
            stream_logger.debug("[*] Client downloads the model from aggregator...")
            F.model_download(aggregator=aggregator, clients=clients)

            # INFO - Client sampling
            stream_logger.debug("[*] Client sampling...")
            sampled_clients = F.client_sampling(clients, sample_ratio=training_setting['sample_ratio'], global_round=gr)

            # INFO - Learning rate decay
            if lr_decay:
                if 'cos' in training_setting['lr_decay'].lower():
                    # INFO - COS decay
                    training_setting['local_lr'] = 1 / 2 * initial_lr * (
                                1 + math.cos(aggregator.global_iter * math.pi / total_g_epochs))
                    stream_logger.debug("[*] Learning rate decay: {}".format(training_setting['local_lr']))
                    summary_logger.info("[{}/{}] Current local learning rate: {}".format(aggregator.global_iter,
                                                                                         total_g_epochs,
                                                                                         training_setting['local_lr']))
                else:
                    raise NotImplementedError("Learning rate decay \'{}\' is not implemented yet.".format(
                        training_setting['lr_decay']))

            stream_logger.debug("[*] Local training process...")
            trained_clients = local_training(clients=sampled_clients,
                                             training_settings=training_setting,
                                             num_of_class=NUMBER_OF_CLASSES[client_setting['dataset'].lower()])

            stream_logger.debug("[*] Federated aggregation scheme...")
            if training_setting['balancer'] is True:
                stream_logger.debug("[*] Aggregation Balancer")
                aggregation_balancer(trained_clients, aggregator, training_setting['global_lr'], training_setting['T'])
            else:
                stream_logger.debug("[*] FedAvg")
                fed_avg(trained_clients, aggregator, training_setting['global_lr'])

            stream_logger.debug("[*] Weight Updates")
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
