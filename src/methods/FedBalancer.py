import torch.nn

from src.methods import *
from .utils import *
from src.losses.loss import FeatureBalanceLoss
from src.clients import FedBalancerClient, AggregationBalancer


@ray.remote(max_calls=1)
def train(client: FedBalancerClient, training_settings: dict, num_of_classes: int):
    device = "cuda" if torch.cuda.is_available() is True else "cpu"

    # INFO: Unblock the code if you use M1 GPU
    # device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')

    summary_writer = SummaryWriter(os.path.join(client.summary_path, "summaries"))
    csvfile = open(os.path.join(client.summary_path, "experiment_result.csv"), "a", newline='')
    csv_writer = csv.writer(csvfile)

    # INFO - Call the model architecture and set parameters.
    model = model_call(training_settings['model'], num_of_classes, features=True)
    model.load_state_dict(client.model)
    model = model.to(device)

    model_g = model_call(training_settings['model'], num_of_classes, features=False)
    model_g.load_state_dict(client.model)
    model_g = model_g.to(device)
    model_g.eval()

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
    loss_fn = FeatureBalanceLoss(training_settings['global_epochs'],
                                 client.num_per_class,
                                 len(client.global_iter)).to(device)

    training_acc, training_losses = 0, 0
    test_acc, test_losses = 0, 0

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

            model_g.train()
            model_g.to(device)
            global_outputs = model_g(inputs)

            loss = loss_fn(outputs, global_outputs, labels, feature_map, i)

            loss.backward()
            optim.step()

        client.epoch_counter += 1
        training_acc, training_losses = F.compute_accuracy(model, client.train_loader, loss_fn, global_model=model_g)
        test_acc, test_losses = F.compute_accuracy(model, client.test_loader, loss_fn, global_model=model_g)

    cos_similarity = torch.nn.CosineSimilarity(dim=-1)
    model_state = model.state_dict()
    model_g_state = model_g.state_dict()

    for key, value in model_state.items():
        flatten_model = value.view(-1)
        flatten_g_model = model_g_state[key].view(-1)
        similarity = cos_similarity(flatten_model.cpu(), flatten_g_model.cpu())
        summary_writer.add_histogram("{}/cos_sim".format(key), similarity, len(client.global_iter))

    # INFO - Epoch summary
    summary_writer.add_scalar('acc/train', training_acc, client.epoch_counter)
    summary_writer.add_scalar('loss/train', training_losses, client.epoch_counter)
    summary_writer.add_scalar('acc/test', test_acc, client.epoch_counter)
    summary_writer.add_scalar('loss/test', test_losses, client.epoch_counter)

    csv_writer.writerow([training_acc, training_losses, test_acc, test_losses])

    # INFO - Local model update
    client.model = OrderedDict({k: v.clone().detach().cpu() for k, v in model.state_dict().items()})
    csvfile.close()
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


def aggregation_balancer(clients: List[FedBalancerClient],
                         aggregator: Union[Aggregator, AggregationBalancer],
                         global_lr: float = 1.0, temperature: float = 1.0):

    csvfile = open(os.path.join(aggregator.summary_path, "experiment_result.csv"), "a", newline='')
    csv_writer = csv.writer(csvfile)

    previous_g_model = aggregator.model.state_dict()
    empty_model = OrderedDict((key, []) for key in aggregator.model.state_dict().keys())

    # INFO: Collect the weights from all client in a same layer.
    for k, v in aggregator.model.state_dict().items():
        for client in clients:
            empty_model[k].append(client.model[k])
        empty_model[k] = torch.stack(empty_model[k])

    # importance_score = client_importance_score(empty_model['classifier.2.weight'], 'cos', previous_g_model['classifier.2.weight'])
    # importance_score = client_importance_score(empty_model['logit.weight'], 'cos', previous_g_model['logit.weight'])
    importance_score = None
    for name, v in empty_model.items():
        if 'features' in name:
            # NOTE: Averaging the Feature extractor.
            empty_model[name] = torch.mean(v, 0) * global_lr
        else:
            # NOTE: FC layer and logit are aggregated with importance score.
            if 'classifier.0.weight' in name:
                importance_score = client_importance_score(empty_model[name],
                                                           'cos',
                                                           previous_g_model[name],
                                                           temperature=temperature)
                score = shape_convert(importance_score, name)
                empty_model[name] = torch.sum(score * v, dim=0) * global_lr
            else:
                empty_model[name] = torch.mean(v, 0) * global_lr
    # for name, v in empty_model.items():
    #     empty_model[name] = torch.mean(v, 0)

    # NOTE: Global model updates
    aggregator.set_parameters(empty_model, strict=True)
    aggregator.global_iter += 1

    aggregator.test_accuracy, accuracy_per_class = aggregator.compute_accuracy()

    aggregator.summary_writer.add_scalar('global_test_acc', aggregator.test_accuracy, aggregator.global_iter)
    aggregator.summary_writer.add_histogram('global_test_acc/per_class', accuracy_per_class, aggregator.global_iter)
    aggregator.summary_writer.add_histogram('aggregation_score', importance_score, aggregator.global_iter)

    csv_writer.writerow(accuracy_per_class)
    csvfile.close()


def client_importance_score(vector, method, global_model, normalize: bool = True, sigma=1, temperature=1.0):
    # weight_vec = vector.view(vector.size()[0], vector.size()[1], -1)
    weight_vec = vector.view(vector.size()[0], -1)

    if method == 'euclidean'.lower():
        g_vector = global_model.view(global_model.size()[0], -1).unsqueeze(0)
        # NOTE: Lower the distance less changes from global
        vector = torch.norm(g_vector - weight_vec, p=2, dim=-1)

        # NOTE: Make distance 0 if distance lower than standard deviation.
        std, mean = torch.std_mean(vector, dim=0)
        threshold = mean - sigma * std
        vector[vector < threshold] = 0

        # NOTE: Squeeze the dimension
        vector = vector.norm(p=2, dim=-1)
        score_vector = torch.exp(-vector)

        std, mean = torch.std_mean(score_vector, dim=0)
        threshold = mean + sigma * std
        score_vector[score_vector > threshold] = 0

    elif method == 'cos'.lower():
        # g_vector = global_model.view(global_model.size()[0], -1).unsqueeze(0)
        g_vector = global_model.view(-1).unsqueeze(0)
        cos_similarity = torch.nn.CosineSimilarity(dim=-1)

        # NOTE: More similar large similarity value -> large similarity means small changes occurs from global model.
        similarity = cos_similarity(g_vector, weight_vec)

        # NOTE: Clipping the value if lower than threshold
        std, mean = torch.std_mean(similarity, dim=-1)
        threshold = mean - sigma * std
        similarity[similarity < threshold] = threshold

        # NOTE: Projection
        score_vector = torch.exp(similarity)
        # print("vector: ", score_vector)
    else:
        raise NotImplementedError('Method {} is not implemented'.format(method))

    if normalize:
        T = temperature
        score_vector = torch.softmax(score_vector/T, dim=0)
    return score_vector


def shape_convert(score, layer):
    if 'bias' in layer:
        return score.unsqueeze(-1)
    if 'features' in layer:
        return score.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    elif 'classifier' in layer:
        return score.unsqueeze(-1).unsqueeze(-1)
    else:
        return score


# NOTE: This Avg function is only for benchmark.
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


def run(client_setting: dict, training_setting: dict):
    stream_logger, _ = get_logger(LOGGER_DICT['stream'])
    summary_logger, _ = get_logger(LOGGER_DICT['summary'])

    # INFO - Dataset creation
    fed_dataset, valid_loader, test_loader = data_preprocessing(client_setting)

    # INFO - Client initialization
    client = FedBalancerClient

    if training_setting['balancer'] is True:
        aggregator: type(AggregationBalancer) = AggregationBalancer
    else:
        aggregator = Aggregator

    clients, aggregator = client_initialize(client, aggregator,
                                            fed_dataset, test_loader, valid_loader,
                                            client_setting, training_setting)
    start_runtime = time.time()
    # INFO - Training Global Steps
    try:
        stream_logger.info("[4] Global step starts...")

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
                    training_setting['local_lr'] = 1/2*initial_lr*(1+math.cos(aggregator.global_iter*math.pi/total_g_epochs))
                    stream_logger.debug("[*] Learning rate decay: {}".format(training_setting['local_lr']))
                    summary_logger.info("[{}/{}] Current local learning rate: {}".format(aggregator.global_iter,
                                                                                         total_g_epochs,
                                                                                         training_setting['local_lr']))
                else:
                    raise NotImplementedError("Learning rate decay \'{}\' is not implemented yet.".format(
                        training_setting['lr_decay']))

            # INFO - Local Training
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
    summary_logger.info("Experiment finished.")
    stream_logger.info("Experiment finished.")
