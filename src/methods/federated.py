import ray.remote_function

from src.methods import *
from src.model import NUMBER_OF_CLASSES
from .utils import *
from src.clients import AggregationBalancer, AvgAggregator, FedBalancerClient


def local_training(train: ray.remote_function.RemoteFunction,
                   clients: list,
                   training_settings: dict,
                   num_of_class: int) -> list:
    """
    Args:
        train: (RemoteFunction) Ray train remote function
        clients: (dict) client ID and Object pair
        training_settings: (dict) Training setting dictionary
        num_of_class: (int) Number of classes
    Returns: (List) Client Object result

    """
    ray_jobs = []
    for client in clients:
        if training_settings['use_gpu']:
            ray_jobs.append(train.options(num_cpus=training_settings['cpus'],
                                          num_gpus=training_settings['gpu_frac']).remote(client,
                                                                                         training_settings,
                                                                                         num_of_class))
        else:
            ray_jobs.append(train.options(num_cpus=training_settings['cpus']).remote(client,
                                                                                     training_settings,
                                                                                     num_of_class))
    trained_result = []
    while len(ray_jobs):
        done_id, ray_jobs = ray.wait(ray_jobs)
        trained_result.append(ray.get(done_id[0]))
    return trained_result


def run(client_setting: dict, training_setting: dict, train_fnc: ray.remote_function.RemoteFunction):
    """
    Run the federated learning process including dataset creation, client selection, global iteration etc...
    Args:
        client_setting: (dict) Client settings. (e.g., dataset, number of clients, dirichlet alpha etc...)
        training_setting: (dict) Training settings. (e.g., Local and global iterations, learning rate, etc...)
        train_fnc: (Ray RemoteFunction) Train function of method its own. It runs with multiprocessing mechanism.
    Returns: (None)
    """
    stream_logger, _ = get_logger(LOGGER_DICT['stream'])
    summary_logger, _ = get_logger(LOGGER_DICT['summary'])

    # INFO - Dataset creation
    fed_dataset, valid_loader, test_loader = data_preprocessing(client_setting)

    # INFO - Client initialization
    if training_setting['aggregator'].lower() == 'balancer':
        _aggregator: type(AggregationBalancer) = AggregationBalancer
        client = FedBalancerClient
    else:
        _aggregator = AvgAggregator
        client = Client

    clients, aggregator = client_initialize(client, _aggregator,
                                            fed_dataset, test_loader, valid_loader,
                                            client_setting,
                                            training_setting)
    start_runtime = time.time()

    # INFO - Training Global Steps
    try:
        stream_logger.info("[4] Global step starts...")

        pbar = tqdm(range(training_setting['global_epochs']), desc="Global steps #",
                    postfix={'global_acc': aggregator.test_accuracy})

        # INFO - Learning rate decay init.
        lr_decay = False
        if 'lr_decay' in training_setting.keys():
            initial_lr = training_setting['local_lr']
            total_g_epochs = training_setting['global_epochs']
            lr_decay = True

        # best_accuracy = aggregator.best_acc
        accuracy_marker = []

        for gr in pbar:
            start_time_global_iter = time.time()

            # # INFO - Save the global model
            # aggregator.save_model()

            # INFO - Download the model from aggregator
            stream_logger.debug("[*] Client downloads the model from aggregator...")
            F.model_download(aggregator=aggregator, clients=clients)

            # INFO - Client sampling
            stream_logger.debug("[*] Client sampling...")
            sampled_clients = F.client_sampling(clients, sample_ratio=client_setting['sample_ratio'], global_round=gr)

            # INFO - Learning rate decay
            if lr_decay:
                if 'cos' in training_setting['lr_decay'].lower():
                    # INFO - COS decay
                    training_setting['local_lr'] = 1 / 2 * initial_lr * (
                                1 + math.cos(aggregator.global_iter * math.pi / total_g_epochs))
                    training_setting['local_lr'] = 0.001 if training_setting['local_lr'] < 0.001 else training_setting['local_lr']
                elif 'manual' in training_setting['lr_decay'].lower():
                    if aggregator.global_iter in [total_g_epochs // 4, (total_g_epochs * 3) // 8]:
                        training_setting['local_lr'] *= 0.1
                else:
                    raise NotImplementedError("Learning rate decay \'{}\' is not implemented yet.".format(
                        training_setting['lr_decay']))

                stream_logger.debug("[*] Learning rate decay: {}".format(training_setting['local_lr']))
                summary_logger.info("[{}/{}] Current local learning rate: {}".format(aggregator.global_iter,
                                                                                     total_g_epochs,
                                                                                     training_setting['local_lr']))

            stream_logger.debug("[*] Local training process...")
            trained_clients = local_training(train_fnc, clients=sampled_clients,
                                             training_settings=training_setting,
                                             num_of_class=NUMBER_OF_CLASSES[client_setting['dataset'].lower()])

            stream_logger.debug("[*] Federated aggregation scheme...")
            aggregator_mode = training_setting['aggregator'].lower()
            if aggregator_mode == 'balancer':
                stream_logger.debug("[*] Aggregation Balancer")
                aggregator.aggregation_balancer(trained_clients,
                                                training_setting['global_lr'],
                                                training_setting['T'],
                                                training_setting['sigma'],
                                                training_setting['inverse'])
            elif aggregator_mode == 'fedavg' or aggregator_mode == 'uniform':
                stream_logger.debug("[*] FedAvg")
                aggregator.fed_avg(trained_clients, training_setting['global_lr'], aggregator_mode)
            else:
                msg = 'Given aggregation scheme does not implemented yet: {}'.format(
                    training_setting['aggregation_scheme'])
                stream_logger.error(msg)
                raise NotImplementedError(msg)

            stream_logger.debug("[*] Weight Updates")
            clients = F.update_client_dict(clients, trained_clients)

            end_time_global_iter = time.time()
            best_result = aggregator.update_test_acc()
            pbar.set_postfix({'global_acc': aggregator.test_accuracy})
            summary_logger.info("Global Running time: {}::{:.2f}".format(gr,
                                                                         end_time_global_iter - start_time_global_iter))
            if best_result:
                # INFO - Save the global model if it has best accuracy
                aggregator.save_model()
                summary_logger.info("Best Test Accuracy: {}".format(aggregator.best_acc))

            accuracy_marker.append(aggregator.test_accuracy)

        accuracy_marker = np.array(accuracy_marker)
        np.savetxt(os.path.join(aggregator.summary_path, "Test_accuracy.csv"), accuracy_marker, delimiter=',')
        summary_logger.info("Global iteration finished successfully.")
    except Exception as e:
        system_logger, _ = get_logger(LOGGER_DICT['system'])
        system_logger.error(traceback.format_exc())
        raise Exception(traceback.format_exc())

    end_run_time = time.time()
    summary_logger.info("Global Running time: {:.2f}".format(end_run_time - start_runtime))
    summary_logger.info("Experiment finished.")
    stream_logger.info("Experiment finished.")
