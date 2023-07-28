from src import *
from src.utils import *
from src.methods import get_logger, LOGGER_DICT, Aggregator


def data_preprocessing(client_settings: dict) -> Tuple[list, DataLoader, DataLoader]:
    stream_logger, _ = get_logger(LOGGER_DICT['stream'])
    summary_logger, _ = get_logger(LOGGER_DICT['summary'])
    try:
        stream_logger.info("[1] Data preprocessing...")
        start = time.time()

        fed_dataset, valid_loader, test_loader = dataset_call(client_settings['dataset'],
                                                              log_path=LOGGER_DICT['path'],
                                                              dirichlet_alpha=client_settings['dirichlet_alpha'],
                                                              num_of_clients=client_settings['num_of_clients'])

        summary_logger.info("Data preprocessing time: {:.2f}".format(time.time() - start))
        stream_logger.info("Data preprocessing finished properly.")
    except Exception as e:
        system_logger, _ = get_logger(LOGGER_DICT['system'])
        system_logger.error(traceback.format_exc())
        raise Exception(traceback.format_exc())
    return fed_dataset, valid_loader, test_loader


def client_initialize(client, aggregator, fed_dataset: list, valid_loader: DataLoader, test_loader: DataLoader,
                      client_settings: dict, train_settings: dict) -> Tuple[Dict[str, Any], Aggregator]:
    # import ray
    stream_logger, _ = get_logger(LOGGER_DICT['stream'])
    summary_logger, _ = get_logger(LOGGER_DICT['summary'])
    # INFO - Create client instances
    try:
        stream_logger.info("[2] Create Client Container...")
        start = time.time()
        clients = {}
        for _id, data in enumerate(fed_dataset):
            clients[str(_id)] = client(str(_id),
                                       data,
                                       batch_size=train_settings['batch_size'],
                                       log_path=LOGGER_DICT['path'])
            # INFO - Ray serializability check
            # ray.util.inspect_serializability(clients[str(_id)])
        stream_logger.info("Client container created successfully.")
        summary_logger.info("Client initializing time: {:.2f}".format(time.time() - start))
    except Exception as e:
        system_logger, _ = get_logger(LOGGER_DICT['system'])
        system_logger.error(traceback.format_exc())
        raise Exception(traceback.format_exc())

    # INFO - Create Aggregator (Federated Server)
    try:
        stream_logger.info("[3] Create Aggregator(Federated Server Container)...")
        start = time.time()

        aggregator_obj = aggregator(test_loader, valid_loader,
                                    client_settings['dataset'].lower(),
                                    log_path=LOGGER_DICT['path'],
                                    train_settings=train_settings)

        stream_logger.info("Aggregation container created successfully.")
        summary_logger.info("Aggregator initializing time: {:.2f}".format(time.time() - start))
    except Exception as e:
        system_logger, _ = get_logger(LOGGER_DICT['system'])
        system_logger.error(traceback.format_exc())
        raise Exception(traceback.format_exc())

    return clients, aggregator_obj


def save_data(clients: dict):
    import torch
    for client in clients.values():
        save_path = os.path.join(client.summary_path, "client_data.pt")
        torch.save({
            'train': client.train,
            'test': client.test
        }, save_path)


def save_model(clients: dict):
    import torch
    for client in clients.values():
        save_path = os.path.join(client.summary_path, "model")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path = os.path.join(save_path, "global_iter_{}.pt".format(client.global_iter[-1]))
        torch.save({
            'global_epochs': client.global_iter,
            'client_name': client.name,
            'model': client.model
        }, save_path)
