from dataset.data_loader import FedMNIST, FedCifar
from torch.utils.data import DataLoader
from torch import optim, nn, cuda, device
from clients.fed_clients import Client, FedClient
from clients.fed_aggregator import Aggregator
from train.local_trainer import Trainer
from utils.logger import get_file_logger, get_stream_logger
from torch.utils.tensorboard import SummaryWriter
from utils.visualizer import save_client_meta
from conf.logger_config import summary_log_path
from collections import OrderedDict

import argparse
import ray
import psutil
import time
import torch
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

if __name__ == '__main__':
    # NOTE: Argument Parser
    parser = argparse.ArgumentParser(description="Federated Learning on Pytorch")
    parser.add_argument('--gpu', type=bool, default=False)
    args = parser.parse_args()

    # NOTE: Experiment meta
    experiment_name = "debug"
    logging_path = "./logs/{}/experiment_summary.log".format(experiment_name)
    memo = "Short memo for this experiment"

    # NOTE: Client settings
    number_of_clients = 10
    number_of_labels_per_client = 3
    random_distribution = False

    # NOTE: Data settings
    dataset = "Cifar-10"
    batch_size = 64
    shuffle = True

    # NOTE: Training parameters
    hyper_parameters = {
        'model': "Custom_CNN",
        'optim': 'SGD',
        'global_lr': 1,
        'local_lr': 0.01,
        'momentum': 0.9,
        'local_epochs': 10,
        'global_iter': 50
    }

    # NOTE: Program settings
    core_per_ray = 3

    if not args.gpu:
        gpu_flag = False
        num_processes = int(psutil.cpu_count(logical=False) / core_per_ray)
        device = torch.device('cpu')
    else:
        if cuda.is_available():
            gpu_flag = True
            num_processes = cuda.device_count()
            device = torch.device('cuda')

        else:
            gpu_flag = False
            num_processes = int(psutil.cpu_count(logical=False) / core_per_ray)
            device = torch.device('cpu')

    # NOTE: Logger Settings
    stream_logger = get_stream_logger(__name__, "DEBUG")
    file_logger = get_file_logger(__name__, logging_path, "INFO")

    # NOTE: Tensorboard summary writer
    summary_path = "{}/{}".format(summary_log_path, experiment_name)
    writer = SummaryWriter(summary_path + "/global")

    '''
    MAIN Starting from here.
    '''
    # 1. Logging the setting meta.
    # TODO: Add the optimizer into log
    file_logger.info("\n\t" + "*" * 15 + " Experiment Memo " + "*" * 15 +
                     "\n\t {}".format(memo)
                     )

    file_logger.info("\n\t" + "*" * 15 + " Client Settings " + "*" * 15 +
                     "\n\t Number of clients: {}".format(number_of_clients) +
                     "\n\t Number of labels per client: {}".format(number_of_labels_per_client) +
                     "\n\t Random distribution: {}".format(random_distribution) +
                     "\n\t" + "*" * 15 + " Training Hyper-parameters " + "*" * 15 +
                     "\n\t Batch size: {}".format(batch_size) +
                     "\n\t Shuffle: {}".format(shuffle) +
                     "\n\t Global Learning rate: {}".format(hyper_parameters['global_lr']) +
                     "\n\t Local Learning rate: {}".format(hyper_parameters['local_lr']) +
                     "\n\t Momentum: {}".format(hyper_parameters['momentum']) +
                     "\n\t Local epochs: {}".format(hyper_parameters['local_epochs']) +
                     "\n\t Global iteration: {}".format(hyper_parameters['global_iter']) +
                     "\n\t" + "*" * 15 + " Dataset and Models " + "*" * 15 +
                     "\n\t Dataset: {}".format(dataset) +
                     "\n\t Model: {}".format(hyper_parameters['model']) +
                     "\n\t" + "*" * 15 + " Device Settings " + "*" * 15 +
                     "\n\t Core per Ray: {}".format(core_per_ray) +
                     "\n\t Number of Processes: {}".format(num_processes) +
                     "\n\t GPU Flag: {}".format(gpu_flag))

    stream_logger.debug(f"Process count: {num_processes}")
    stream_logger.debug(f"GPU Flag: {gpu_flag}")

    # 2. Loading the data
    # TODO: Data should call by name in the future.
    start = time.time()
    fed_train, test, clients_meta = \
        FedCifar().load(number_of_clients, number_of_labels_per_client, random_dist=random_distribution)

    file_logger.info("\n\t" + "*" * 15 + " Client meta " + "*" * 15 +
                     "\n\t" + str(clients_meta))

    save_client_meta(summary_path, clients_meta)
    file_logger.info("Data preprocessing time: {}".format(time.time() - start))

    # 3. Client initialization
    start = time.time()
    clients = {}
    for client in fed_train:
        clients[client] = FedClient(client, fed_train[client], experiment_name)
    file_logger.info("Client initializing time: {}".format(time.time() - start))

    # 4. Create Aggregator
    aggregator = Aggregator(hyper_parameters['model'], lr=hyper_parameters['global_lr'])
    # model = model_manager.get_model("Resnet-50")
    # model = model_manager.get_model("custom_CNN")

    # 5. Create Trainer
    ray.init()
    if gpu_flag:
        RayTrainer = ray.remote(num_gpus=1)(Trainer)
    else:
        RayTrainer = ray.remote(num_cpus=core_per_ray)(Trainer)

    ray_trainer = [RayTrainer.remote(experiment_name, hyper_parameters['model']) for _ in range(num_processes)]

    # 6. Training Global Steps
    for gr in range(hyper_parameters['global_iter']):
        stream_logger.info("Global round: %d" % gr)

        start = time.time()
        collected_weights = []
        result_ids = []

        # 6-1: Client queue reset
        client_queue = [key for key in clients.keys()]

        while client_queue:
            # 6-2: Assign Trainers to client
            for core in ray_trainer:
                try:
                    client = clients[client_queue.pop()]
                    global_model = aggregator.get_weights(deep_copy=True)
                    # 6-3: Set global weights and train with each client data
                    result_id = core.train.remote(client.name, gr, client.training_loss,
                                                  global_model, client.train_loader,
                                                  hyper_parameters['local_lr'], hyper_parameters['momentum'],
                                                  hyper_parameters['local_epochs'], device)
                    result_ids.append(result_id)
                except IndexError:
                    # NOTE: Pass, if client queue is empty.
                    pass

            # 6-4: Ray get method; get weights after training.
            while len(result_ids):
                done_id, result_ids = ray.wait(result_ids)
                collected_weights.append(ray.get(done_id[0]))

        # 6-5: FedAvg
        aggregator.fedAvg(collected_weights)

        # 6-6: Calculate Evaluation
        accuracy = aggregator.evaluation(test_data=test)

        # NOTE: Logging
        stream_logger.info("Global accuracy: %2.2f %%" % accuracy)
        file_logger.info(
            "[Global Round: {}/{}] Accuracy: {:2.2f}%".format(gr+1, hyper_parameters['global_iter'], accuracy))
        writer.add_scalar('Global Training Accuracy', accuracy, gr)
        file_logger.info("Global iteration time: {}".format(time.time() - start))
        stream_logger.info("Global iteration time: {}".format(time.time() - start))
