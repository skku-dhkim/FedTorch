from src.utils.data_loader import FedCifar
from src.utils.logger import get_file_logger, get_stream_logger
from src.clients.fed_clients import FedClient, Client
from src.model import model_manager
from src.train.local_trainer import Trainer
from src.train.fed_aggregator import Aggregator
from src import num_of_classes

from torch import cuda
from torch.utils.tensorboard import SummaryWriter
from distutils.util import strtobool
from tqdm import tqdm

import argparse
import psutil
import time
import torch
import os
import ray


if __name__ == '__main__':
    # Argument Parser
    parser = argparse.ArgumentParser(description="Federated Learning on Pytorch")

    # GPU settings
    parser.add_argument('--gpu', type=lambda x: bool(strtobool(x)), default=False)

    # Data settings
    parser.add_argument('--n_clients', type=int, required=True)
    parser.add_argument('--dirichlet_alpha', type=float, default=0.5)
    parser.add_argument('--dataset', type=str, default='Cifar-10')

    # Model settings
    parser.add_argument('--model', type=str, default='Custom_CNN')

    # Training settings
    parser.add_argument('--opt', type=str, default='SGD')
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--local_iter', type=int, default=10)
    parser.add_argument('--global_iter', type=int, default=100)
    parser.add_argument('--local_lr', type=float, default=0.01)
    parser.add_argument('--global_lr', type=float, default=1.0)
    parser.add_argument('--momentum', type=float, default=0.0)

    # Logs settings
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--slog', type=str, default='INFO')
    parser.add_argument('--flog', type=str, default='INFO')

    # System settings
    parser.add_argument('--ray_core', type=int, default=1)

    args = parser.parse_args()

    # Program settings
    ray.init()
    core_per_ray = args.ray_core

    if not args.gpu:
        gpu_flag = False
        num_processes = int(psutil.cpu_count(logical=False))
        device = torch.device('cpu')
    else:
        if cuda.is_available():
            gpu_flag = True
            fraction = 4
            num_processes = cuda.device_count()
            device = torch.device('cuda')
        else:
            # TODO: File logging will be implemented.
            raise ValueError("GPU is not detectable. Check your settings.")

    # Log settings
    experiment_name = args.exp_name
    log_path = os.path.join("./logs", "{}".format(args.exp_name))
    os.makedirs(log_path, exist_ok=True)

    #   - Logger Settings
    stream_logger = get_stream_logger(__name__, args.slog)
    file_logger = get_file_logger(__name__, os.path.join(log_path, "experiment_summary.log"), args.flog)

    #   - Tensorboard summary writer
    summary_path = os.path.join(log_path, "tensorboard/global")
    writer = SummaryWriter(summary_path)

    # Client settings
    number_of_clients = args.n_clients
    dirichlet_alpha = args.dirichlet_alpha
    dataset = args.dataset

    # # Data settings
    # batch_size = args.batch

    # Training settings
    hyper_parameters = {
        'model': args.model,
        'optim': args.opt,
        'global_lr': args.global_lr,
        'local_lr': args.local_lr,
        'momentum': args.momentum,
        'local_epochs': args.local_iter,
        'global_iter': args.global_iter,
        'batch_size': args.batch
    }

    # Logging setting meta.
    file_logger.info("\n\t" + "*" * 15 + " Client Settings " + "*" * 15 +
                     "\n\t Number of clients: {}".format(number_of_clients) +
                     "\n\t Dirichlet alpha: {}".format(dirichlet_alpha) +
                     "\n\t Dataset: {}".format(dataset) +

                     "\n\t" + "*" * 15 + " Training Hyper-parameters " + "*" * 15 +
                     "\n\t Batch size: {}".format(args.batch) +
                     "\n\t Optimizer: {}".format(hyper_parameters['optim']) +
                     "\n\t Momentum: {}".format(hyper_parameters['momentum']) +
                     "\n\t Global Learning rate: {}".format(hyper_parameters['global_lr']) +
                     "\n\t Local Learning rate: {}".format(hyper_parameters['local_lr']) +
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

    stream_logger.info("Main Logic Started.")
    
    # TODO: Main function starts, this function should be wrapped in the future.
    # TODO: Data should call by name in the future.
    # INFO - 1. Loading data
    stream_logger.info("[1] Data preprocessing...")
    start = time.time()
    dataset_object = FedCifar(mode=args.dataset.lower(), log_path=log_path)
    fed_dataset, test_loader = dataset_object.load(number_of_clients, dirichlet_alpha=dirichlet_alpha)
    file_logger.info("Data preprocessing time: {:.2f}".format(time.time() - start))

    # INFO - 2. Client initialization
    stream_logger.info("[2] Create Client Container...")
    start = time.time()
    clients = {}
    for _id, data in enumerate(fed_dataset):
        clients[str(_id)] = Client(str(_id),
                                   {'dataset': args.dataset.lower(), 'data': data},
                                   hyper_parameters, log_path=log_path)
    file_logger.info("Client initializing time: {:.2f}".format(time.time() - start))

    # INFO - 3. Create Aggregator (Federated Server)
    stream_logger.info("[3] Create Aggregator(Federated Server Container)...")
    start = time.time()
    aggregator = Aggregator(test_loader,
                            model_name=hyper_parameters['model'], data_name=args.dataset,
                            global_lr=hyper_parameters['global_lr'],
                            log_path=log_path)
    file_logger.info("Aggregator initializing time: {:.2f}".format(time.time() - start))

    # INFO - 4. Create Trainer
    stream_logger.info("[4] Create Train Container...")
    start = time.time()

    if num_processes >= number_of_clients:
        num_processes = number_of_clients

    if gpu_flag:
        container = [ray.remote(num_gpus=0.25)(Trainer) for _ in range(num_processes)]
    else:
        container = [ray.remote(num_cpus=1)(Trainer) for _ in range(num_processes)]

    ray_train_container = [container.remote(log_path, aggregator.global_model, test_loader) for container in container]
    # [container.set_model.remote(aggregator.global_model) for container in ray_train_container]
    file_logger.info("Trainer initializing time: {:.2f}".format(time.time() - start))

    # INFO - 5. Training Global Steps
    # TODO: Global training function should be implemented separately for the future.
    stream_logger.info("[5] Global step starts...")
    for gr in tqdm(range(hyper_parameters['global_iter']), desc="Global steps #"):
        start = time.time()
        clients_done = []
        unfinished = []

        # INFO - (1) Client queue reset (Client selection)
        client_queue = [key for key in clients.keys()]

        stream_logger.debug("[*] Local training process...")
        while client_queue:
            # INFO - (2) Assign Trainers to client
            for core in ray_train_container:
                try:
                    client = clients[client_queue.pop()]
                    # INFO - (3) Set global weights
                    client.set_parameters(aggregator.global_model.parameters())
                    # INFO - (4) Train with each client data
                    result_id = core.train.remote(client, device)
                    unfinished.append(result_id)
                except IndexError:
                    # NOTE: Pass, if client queue is empty.
                    pass

        stream_logger.debug("[**] Waiting result...")
        # INFO - (5) Ray get method; get weights after training.
        while unfinished:
            finished, unfinished = ray.wait(unfinished)
            clients_done += ray.get(finished)
        clients = {client.name: client for client in clients_done}

        stream_logger.debug("[***]FedAveraging...")

        # INFO - (6) FedAvg
        aggregator.fedAvg([(client.get_parameters(), client.data_len()) for k, client in clients.items()])
        # # TODO: Testing
        # if gr == 0:
        #     aggregator.fedCat([client.model for k, client in clients.items()], classifier=False)
        #     [core.set_model.remote(aggregator.global_model) for core in ray_train_container]
        #     for key in clients.keys():
        #         clients[key].training_settings['local_epoch'] = 10
        # else:
        #     aggregator.fedConCat([client.model for k, client in clients.items()], False)

        stream_logger.debug("[****]Evaluation...")
        # (7) Calculate Evaluation
        accuracy = aggregator.evaluation()

        # Logging
        stream_logger.info("Global accuracy: %2.2f %%" % accuracy)
        file_logger.info(
            "[Global Round: {}/{}] Accuracy: {:2.2f}%".format(gr+1, hyper_parameters['global_iter'], accuracy))
        file_logger.info("Global iteration time: {:2.3f}".format(time.time() - start))
        stream_logger.info("Global iteration time: {:2.3f}".format(time.time() - start))
