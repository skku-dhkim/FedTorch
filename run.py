from src.utils.data_loader import FedCifar
from src.utils.logger import get_file_logger, get_stream_logger
from src.clients.fed_clients import FedClient
from src.model import model_manager
from src.train.local_trainer import Trainer
from src.train.fed_aggregator import Aggregator

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

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

if __name__ == '__main__':
    # NOTE: Argument Parser
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
    parser.add_argument('--slog', type=str, default='DEBUG')
    parser.add_argument('--flog', type=str, default='INFO')

    # System settings
    parser.add_argument('--ray_core', type=int, default=1)

    args = parser.parse_args()

    # Program settings
    ray.init()
    core_per_ray = args.ray_core

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

    # Data settings
    batch_size = args.batch

    # Training settings
    hyper_parameters = {
        'model': args.model,
        'optim': args.opt,
        'global_lr': args.global_lr,
        'local_lr': args.local_lr,
        'momentum': args.momentum,
        'local_epochs': args.local_iter,
        'global_iter': args.global_iter
    }

    # Logging setting meta.
    file_logger.info("\n\t" + "*" * 15 + " Client Settings " + "*" * 15 +
                     "\n\t Number of clients: {}".format(number_of_clients) +
                     "\n\t Dirichlet alpha: {}".format(dirichlet_alpha) +
                     "\n\t Dataset: {}".format(dataset) +

                     "\n\t" + "*" * 15 + " Training Hyper-parameters " + "*" * 15 +
                     "\n\t Batch size: {}".format(batch_size) +
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

    'Main Here'
    # Loading the data
    # TODO: Data should call by name in the future.
    stream_logger.info("[1] Data preprocessing...")
    start = time.time()
    fed_dataset = FedCifar(mode=args.dataset.lower()).load(number_of_clients, dirichlet_alpha=dirichlet_alpha)
    file_logger.info("Data preprocessing time: {:.2f}".format(time.time() - start))

    # Client initialization
    stream_logger.info("[2] Create Client Container...")
    start = time.time()
    clients = {}
    global_model = model_manager.get_model(model_name=hyper_parameters['model'])
    for _id, data in enumerate(fed_dataset):
        clients[str(_id)] = FedClient(str(_id), data, batch_size, hyper_parameters)
    file_logger.info("Client initializing time: {:.2f}".format(time.time() - start))

    # Create Aggregator (Federated Server)
    stream_logger.info("[3] Create Aggregator(Federated Server Container)...")
    start = time.time()
    aggregator = Aggregator(fed_dataset[0]['test'],
                            hyper_parameters['model'],
                            global_lr=hyper_parameters['global_lr'],
                            log_path=log_path)
    file_logger.info("Aggregator initializing time: {:.2f}".format(time.time() - start))

    # Create Trainer
    stream_logger.info("[4] Create Train Container...")
    start = time.time()

    # if gpu_flag:
    #     RayTrainer = ray.remote(num_gpus=1)(Trainer)
    # else:
    #     RayTrainer = ray.remote(num_cpus=core_per_ray)(Trainer)

    ray_train_container = [Trainer.remote(log_path, hyper_parameters['model']) for _ in range(num_processes)]
    file_logger.info("Trainer initializing time: {:.2f}".format(time.time() - start))

    # Training Global Steps
    stream_logger.info("[5] Global step starts...")
    for gr in tqdm(range(hyper_parameters['global_iter']), desc="Global steps #"):
        start = time.time()
        clients_done = []
        unfinished = []

        # (1) Client queue reset (Client selection)
        client_queue = [key for key in clients.keys()]

        stream_logger.info("[*] Local training process...")
        while client_queue:
            # (2) Assign Trainers to client
            for core in ray_train_container:
                try:
                    client = clients[client_queue.pop()]
                    # (3) Set global weights
                    client.model = aggregator.global_model.get_weights()
                    # (4) Train with each client data
                    result_id = core.train.remote(client, device)
                    unfinished.append(result_id)
                except IndexError:
                    # NOTE: Pass, if client queue is empty.
                    pass

        stream_logger.info("[**] Waiting result...")
        # (5) Ray get method; get weights after training.
        while unfinished:
            finished, unfinished = ray.wait(unfinished)
            clients_done += ray.get(finished)
        clients = {client.name: client for client in clients_done}

        stream_logger.info("[***]FedAveraging...")
        # (6) FedAvg
        aggregator.fedAvg([client.model for k, client in clients.items()])

        stream_logger.info("[****]Evaluation...")
        # (7) Calculate Evaluation
        accuracy = aggregator.evaluation(device=device)

        # Logging
        stream_logger.info("Global accuracy: %2.2f %%" % accuracy)
        file_logger.info(
            "[Global Round: {}/{}] Accuracy: {:2.2f}%".format(gr+1, hyper_parameters['global_iter'], accuracy))
        file_logger.info("Global iteration time: {}".format(time.time() - start))
        stream_logger.info("Global iteration time: {}".format(time.time() - start))
