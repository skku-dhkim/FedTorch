from src.utils import *
from src.utils.logger import get_file_logger, get_stream_logger
from src.clients.fed_clients import Client
from src.clients.aggregator import Aggregator
from src.train import functions as F

from torch import cuda
from distutils.util import strtobool
from tqdm import tqdm

import argparse
import time
import os
import ray
import traceback


def main():
    # Program settings
    ray.init()
    core_per_ray = args.ray_core

    # Client settings
    client_settings = {
        'num_of_clients': args.n_clients,
        'dirichlet_alpha': args.dirichlet_alpha,
        'dataset': args.dataset
    }

    # Training settings
    train_settings = {
        'model': args.model,
        'optim': args.opt,
        'global_lr': args.global_lr,
        'local_lr': args.local_lr,
        'momentum': args.momentum,
        'local_epochs': args.local_iter,
        'global_iter': args.global_iter,
        'batch_size': args.batch,
        'use_gpu': args.gpu,
        'summary_count': args.summary_count
    }

    # Logging setting meta.
    summary_logger.info("\n\t" + "*" * 15 + " Client Settings " + "*" * 15 +
                        "\n\t Number of clients: {}".format(client_settings["num_of_clients"]) +
                        "\n\t Dirichlet alpha: {}".format(client_settings["dirichlet_alpha"]) +
                        "\n\t Dataset: {}".format(client_settings["dataset"]) +

                        "\n\t" + "*" * 15 + " Training Hyper-parameters " + "*" * 15 +
                        "\n\t Batch size: {}".format(train_settings['batch_size']) +
                        "\n\t Optimizer: {}".format(train_settings['optim']) +
                        "\n\t Momentum: {}".format(train_settings['momentum']) +
                        "\n\t Global Learning rate: {}".format(train_settings['global_lr']) +
                        "\n\t Local Learning rate: {}".format(train_settings['local_lr']) +
                        "\n\t Local epochs: {}".format(train_settings['local_epochs']) +
                        "\n\t Global iteration: {}".format(train_settings['global_iter']) +

                        "\n\t" + "*" * 15 + " Dataset and Models " + "*" * 15 +
                        "\n\t Dataset: {}".format(client_settings['dataset']) +
                        "\n\t Model: {}".format(train_settings['model']) +

                        "\n\t" + "*" * 15 + " Device Settings " + "*" * 15 +
                        "\n\t Core per Ray: {}".format(core_per_ray) +
                        "\n\t Number of Processes: {}".format(os.cpu_count()) +
                        "\n\t GPU Flag: {}".format(train_settings['use_gpu']) +
                        "\n\t GPU Fraction: {}".format(args.gpu_frac))

    stream_logger.debug(f"Process count: {os.cpu_count()}")
    stream_logger.debug(f"GPU Flag: {train_settings['use_gpu']}")
    stream_logger.debug(f"GPU Fraction: {args.gpu_frac}")

    stream_logger.info("Main Logic Started.")

    # INFO - 1. Loading data
    try:
        stream_logger.info("[1] Data preprocessing...")
        start = time.time()
        ######################################################################################################
        fed_dataset, test_loader = dataset_call(client_settings['dataset'],
                                                log_path=log_path,
                                                dirichlet_alpha=client_settings['dirichlet_alpha'],
                                                num_of_clients=client_settings['num_of_clients'])
        ######################################################################################################
        summary_logger.info("Data preprocessing time: {:.2f}".format(time.time() - start))
        system_logger.info("Data preprocessing finished properly.")
    except Exception as e:
        system_logger.error(traceback.format_exc())
        raise Exception(traceback.format_exc())

    # INFO - 2. Client initialization
    try:
        stream_logger.info("[2] Create Client Container...")
        start = time.time()
        ######################################################################################################
        clients = {}
        for _id, data in enumerate(fed_dataset):
            clients[str(_id)] = Client.options(num_cpus=args.ray_core, num_gpus=args.gpu_frac).remote(str(_id),
                                                                                                      args.dataset.lower(),
                                                                                                      data,
                                                                                                      train_settings,
                                                                                                      log_path=log_path)
        summary_logger.info("Client initializing time: {:.2f}".format(time.time() - start))
        system_logger.info("Client container created successfully.")
        ######################################################################################################
    except Exception as e:
        system_logger.error(traceback.format_exc())
        raise Exception(traceback.format_exc())

    # INFO - 3. Create Aggregator (Federated Server)
    try:
        stream_logger.info("[3] Create Aggregator(Federated Server Container)...")
        start = time.time()
        ######################################################################################################
        aggregator = Aggregator(test_loader,
                                args.dataset.lower(),
                                log_path=log_path,
                                train_settings=train_settings)
        ######################################################################################################
        summary_logger.info("Aggregator initializing time: {:.2f}".format(time.time() - start))
        system_logger.info("Aggregation container created successfully.")
    except Exception as e:
        system_logger.error(traceback.format_exc())
        raise Exception(traceback.format_exc())

    start_runtime = time.time()
    # INFO - 4. Training Global Steps
    try:
        stream_logger.info("[4] Global step starts...")

        pbar = tqdm(range(train_settings['global_iter']), desc="Global steps #",
                    postfix={'global_acc': aggregator.test_accuracy})
        for gr in pbar:
            start_time_global_iter = time.time()

            # INFO - (1) Client queue reset (Client selection)
            stream_logger.debug("[*] Model distribution...")
            F.model_distribution(aggregator=aggregator, clients=clients)

            stream_logger.debug("[*] Local training process...")
            F.local_training(clients)

            stream_logger.debug("[*] Model collection...")
            F.model_collection(clients=clients, aggregator=aggregator, with_data_len=True)

            stream_logger.debug("[*] Federated aggregation scheme...")
            # TODO: Call by aggregation scheme in the future.
            aggregator.fedAvg()

            end_time_global_iter = time.time()
            pbar.set_postfix({'global_acc': aggregator.test_accuracy})
            summary_logger.info("Global Running time: {}::{:.2f}".format(gr,
                                                                         end_time_global_iter - start_time_global_iter))
            summary_logger.info("Test Accuracy: {}".format(aggregator.test_accuracy))
        system_logger.info("Global iteration finished successfully.")
    except Exception as e:
        system_logger.info(traceback.format_exc())
        raise Exception(traceback.format_exc())

    end_run_time = time.time()
    summary_logger.info("Experiment finished: {}".format(args.exp_name))
    summary_logger.info("Global Running time: {:.2f}".format(end_run_time - start_runtime))
    system_logger.info("Program finished successfully with experiment name: {}".format(experiment_name))


if __name__ == '__main__':
    # Argument Parser
    parser = argparse.ArgumentParser(description="Federated Learning on Pytorch")

    # GPU settings
    parser.add_argument('--gpu', type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument('--gpu_frac', type=float, default=1.0)

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
    parser.add_argument('--summary_count', type=int, default=10)

    # System settings
    parser.add_argument('--ray_core', type=int, default=1)

    args = parser.parse_args()

    # Log settings
    experiment_name = args.exp_name
    log_path = os.path.join("./logs", "{}".format(experiment_name))
    os.makedirs(log_path, exist_ok=True)

    #   - Logger Settings
    stream_logger = get_stream_logger("{}".format(args.exp_name), args.slog)
    summary_logger = get_file_logger("{}".format(args.exp_name),
                                     os.path.join(log_path, "experiment_summary.log"), args.flog)
    system_logger = get_file_logger("system_logger[{}]".format(args.exp_name),
                                    os.path.join("./logs", "program.log"), args.flog)

    # INFO: Exceptions
    system_logger.info("Program start with experiment name: {}".format(experiment_name))
    if args.gpu:
        if not cuda.is_available():
            system_logger.error("GPU is not detectable.")
            raise ValueError("GPU is not detectable. Check your settings.")

    # INFO: Main starts
    try:
        main()
    except Exception as e:
        system_logger.error(traceback.format_exc())
        raise Exception(traceback.format_exc())
