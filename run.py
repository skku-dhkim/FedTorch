from src.utils.logger import get_logger, LOGGER_DICT, write_experiment_summary
from conf.logger_config import STREAM_LOG_LEVEL, SUMMARY_LOG_LEVEL, SYSTEM_LOG_LEVEL

from torch import cuda
from distutils.util import strtobool
from datetime import datetime
from src.methods import FedAvg, FedKL, FedConst, Fedprox, Scaffold, MOON, FedBalancer # FedIndi,

import argparse
import os
import traceback


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
    parser.add_argument('--save_data', type=lambda x: bool(strtobool(x)), default=False)

    # Model settings
    parser.add_argument('--model', type=str, default='Custom_cnn')
    parser.add_argument('--save_model', type=lambda x: bool(strtobool(x)), default=False)

    # Training settings
    parser.add_argument('--opt', type=str, default='SGD')
    parser.add_argument('--batch', type=int, default=50)
    parser.add_argument('--local_iter', type=int, default=5)
    parser.add_argument('--global_iter', type=int, default=50)
    parser.add_argument('--local_lr', type=float, default=0.01)
    parser.add_argument('--global_lr', type=float, default=1.0)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--sample_ratio', type=float, default=1.0)
    parser.add_argument('--T', type=float, default=1.0)
    parser.add_argument('--mu', type=float, default=0.01)
    #parser.add_argument('--var_client', type=list, default=[])

    # Logs settings
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--summary_count', type=int, default=50)

    # System settings
    parser.add_argument('--ray_core', type=int, default=1)

    args = parser.parse_args()

    # INFO: Log settings
    experiment_name = args.exp_name
    log_path = os.path.join("./logs", "{}_{}".format(datetime.today().date(), experiment_name))
    os.makedirs(log_path, exist_ok=True)
    LOGGER_DICT['path'] = log_path

    # INFO:  - Logger Settings
    stream_logger, st_logger_name = get_logger("{}".format(args.exp_name), STREAM_LOG_LEVEL)
    summary_logger, sum_logger_name = get_logger("{}".format(args.exp_name),
                                                 os.path.join(log_path, "experiment_summary.log"),
                                                 SUMMARY_LOG_LEVEL,
                                                 log_type='file')
    system_logger, sys_logger_name = get_logger("system_logger[{}]".format(args.exp_name),
                                                os.path.join("./logs", "system_log_{}.log".format(datetime.today().date())),
                                                SYSTEM_LOG_LEVEL,
                                                log_type='file')

    LOGGER_DICT['stream'] = st_logger_name
    LOGGER_DICT['summary'] = sum_logger_name
    LOGGER_DICT['system'] = system_logger

    stream_logger.info("Program start with experiment name: {}".format(experiment_name))

    # INFO: GPU Exceptions
    if args.gpu:
        if not cuda.is_available():
            system_logger.error("GPU is not detectable.")
            raise ValueError("GPU is not detectable. Check your settings.")

    # INFO: Client settings
    client_settings = {
        'num_of_clients': args.n_clients,
        'dirichlet_alpha': args.dirichlet_alpha,
        'dataset': args.dataset
    }

    # INFO: Training settings
    train_settings = {
        'model': args.model,
        'optim': args.opt,
        'global_lr': args.global_lr,
        'local_lr': args.local_lr,
        'momentum': args.momentum,
        'local_epochs': args.local_iter,
        'global_epochs': args.global_iter,
        'batch_size': args.batch,
        'use_gpu': args.gpu,
        'gpu_frac': args.gpu_frac,
        'summary_count': args.summary_count,
        'sample_ratio': args.sample_ratio,
        'temperature': args.T,
        'weight_decay': 1e-5,
        'kl_temp': 2,
        'indicator_temp': 1,
        'mu': args.mu,
    }

    write_experiment_summary("Client Setting", client_settings)
    write_experiment_summary("Training Hyper-parameters", train_settings)
    write_experiment_summary("Device Settings", {'Core per ray': args.ray_core, 'Num of Processor': os.cpu_count(),
                                                 'GPU Fraction': args.gpu_frac})

    # INFO: Main starts
    try:
        # INFO: Run Function
        # TODO: Make additional Federated method
        # FedKL.run(client_settings, train_settings, b_save_model=args.save_model, b_save_data=args.save_data)
        FedBalancer.run(client_settings, train_settings)
        # FedAD.run(client_settings, train_settings, experiment_name,
        #           b_save_model=args.save_model, b_save_data=args.save_data)
        # FedIndi.run(client_settings, train_settings, b_save_model=args.save_model, b_save_data=args.save_data)
        # FedAvg.run(client_settings, train_settings, b_save_model=args.save_model, b_save_data=args.save_data)
        # Fedprox.run(client_settings, train_settings, b_save_model=args.save_model, b_save_data=args.save_data)
        # FedConst.run(client_settings, train_settings, b_save_model=args.save_model, b_save_data=args.save_data)
        # Scaffold.run(client_settings, train_settings, b_save_model=args.save_model, b_save_data=args.save_data)
        # MOON.run(client_settings, train_settings, b_save_model=args.save_model, b_save_data=args.save_data)

    except Exception as e:
        system_logger.error(traceback.format_exc())
        raise Exception(traceback.format_exc())
