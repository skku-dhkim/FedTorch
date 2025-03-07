from src.utils.logger import get_logger, LOGGER_DICT, write_experiment_summary
from conf.logger_config import STREAM_LOG_LEVEL, SUMMARY_LOG_LEVEL, SYSTEM_LOG_LEVEL

from torch import cuda
from distutils.util import strtobool
from datetime import datetime
from src.methods import FedAvg, FedBalancer, FedConst, FedDyn, FedKL, FedNova, FedProx, Scaffold, MOON


import argparse
import os
import traceback


if __name__ == '__main__':
    # Argument Parser
    parser = argparse.ArgumentParser(description="Federated Learning on PyTorch")

    # INFO: Required Parameters
    # Data settings
    parser.add_argument('--n_clients', type=int, required=True)
    parser.add_argument('--dirichlet_alpha', type=float, required=True)

    # Training Settings
    parser.add_argument('--method', type=str, required=True)
    parser.add_argument('--aggregator', type=str, default='FedAvg')
    parser.add_argument('--exp_name', type=str, required=True)

    # INFO: Optional Settings
    # GPU settings
    parser.add_argument('--gpu', type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument('--gpu_frac', type=float, default=1.0)

    # Data settings
    parser.add_argument('--dataset', type=str, default='Cifar-10')
    parser.add_argument('--save_data', type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument('--sample_ratio', type=float, default=1.0)

    # Model settings
    parser.add_argument('--model', type=str, default='Custom_cnn')
    parser.add_argument('--save_model', type=lambda x: bool(strtobool(x)), default=False)

    # Training settings
    parser.add_argument('--opt', type=str, default='SGD')
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--local_iter', type=int, default=10)
    parser.add_argument('--global_iter', type=int, default=100)
    parser.add_argument('--local_lr', type=float, default=0.01)
    parser.add_argument('--global_lr', type=float, default=1.0)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--T', type=float, default=0.5)
    parser.add_argument('--NT', type=float, default=2.0)
    parser.add_argument('--mu', type=float, default=0.01)
    parser.add_argument('--balancer', type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--inverse', type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument('--server_epochs', type=float, default=20)

    # Logs settings
    parser.add_argument('--summary_count', type=int, default=50)

    # System settings
    parser.add_argument('--cpus', type=int, default=1)

    args = parser.parse_args()

    # INFO: Log settings
    experiment_name = args.exp_name
    log_path = os.path.join("./logs", "{}".format(experiment_name))

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
        'dataset': args.dataset,
        'sample_ratio': args.sample_ratio,
    }

    # INFO: Training settings
    train_settings = {
        'method': args.method,
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
        'cpus': args.cpus,
        'summary_count': args.summary_count,
        'T': args.T,
        'NT': args.NT,
        'weight_decay': args.weight_decay,
        'mu': args.mu,
        'aggregator': args.aggregator,
        # 'lr_decay': 'manual',
        'inverse': args.inverse,
        'dyn_alpha': 0.1,
        'server_epochs': args.server_epochs,
        'server_funct': 'exp',
    }

    write_experiment_summary("Client Setting", client_settings)
    write_experiment_summary("Training Hyper-parameters", train_settings)
    write_experiment_summary("Device Settings", {'Core per ray': args.cpus, 'Num of Processor': os.cpu_count(),
                                                 'GPU Fraction': args.gpu_frac})

    # INFO: Main starts
    try:
        # INFO: Run Function
        # TODO: Make additional Federated method
        if 'fedavg' in args.method.lower():
            FedAvg.run_fedavg(client_settings, train_settings)
        elif 'fedbal' in args.method.lower():
            FedBalancer.run_fedbal(client_settings, train_settings)
        elif 'fedconst' in args.method.lower():
            FedConst.run_fedconst(client_settings, train_settings)
        # elif 'fedkl' in args.method.lower():
        #     FedKL.run_fedkl(client_settings, train_settings)
        elif 'fedprox' in args.method.lower():
            FedProx.run_fedprox(client_settings, train_settings)
        elif 'scaffold' in args.method.lower():
            Scaffold.run_scaffold(client_settings, train_settings)
        elif 'fednova' in args.method.lower():
            FedNova.run_fednova(client_settings, train_settings)
        elif 'moon' in args.method.lower():
            MOON.run_moon(client_settings, train_settings)
        elif 'feddyn' in args.method.lower():
            FedDyn.run_feddyn(client_settings, train_settings)
        else:
            raise NotImplementedError("\'{}\' is not implemented method.".format(args.method))

    except Exception as e:
        system_logger.error(traceback.format_exc())
        raise Exception(traceback.format_exc())
    except KeyboardInterrupt:
        print("Terminating process...")
        raise SystemExit()
