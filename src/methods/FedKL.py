# from . import *
# from src import *
from src.train import functions as F
from .utils import *
from src.model import NUMBER_OF_CLASSES
import re


def fed_kl(clients: List[Client], aggregator: Aggregator, global_lr: float, model_save: bool = False):
    total_len = 0
    empty_model = OrderedDict()

    # TODO: Consider not to use data length division, use divided per client instead.
    for client in clients:
        total_len += client.data_len()
    regular_exp = re.compile(r'projection_header.*')

    for k, v in aggregator.model.state_dict().items():
        if regular_exp.search(k):
            pass
        else:
            for client in clients:
                if k not in empty_model.keys():
                    empty_model[k] = client.model[k] * (client.data_len()/total_len) * global_lr
                else:
                    empty_model[k] += client.model[k] * (client.data_len()/total_len) * global_lr

    # Global model updates
    aggregator.set_parameters(empty_model, strict=False)
    aggregator.global_iter += 1

    aggregator.test_accuracy = aggregator.compute_accuracy()

    # TODO: Adapt in a future.
    # Calculate cos_similarity with previous representations
    # aggregator.calc_rep_similarity()
    #
    # # Calculate cos_similarity of weights
    # current_model = self.get_parameters()
    # self.calc_cos_similarity(original_model, current_model)
    aggregator.summary_writer.add_scalar('global_test_acc', aggregator.test_accuracy, aggregator.global_iter)

    if model_save:
        aggregator.save_model()


def run(client_setting: dict, training_setting: dict, b_save_model: bool = False, b_save_data: bool = False):
    stream_logger, _ = get_logger(LOGGER_DICT['stream'])
    summary_logger, _ = get_logger(LOGGER_DICT['summary'])

    # INFO - Dataset creation
    fed_dataset, valid_loader, test_loader = data_preprocessing(client_setting)

    # INFO - Client initialization
    clients, aggregator = client_initialize(fed_dataset, test_loader, valid_loader, client_setting, training_setting)

    start_runtime = time.time()
    # INFO - Training Global Steps
    try:
        stream_logger.info("[3] Global step starts...")

        pbar = tqdm(range(training_setting['global_iter']), desc="Global steps #",
                    postfix={'global_acc': aggregator.test_accuracy})

        for gr in pbar:
            start_time_global_iter = time.time()

            # INFO - Save the global model
            aggregator.save_model()

            # INFO - Download the model from aggregator
            stream_logger.debug("[*] Client downloads the model from aggregator...")
            F.model_download(aggregator=aggregator, clients=clients)

            stream_logger.debug("[*] Local training process...")
            # INFO - Normal Local Training
            sampled_clients = F.client_sampling(clients, sample_ratio=training_setting['sample_ratio'], global_round=gr)
            trained_clients = F.local_training(clients=sampled_clients,
                                               training_settings=training_setting,
                                               num_of_class=NUMBER_OF_CLASSES[client_setting['dataset'].lower()])
            stream_logger.debug("[*] Federated aggregation scheme...")
            fed_kl(trained_clients, aggregator, training_setting['global_lr'])
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

