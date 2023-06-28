from collections import OrderedDict

import torch
from torch import Tensor

from src.methods import *
from .utils import *


@ray.remote(max_calls=1)
def train(
        client: Client,
        training_settings: dict,
        num_of_classes: int,
        experiment_name: str,
        early_stopping: bool = False):
    device = "cuda" if torch.cuda.is_available() is True else "cpu"

    # INFO: Unblock the code if you use M1 GPU
    # device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')

    summary_writer = SummaryWriter(os.path.join(client.summary_path, "summaries"))

    # INFO - Call the model architecture and set parameters.
    model = model_call(training_settings['model'], num_of_classes, threshold=0)
    model.load_state_dict(client.model)
    model = model.to(device)

    # model_g = model_call(training_settings['model'], num_of_classes)
    # model_g.load_state_dict(client.model)
    # model_g = model_g.to(device)

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

    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    # model_state_dict = filter_pruning(model.state_dict(), model.activation_counts, pruning_rate=0.2)
    # model.load_state_dict(model_state_dict)
    # model.fc.requires_grad_(False)

    # INFO: Local training logic
    for _ in range(training_settings['local_epochs']):
        model.init_activation_counter()
        # INFO: Training steps
        for x, y in client.train_loader:
            inputs = x.to(device)
            labels = y.to(device)

            model.train()
            model.to(device)

            optim.zero_grad()

            outputs = model(inputs)

            # model_g.train()
            # model_g.to(device)
            # global_outputs = model_g(inputs)

            # one_hot = F.one_hot_encode(labels.detach(), num_classes=num_of_classes, device=device)

            # cross_entropy_loss = loss_fn(outputs, labels)
            # outputs_kl_loss = F.loss_fn_kd(outputs, global_outputs.detach(),
            #                                alpha=0.5,
            #                                temperature=training_settings['kl_temp'])

            loss = loss_fn(outputs, labels)
            # loss = cross_entropy_loss + outputs_kl_loss

            loss.backward()
            optim.step()

        model_state_dict = filter_pruning(model.state_dict(), pruning_rate=0.2)
        model.load_state_dict(model_state_dict)
        client.activation_counts = model.activation_count

        client.epoch_counter += 1
        training_acc, training_losses = F.compute_accuracy(model, client.train_loader, loss_fn)
        test_acc, test_losses = F.compute_accuracy(model, client.test_loader, loss_fn)

        # INFO - Epoch summary
        summary_writer.add_scalar('acc/train', training_acc, client.epoch_counter)
        summary_writer.add_scalar('loss/train', training_losses, client.epoch_counter)
        summary_writer.add_scalar('acc/test', test_acc, client.epoch_counter)
        summary_writer.add_scalar('loss/test', test_losses, client.epoch_counter)

        # F.mark_accuracy(model_l=model, model_g=model_g, dataloader=client.train_loader, summary_writer=summary_writer,
        #                 tag='epoch_metric/train_data', epoch=client.epoch_counter)
        #
        # F.mark_accuracy(model_l=model, model_g=model_g, dataloader=client.test_loader, summary_writer=summary_writer,
        #                 tag='epoch_metric/test_data', epoch=client.epoch_counter)

    # INFO - Local model update
    client.model = OrderedDict({k: v.clone().detach().cpu() for k, v in model.state_dict().items()})

    return client


# NOTE: Codebook may not be necessary if we prune the fc layer and bias either
def get_codebook(model: OrderedDict):
    codebook = list()
    for k, v in model.items():
        if 'features' in k and 'weight' in k:
            codebook.append(k)
        elif 'fc' in k and 'weight' in k:
            codebook.append(k)
    return codebook


def filter_pruning(model: OrderedDict, pruning_rate: float, activation_counts: Optional[OrderedDict] = None) -> OrderedDict:
    filter_index = None
    fc_index = None
    for name in model.keys():
        if 'features' in name:
            if filter_index is None:
                filter_pruned_num = int(model[name].size()[0] * pruning_rate)
                weight_vec = model[name].view(model[name].size()[0], -1)
                # NOTE: l1_norm also need to be tested.
                norm_vector = weight_vec.norm(p=2, dim=1).cpu().numpy()
                # norm_vector = weight_vec.norm(p=1, dim=1).cpu().numpy()
                filter_index = norm_vector.argsort()[:filter_pruned_num]
                for index in filter_index:
                    model[name][index] = torch.zeros_like(model[name][index])
            else:
                for index in filter_index:
                    model[name][index] = torch.zeros_like(model[name][index])
                filter_index = None
        elif 'fc' in name:
            # INFO: FC layers
            if activation_counts:
                if fc_index is None:
                    # NOTE: Pruning never activated neuron
                    # zero_indices = torch.squeeze(torch.nonzero(activation_counts[name] == 0))
                    # fc_index = zero_indices
                    # NOTE: Pruning with pruning rate
                    pruning_indices = int(len(activation_counts[name]) * pruning_rate)
                    _, indices = activation_counts[name].sort(axis=0)
                    fc_index = indices[:pruning_indices]
                    # NOTE: Pruning with std
                    # std = torch.std(activation_counts[name])
                    # indices = torch.squeeze(torch.nonzero(activation_counts[name] <= std))
                    # fc_index = indices
                    model[name][fc_index] = 0
                else:
                    model[name][fc_index] = 0
                    fc_index = None
        else:
            # INFO: Logit layer
            continue
    return model


def local_training(clients: list,
                   training_settings: dict,
                   num_of_class: int,
                   experiment_name: str) -> list:
    """
    Args:
        clients: (list) client list to join the training.
        training_settings: (dict) Training setting dictionary
        num_of_class: (int) Number of classes
        experiment_name: (str) Experiment name
    Returns: (List) Client Object result

    """
    ray_jobs = []
    for client in clients:
        if training_settings['use_gpu']:
            ray_jobs.append(train.options(num_gpus=training_settings['gpu_frac']).remote(client,
                                                                                         training_settings,
                                                                                         num_of_class,
                                                                                         experiment_name))
        else:
            ray_jobs.append(train.options().remote(client,
                                                   training_settings,
                                                   num_of_class,
                                                   experiment_name))
    trained_result = []
    while len(ray_jobs):
        done_id, ray_jobs = ray.wait(ray_jobs)
        trained_result.append(ray.get(done_id[0]))

    return trained_result


def fed_cat(clients: List[FedCat_Client], aggregator: Aggregator, global_lr: float, model_save: bool = False):
    empty_model = OrderedDict((key, []) for key in aggregator.model.state_dict().keys())
    empty_activation_counter = OrderedDict((key, []) for key in aggregator.model.state_dict().keys())

    # INFO: Collect the weights from all client in a same layer.
    for k, v in aggregator.model.state_dict().items():
        for client in clients:
            empty_model[k].append(client.model[k])
        empty_model[k] = torch.stack(empty_model[k])

    # INFO: Collect the activation counter
    for k, v in aggregator.model.state_dict().items():
        if 'fc' in k and 'weight' in k:
            for client in clients:
                empty_activation_counter[k].append(client.activation_counts[k])
            empty_activation_counter[k] = torch.stack(empty_activation_counter[k])

    features_indices = None
    fc_indices = None

    for name, v in empty_model.items():
        if 'features' in name:
            if features_indices is None:
                # INFO: To have client index who has largest norm distance.
                weight_vec = v.view(v.size()[0], v.size()[1], -1)
                norm_vector = weight_vec.norm(p=2, dim=2)
                # INFO: Select the client who has largest norm distance
                _, indices = norm_vector.sort(axis=0, descending=True)
                features_indices = indices[0]
                # INFO: (Client, out_channel, in_channel, H, W) -> (out_channel, Client, in_channel, H, W)
                weights = v.permute((1, 0, 2, 3, 4))
                selected_tensor = []
                # INFO: Iterate every output channel
                for i, w in enumerate(weights):
                    # INFO: Put the output channel from selected client.
                    # INFO: 각 output 채널에서 norm vector distance가 가장 큰 클 라이언트의 가중치(weight)와 편향(bias)를 반영
                    selected_tensor.append(w[features_indices[i]])
                selected = torch.stack(selected_tensor)
                empty_model[name] = selected
            else:
                # INFO: Same mechanism for bias
                weights = v.permute((1, 0))
                selected_tensor = []
                for i, w in enumerate(weights):
                    selected_tensor.append(w[features_indices[i]])
                selected = torch.stack(selected_tensor)
                empty_model[name] = selected
                features_indices = None

        # elif 'fc' in name:
        #     if fc_indices is None:
        #         # INFO: empty_activation_counter[name] := (Client, Number of Neurons)
        #         counter_vector = empty_activation_counter[name]
        #         _, indices = counter_vector.sort(axis=0, descending=True)
        #         fc_indices = indices[-1]
        #
        #         # INFO: v:= (Client, Output channel, Input channel) -> (Output channel, Client, Input channel)
        #         weights = v.permute((1, 0, 2))
        #         selected_tensor = []
        #         for i, w in enumerate(weights):
        #             selected_tensor.append(w[fc_indices[i]])
        #         selected = torch.stack(selected_tensor)
        #         empty_model[name] = selected
        #     else:
        #         # INFO: Same for bias
        #         weights = v.permute((1, 0))
        #         selected_tensor = []
        #         for i, w in enumerate(weights):
        #             selected_tensor.append(w[fc_indices[i]])
        #         selected = torch.stack(selected_tensor)
        #         empty_model[name] = selected
        #         fc_indices = None
        else:
            # NOTE: Averaging the logit.
            # print("Name: {}, Size: {}".format(name, v.size()))
            empty_model[name] = torch.mean(v, 0)
            # print("Name: {}, Size: {}".format(name, empty_model[name].size()))


    # # INFO: Original FedAvg
    # for client in clients:
    #     total_len += client.data_len()
    #
    # for k, v in aggregator.model.state_dict().items():
    #     for client in clients:
    #         if k not in empty_model.keys():
    #             empty_model[k] = client.model[k] * (client.data_len() / total_len) * global_lr
    #         else:
    #             empty_model[k] += client.model[k] * (client.data_len() / total_len) * global_lr

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


def run(client_setting: dict, training_setting: dict, experiment_name: str,
        b_save_model: bool = False, b_save_data: bool = False):
    stream_logger, _ = get_logger(LOGGER_DICT['stream'])
    summary_logger, _ = get_logger(LOGGER_DICT['summary'])

    # INFO - Dataset creation
    fed_dataset, valid_loader, test_loader = data_preprocessing(client_setting)

    # INFO - Client initialization
    client = Client
    clients, aggregator = client_initialize(client, fed_dataset, test_loader, valid_loader,
                                            client_setting, training_setting)
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
            trained_clients = local_training(clients=sampled_clients,
                                             training_settings=training_setting,
                                             num_of_class=NUMBER_OF_CLASSES[client_setting['dataset'].lower()],
                                             experiment_name=experiment_name)
            stream_logger.debug("[*] Federated aggregation scheme...")
            fed_cat(trained_clients, aggregator, training_setting['global_lr'])
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
    finally:
        wandb.finish()

    end_run_time = time.time()
    summary_logger.info("Global Running time: {:.2f}".format(end_run_time - start_runtime))

    # INFO - Save client's data
    if b_save_data:
        save_data(clients)
        aggregator.save_data()

    summary_logger.info("Experiment finished.")
    stream_logger.info("Experiment finished.")
