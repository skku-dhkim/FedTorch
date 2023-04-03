from src import *
from src.clients import Aggregator
from torch.nn import Module
from torch.utils.tensorboard import SummaryWriter
from src.train.train_utils import *

import random
import ray
import torch


def model_download(aggregator: Aggregator, clients: dict) -> None:
    """
    Client download the model from the aggregator.
    Args:
        aggregator: (class Aggregator) Model aggregator for federated learning.
        clients: (dict) client list of Client class.
    Returns: None
    """
    model_weights = aggregator.get_parameters()
    for k, client in clients.items():
        client.model = model_weights


def model_collection(clients: dict, aggregator: Aggregator, with_data_len: bool = False) -> None:
    """
    Clients uploads each model to aggregator.
    Args:
        clients: (dict) {str: Client class} form.
        aggregator: (Aggregator) Aggregator class.
        with_data_len: (bool) Data length return if True.

    Returns: None

    """
    collected_weights = {}
    for k, client in clients.items():
        if with_data_len:
            collected_weights[k] = {'weights': client.get_parameters(),
                                    'data_len': client.data_len()}
        else:
            collected_weights[k] = {'weights': client.get_parameters()}
    aggregator.collected_weights = collected_weights


def compute_accuracy(model: Module, data_loader: DataLoader, loss_fn: Optional[Module] = None) -> Union[float, tuple]:
    """
    Compute the accuracy using its whole data.

    Args:
        model: (torch.Module) Training model.
        data_loader: (torch.utils.Dataloader) Dataloader.
        loss_fn: (torch.Module) Optional. Loss function.

    Returns: ((float) accuracy, (float) loss)

    """
    device = "cpu"

    model.to(device)
    model.eval()

    correct = []
    loss_list = []
    total_len = []

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            if loss_fn is not None:
                loss = loss_fn(outputs, y)
                loss_list.append(loss.item())
            y_max_scores, y_max_idx = outputs.max(dim=1)
            correct.append((y == y_max_idx).sum().item())
            total_len.append(len(x))
        acc = sum(correct) / sum(total_len)

        if loss_fn is not None:
            loss = sum(loss_list) / sum(total_len)
            return acc, loss
        else:
            return acc


def client_sampling(clients: dict, sample_ratio: float, global_round: int) -> list:
    """
    Sampling the client from total clients.
    Args:
        clients: (dict) Clients dictionary that have all the client instances.
        sample_ratio: (float) Sample ration.
        global_round: (int) Current global round.

    Returns: (list) Sample 'client class'

    """
    sampled_clients = random.sample(list(clients.values()), k=int(len(clients.keys()) * sample_ratio))
    for client in sampled_clients:
        # NOTE: I think it is purpose to check what clients are joined corresponding global iteration.
        client.global_iter.append(global_round)
    return sampled_clients


def update_client_dict(clients: dict, trained_client: list) -> dict:
    """
    Updates the client dictionary. Only trained client are updated.
    Args:
        clients: (dict) Clients dictionary.
        trained_client: (list) Trained clients.

    Returns: (dict) Clients dictionary.

    """
    for client in trained_client:
        clients[client.name] = client
    return clients


def mark_accuracy(model_l: Module, model_g: Module, dataloader: DataLoader,
                  summary_writer: SummaryWriter, tag: str, epoch: int) -> None:
    """
    Accuracy mark for experiment.
    Args:
        model_l: (torch.Module) Local model.
        model_g: (torch.Module) Global model.
        dataloader: (DataLoader) Client's dataloader.
        summary_writer: (SummaryWriter class) SummaryWriter instance.
        tag: (str) Summary tag.
        epoch: (str) Client epochs.

    Returns: (None)

    """
    device = "cpu"

    model_l.to(device)
    model_g.to(device)

    model_l.eval()
    model_g.eval()

    accuracy_l = compute_accuracy(model_l, dataloader)
    accuracy_g = compute_accuracy(model_g, dataloader)

    summary_writer.add_scalar('{}/accuracy/local_model'.format(tag), accuracy_l, epoch)
    summary_writer.add_scalar('{}/accuracy/global_model'.format(tag), accuracy_g, epoch)


def mark_entropy(model_l: Module, model_g: Module, dataloader: DataLoader,
                 summary_writer: SummaryWriter, epoch: int) -> None:
    """
    Mark the entropy and entropy gap from certain dataloader.
    Args:
        model_l: (torch.Module) Local Model
        model_g: (torch.Module) Global Model
        dataloader: (DataLoader) Dataloader either train or test.
        summary_writer: (SummaryWriter) SummaryWriter object.
        epoch: (int) Current global round.

    Returns: (None)

    """
    device = "cpu"
    model_l.to(device)
    model_g.to(device)

    model_l.eval()
    model_g.eval()

    output_entropy_l_list = []
    output_entropy_g_list = []
    feature_entropy_l_list = []
    feature_entropy_g_list = []

    output_gap_list = []
    feature_gap_list = []

    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            outputs_l = model_l(x)
            outputs_g = model_g(x)

            features_l = model_l.feature_maps(x)
            features_g = model_g.feature_maps(x)

            # INFO: Calculates entropy gap
            # INFO: Output
            # Probability distributions
            output_prob_l = get_probability(outputs_l, logit=True)
            output_prob_g = get_probability(outputs_g, logit=True)

            # INFO: Output entropy
            output_entr_l = entropy(output_prob_l, base='exp')
            output_entr_g = entropy(output_prob_g, base='exp')

            # Insert the original output entropy value
            output_entropy_l_list.append(torch.mean(output_entr_l))
            output_entropy_g_list.append(torch.mean(output_entr_g))

            # INFO: Calculates the entropy gap
            outputs_entr_gap = torch.abs(output_entr_g - output_entr_l)
            output_gap_list.append(torch.mean(outputs_entr_gap))

            # INFO: Feature
            # Probability distributions
            feature_prob_l = get_probability(features_l)
            feature_prob_g = get_probability(features_g)

            # Calculate the entropy
            feature_entr_l = entropy(feature_prob_l)
            feature_entr_g = entropy(feature_prob_g)

            # Insert the value
            feature_entropy_l_list.append(sum_mean(feature_entr_l))
            feature_entropy_g_list.append(sum_mean(feature_entr_g))

            # INFO: Calculates the entropy gap
            feature_entr_gap = torch.abs(feature_entr_g - feature_entr_l)
            feature_gap_list.append(sum_mean(feature_entr_gap))

    output_entropy_l = torch.Tensor(output_entropy_l_list)
    output_entropy_g = torch.Tensor(output_entropy_g_list)
    feature_entropy_l = torch.Tensor(feature_entropy_l_list)
    feature_entropy_g = torch.Tensor(feature_entropy_g_list)

    summary_writer.add_scalar("entropy/feature/local", torch.mean(feature_entropy_l), epoch)
    summary_writer.add_scalar("entropy/feature/global", torch.mean(feature_entropy_g), epoch)
    summary_writer.add_scalar("entropy/classifier/local", torch.mean(output_entropy_l), epoch)
    summary_writer.add_scalar("entropy/classifier/global", torch.mean(output_entropy_g), epoch)

    output_gap = torch.Tensor(output_gap_list)
    feature_gap = torch.Tensor(feature_gap_list)
    summary_writer.add_scalar("entropy_gap/classifier", torch.mean(output_gap), epoch)
    summary_writer.add_scalar("entropy_gap/feature", torch.mean(feature_gap), epoch)


def mark_norm_gap(model_l: Module, model_g: Module, dataloader: DataLoader,
                  summary_writer: SummaryWriter, epoch: int, norm: int = 1, prob: bool = False) -> None:
    """
    Mark the norm gap from certain dataloader.
    Args:
        model_l: (torch.Module) Local Model
        model_g: (torch.Module) Global Model
        dataloader: (DataLoader) Dataloader either train or test.
        summary_writer: (SummaryWriter) SummaryWriter object.
        epoch: (int) Current global round.
        norm: (int) Level of norm value. Default is 1.
        prob: (bool) To check to use probability form when calculate the norm.

    Returns: (None)

    """
    device = "cpu"
    model_l.to(device)
    model_g.to(device)

    model_l.eval()
    model_g.eval()

    outputs_norm_gap_list = []
    features_norm_gap_list = []

    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            outputs_l = model_l(x)
            outputs_g = model_g(x)

            features_l = model_l.feature_maps(x)
            features_g = model_g.feature_maps(x)

            # INFO: Calculates entropy gap

            # INFO: Output
            # Probability distributions
            if prob:
                outputs_l = get_probability(outputs_l, logit=True)
                outputs_g = get_probability(outputs_g, logit=True)

                features_l = get_probability(features_l)
                features_g = get_probability(features_g)

            # INFO: Calculates norm gap
            outputs_norm_gap = calc_norm(outputs_g - outputs_l, logit=True, p=norm)
            features_norm_gap = calc_norm(features_g - features_l, p=norm)

            outputs_norm_gap_list.append(torch.mean(outputs_norm_gap))
            features_norm_gap_list.append(sum_mean(features_norm_gap))

    outputs_gap = torch.Tensor(outputs_norm_gap_list)
    features_gap = torch.Tensor(features_norm_gap_list)

    if prob:
        summary_writer.add_scalar("norm_gap/l{}-probability/feature".format(norm), torch.mean(outputs_gap), epoch)
        summary_writer.add_scalar("norm_gap/l{}-probability/classifier".format(norm), torch.mean(features_gap), epoch)
    else:
        summary_writer.add_scalar("norm_gap/l{}/feature".format(norm), torch.mean(outputs_gap), epoch)
        summary_writer.add_scalar("norm_gap/l{}/classifier".format(norm), torch.mean(features_gap), epoch)


# TODO: Need to check it is surely works.
def kl_indicator(local_tensor, global_tensor, logit=False, alpha=1):
    # INFO: Calculates entropy gap
    entr_gap = calculate_entropy_gap(local_tensor, global_tensor, logit=logit)

    # INFO: Calculates norm gap
    l1_norm_gap = calculate_norm_gap(local_tensor, global_tensor, logit=logit)

    if logit:
        entr_gap = torch.mean(entr_gap)
        l1_norm_gap = torch.mean(l1_norm_gap)
    else:
        # Mean of feature maps and then batch.
        entr_gap = mean_mean(entr_gap)
        l1_norm_gap = mean_mean(l1_norm_gap)

    indicator = torch.sqrt(l1_norm_gap / (1 + alpha * entr_gap)).detach()
    return torch.nan_to_num(indicator)


# TODO: Need to be fixed.
# New Modification 22.07.20
def local_training_moon(clients: dict) -> None:
    """

    Args:
        clients: (dict) {str: ray Client Actor}

    Returns: None

    """
    ray.get([client.train_moon.remote() for _, client in clients.items()])
