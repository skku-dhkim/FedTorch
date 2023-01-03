import ray

from src.clients.aggregator import Aggregator


def model_distribution(aggregator: Aggregator, clients: dict) -> None:
    """

    Args:
        aggregator: (Aggregator class)
        clients: (dict) client list of Ray Client Actor.

    Returns: None

    """
    model_weights = aggregator.get_parameters()
    ray.get([client.set_parameters.remote(model_weights) for _, client in clients.items()])


def model_collection(clients: dict, aggregator: Aggregator, with_data_len: bool = False) -> None:
    """

    Args:
        clients: (dict) {str: ray Client Actor}
        aggregator: (Aggregator)
        with_data_len: (bool) Data length return if True.

    Returns: None

    """
    collected_weights = {}
    for k, client in clients.items():
        if with_data_len:
            collected_weights[k] = {'weights': ray.get(client.get_parameters.remote()),
                                    'data_len': ray.get(client.data_len.remote())}
        else:
            collected_weights[k] = ray.get(client.get_parameters.remote())
    aggregator.collected_weights = collected_weights


def local_training(clients: dict) -> None:
    """

    Args:
        clients: (dict) {str: ray Client Actor}

    Returns: None

    """
    ray.get([client.train.remote(early_stopping=True) for _, client in clients.items()])


# New Modification 22.07.20
def local_training_moon(clients: dict) -> None:
    """

    Args:
        clients: (dict) {str: ray Client Actor}

    Returns: None

    """
    ray.get([client.train_moon.remote() for _, client in clients.items()])

